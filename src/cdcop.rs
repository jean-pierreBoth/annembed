//! This module implements Carré Du champ operator computations.
//! It builds upon DiffusionMaps tools.
//!
//! Bibliography
//! - *Diffusion Maps*. Coifman Lafon Appl. Comput. Harmon. Anal. 21 (2006) 5–30
//! - *Diffusion Geometry*. Iolo Jones 2024 <https://arxiv.org/abs/2405.10858>
//! - *Bamberger.J Jones.I* Carre du Champ 2025. <https://arxiv.org/abs/2510.05930>
//!

use anyhow::*;
use indexmap::IndexSet;
use ndarray::{Array1, Array2};

use hnsw_rs::prelude::*;

use crate::diffmaps::*;
use crate::graphlaplace::*;
use crate::tools::{matrepr::*, svdapprox::*};

/// just an encapsulation of the (symetric covariance) matrix
pub struct CdcMat(Array2<f32>);

impl CdcMat {
    pub(crate) fn get_array_ref(&self) -> &Array2<f32> {
        &self.0
    }

    pub fn get_trace(&self) -> f32 {
        let (nrow, ncol) = self.0.dim();
        assert_eq!(nrow, ncol);
        (0..nrow).into_iter().map(|i| self.0[[i, i]]).sum::<f32>()
    }

    /// returns spectrum by approximated svd.
    /// If fraction is >= 1, return all eigenvalues computed.  
    /// If fraction < 1 returns eigenvalues up to rank such that sum of largest eigenvlaues exceeds fraction * total trace
    pub fn get_spectrum(&self, info: bool) -> anyhow::Result<Array1<f32>> {
        let matrepr = MatRepr::from_array2(self.0.clone()); // TODO: avoid the clone
        //
        let (nrow, ncol) = self.0.dim();
        assert_eq!(nrow, ncol);
        let dim = nrow;
        let mut svdapprox = SvdApprox::new(&matrepr);
        let precision = RangePrecision::new(0.02, 5, dim);
        let svdmode = RangeApproxMode::EPSIL(precision);
        let svd_res = svdapprox.direct_svd(svdmode).unwrap();
        if let Some(s) = svd_res.get_sigma() {
            let full_trace = self.get_trace();
            let partial_trace = s.sum();
            log::info!(
                "got nb eigenvalues : {}, partial_trace : {:.3e} , full trace : {:.3e}",
                s.len(),
                partial_trace,
                full_trace
            );
            if info {
                spectrum_quantiles(full_trace, s);
            }
            return Ok(s.clone());
        } else {
            log::error!("get_cdc_at_point failed to get s in svd",);
            Err(anyhow!("svd failed"))
        }
        //
    }
}

// simple quantiles reached
fn spectrum_quantiles(full_trace: f32, s: &Array1<f32>) {
    let q: Vec<f32> = vec![0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.];
    let mut partial_trace = 0.;
    let mut reached = 0;
    println!(" lambda index    lambda        trace fraction");
    for i in 0..s.len() {
        partial_trace += s[i];
        let f = partial_trace / full_trace;
        if f >= q[reached] {
            println!("     {:4}        {:.3e}      {:.3e}", i, s[i], f);
            reached += 1
        }
    }
    println!("");
}

/// The structure first computes the transition kernel to neighbours using DiffusionMaps with adhoc parameters.  
/// Then it computes the Covariance of the transition kernel at each asked for point.
/// This provides the best local normal approximation of the data and gives information on the geometry of the data
/// as proved in Bamberger and al.
///   
pub struct CarreDuChamp {
    _params: DiffusionParams,
    //
    glaplacian: Option<GraphLaplacian>,
    // to keep track of rank DataId conversion
    index: Option<IndexSet<DataId>>,
    // We need coordinates to compute cdc. shape is (nb_data , dim)
    data: Array2<f32>,
}

impl CarreDuChamp {
    /// Construct the CarreDuChamp, consuming the Hnsw structure, just keep the data point
    pub fn from_hnsw<T, D>(hnsw: Hnsw<T, D>) -> CarreDuChamp
    where
        T: Copy + Clone + Send + Sync + Into<f32>,
        D: Distance<T> + Send + Sync,
    {
        Self::from_hnsw_ref(&hnsw)
    }

    /// Construct the CarreDuChamp
    pub fn from_hnsw_ref<T, D>(hnsw: &Hnsw<T, D>) -> CarreDuChamp
    where
        T: Copy + Clone + Send + Sync + Into<f32>,
        D: Distance<T> + Send + Sync,
    {
        let mut dparams = DiffusionParams::build_with_variable_bandwidth();
        dparams.set_alfa(0.);
        dparams.set_beta(0.);
        //
        // We need to collect point coordintates. (Cf Kgraph construction)
        //
        let point_indexation = hnsw.get_point_indexation();
        let nb_point = point_indexation.get_nb_point();
        let mut index_set = IndexSet::<DataId>::with_capacity(nb_point);
        let dimension = point_indexation.get_data_dimension();
        let mut data = Array2::<f32>::zeros((nb_point, dimension));
        //
        let point_iter = point_indexation.into_iter();
        for point in point_iter {
            let point_id = point.get_origin_id();
            // remap _point_id
            let (index, _) = index_set.insert_full(point_id);
            let coord = ndarray::ArrayView1::from(point.get_v());
            for i in 0..dimension {
                data.index_axis_mut(ndarray::Axis(0), index)[i] = coord[i].into();
            }
        }
        //
        let mut dmap = DiffusionMaps::new(dparams);
        let laplacian = dmap.laplacian_from_hnsw::<T, D, f32>(hnsw, &dparams);
        CarreDuChamp {
            _params: dparams,
            glaplacian: Some(laplacian),
            index: Some(index_set),
            data,
        }
    }
    //

    /// compute carre du champ at point given its rank. Returns a symetric matrix.
    pub fn get_cdc_at_point(&self, point_rank: usize) -> (Array1<f32>, CdcMat) {
        //
        let glaplacian = self.glaplacian.as_ref().unwrap();
        // compute covariance along data dimension
        let dim = self.data.shape()[1];
        let mut cov = Array2::<f32>::zeros((dim, dim));
        // get list of index conscerned by row point_idx
        let neighbours = glaplacian.get_kernel_row_csvec(point_rank);
        // compute mean
        let mut mean = Array1::<f32>::zeros(dim);
        for (n, proba) in neighbours.iter() {
            for i in 0..dim {
                mean[i] += proba * self.data[[n, i]];
            }
        }
        let mut cumul = 0.;
        for (n, proba) in neighbours.iter() {
            for i in 0..dim {
                for j in 0..=i {
                    cov[[i, j]] +=
                        proba * (self.data[[n, i]] - mean[i]) * (self.data[[n, j]] - mean[j]);
                    cov[[j, i]] = cov[[i, j]];
                }
            }
            cumul += proba;
        }
        // compute trace, eigenvalue and possible renormalization
        let trace = (0..dim).fold(0., |acc, i| acc + cov[[i, i]]);
        log::debug!(" cdc trace at point {}, {:.3e}", point_rank, trace);
        let matrepr = MatRepr::from_array2(cov);
        // consume matrepr and get back array
        let mat = CdcMat(matrepr.retrieve_array().unwrap());
        (mean, mat)
    }

    /// computes distances between cdc operator at 2 different points.
    /// A cpu intensive function ...
    pub fn get_cdc_dist(&self, point_id1: DataId, point_id2: DataId) -> anyhow::Result<f32> {
        let index_ref = self.index.as_ref().unwrap();
        // convert index to rank
        let rank1 = index_ref.get_index_of(&point_id1);
        let rank2 = index_ref.get_index_of(&point_id2);
        if rank1.is_none() {
            return Err(anyhow!("point {} not found in indexset", point_id1));
        }
        if rank2.is_none() {
            return Err(anyhow!("point {} not found in indexset", point_id2));
        }
        let rank1 = rank1.unwrap();
        let rank2 = rank2.unwrap();
        //
        let (_, cov1) = self.get_cdc_at_point(rank1);
        let (_, cov2) = self.get_cdc_at_point(rank2);
        // use psd_dist function
        Ok(psd_dist(&cov1, &cov2))
    }
}

#[cfg_attr(doc, katexit::katexit)]
/// computes the Wasserstein (or Bures) distance between 2 symetric matrices
/// obtained by [CarreDuChamp::get_cdc_at_point()]
/// according to:   
///     *On the Bures–Wasserstein distance between positive definite matrices*
///     See [Bhatia](https://www.sciencedirect.com/science/article/pii/S0723086918300021)
///
/// The distance between 2 symetric matrices A and B is defined by:
/// $$ d(A,B) = \left( tr (A) + tr(B) - 2 \ tr(A^{1/2} B A^{1/2} \right)^{1/2} $$
///
pub fn psd_dist(mata: &CdcMat, matb: &CdcMat) -> f32 {
    //
    let mata = mata.get_array_ref();
    let matb = matb.get_array_ref();
    //
    assert_eq!(mata.shape()[0], matb.shape()[1]);
    //
    let mut tra: f32 = 0.0f32;
    let mut trb: f32 = 0.0f32;
    let mut trab: f32 = 0.0f32;
    //
    for i in 0..mata.shape()[0] {
        tra += mata[[i, i]];
        trb += matb[[i, i]];
        for j in 0..mata.shape()[0] {
            trab += mata[[i, j]] * matb[[j, i]];
        }
    }
    let d2 = tra + trb - 2.0 * trab.sqrt();
    log::debug!("d2 = {:.3e}", d2);
    assert!(d2 >= 0.);
    d2.sqrt()
}

#[cfg(test)]

mod tests {

    use super::*;
    use rand::Rng;
    use rand_distr::Uniform;

    use hnsw_rs::anndists::dist;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_cdc_full() {
        //
        log_init_test();
        println!("\n\n test_cdc_full");
        //
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
        let nbdata = 5000;
        let dim = 256;
        let mut xsi;
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(nbdata);
        for i in 0..nbdata {
            data.push(Vec::<f32>::with_capacity(dim));
            for _ in 0..dim {
                xsi = rng.sample(unif);
                data[i].push(xsi);
            }
        }
        // hnsw
        let ef_construct = 25;
        let nb_connection = 10;
        let hnsw = Hnsw::<f32, dist::DistL1>::new(
            nb_connection,
            nbdata,
            16,
            ef_construct,
            dist::DistL1 {},
        );
        for (i, d) in data.iter().enumerate() {
            hnsw.insert((d, i));
        }
        log::debug!("hnsw built");
        //
        let cdc = CarreDuChamp::from_hnsw_ref(&hnsw);
        let p_5 = 5;
        let (_mean_at_5, cdc_point_5) = cdc.get_cdc_at_point(p_5);
        let info = true;
        let spectrum = cdc_point_5.get_spectrum(info).unwrap();
        log::info!("spectrum at point : {} is : {:?}", p_5, spectrum);
        let p_6 = 6;
        let (_mean_at_6, cdc_point_6) = cdc.get_cdc_at_point(p_6);
        let spectrum = cdc_point_6.get_spectrum(info).unwrap();
        log::info!(
            "spectrum at point : {} , nb eigenvalues: {},  {:?}",
            p_6,
            spectrum.len(),
            spectrum
        );
        //
        let d_5_6 = psd_dist(&cdc_point_5, &cdc_point_6);
        log::info!(
            "cdc distance between points  : {} and {} is : {:?}",
            p_5,
            p_6,
            d_5_6
        );
    }

    // TODO: test with nbdata > 5000 to check csr
    #[test]
    fn test_cdc_csr() {
        println!("\n\n test_cdc_csr");
        std::process::exit(1);
    }
}
