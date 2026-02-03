//! This module implements Carré Du champ operator computations.
//! It builds upon DiffusionMaps tools.
//!
//! Bibliography
//! - *Diffusion Maps*. Coifman Lafon Appl. Comput. Harmon. Anal. 21 (2006) 5–30
//! - *Diffusion Geometry*. Ioo Jones 2024 https://arxiv.org/abs/2405.10858
//! - *Bamberger.J Jones.I Carre du Champ 2025. https://arxiv.org/abs/2510.05930
//!

#![allow(unused)]

use indexmap::IndexSet;
use ndarray::{Array1, Array2, ArrayView};
use num_traits::Float;
use num_traits::cast::FromPrimitive;

use hnsw_rs::prelude::*;

use crate::diffmaps::*;
use crate::fromhnsw::{kgraph::KGraph, kgraph_from_hnsw_all};
use crate::graphlaplace::*;
use crate::tools::{matrepr::*, svdapprox::*};

/// The structure first computes the transition kernel to neighbours using DiffusionMaps with adhoc parameters.  
/// Then it computes the Covariance of the transition kernel at each asked for point.
/// This provides the best local normal approximation of the data and gives information on the geometry of the data
/// as proved in Bamberger and al.
///   
pub struct CarreDuChamp {
    dparams: DiffusionParams,
    //
    glaplacian: Option<GraphLaplacian>,
    // to keep track of rank DataId conversion
    index: Option<IndexSet<DataId>>,
    // We need coordinates to compute cdc
    data: Array2<f32>,
}

fn graph_laplacian_from_hnsw<T, D>(hnsw: &Hnsw<T, D>) -> GraphLaplacian
where
    T: Send + Sync + Clone + Float + FromPrimitive + Into<f32>,
    D: Distance<T> + Send + Sync,
{
    let mut dparams = DiffusionParams::build_with_variable_bandwidth();
    dparams.set_alfa(0.);
    dparams.set_beta(0.);
    //
    let mut dmap = DiffusionMaps::new(dparams);
    dmap.laplacian_from_hnsw::<T, D, f32>(hnsw, &dparams)
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
            let mut coord = ndarray::ArrayView1::from(point.get_v());
            for i in 0..dimension {
                data.index_axis_mut(ndarray::Axis(0), index)[i] = coord[i].into();
            }
        }
        //
        let mut dmap = DiffusionMaps::new(dparams);
        let laplacian = dmap.laplacian_from_hnsw::<T, D, f32>(hnsw, &dparams);
        CarreDuChamp {
            dparams,
            glaplacian: Some(laplacian),
            index: Some(index_set),
            data,
        }
    }
    //

    /// compute carre du champ at point given its rank. Returns a symetric matrix.
    pub fn get_cdc_at_point(&self, point_idx: usize) -> Array2<f32> {
        //
        let glaplacian = self.glaplacian.as_ref().unwrap();
        // retrieve point
        let point_in = self.data.row(point_idx);
        // compute mean
        // check distance between point_in and point_out

        // compute covariance along data dimension
        let dim = self.data.shape()[1];
        let mut cov = Array2::<f32>::zeros((dim, dim));
        // get list of index conscerned by row point_idx
        let neighbours = glaplacian.get_kernel_row_csvec(point_idx);
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
                }
            }
            cumul += proba;
        }
        // compute trace, eigenvalue and possible renormalization
        let trace = (0..dim).fold(0., |acc, i| acc + self.data[[i, i]]);
        log::info!(" cdc trace at point {}, {:.3e}", point_idx, trace);
        let matrepr = MatRepr::from_array2(cov);
        let mut svdapprox = SvdApprox::new(&matrepr);
        let precision = RangePrecision::new(0.1, 5, dim);
        let svdmode = RangeApproxMode::EPSIL(precision);
        let svd_res = svdapprox.direct_svd(svdmode).unwrap();
        log::info!(" cdc spectrum at point {}", point_idx);
        if let Some(s) = svd_res.get_sigma() {
            let dump_size = if log::log_enabled!(log::Level::Debug) {
                dim
            } else {
                20
            };
            let mut i = 0;
            while i < dump_size && s[i] > s[0] / 10. {
                log::info!(" i = {}, s =  {:.3e}", i, s[i]);
                i += 1;
            }
        } else {
            log::error!(
                "get_cdc_at_point failed to get s in svd at point : {}",
                point_idx
            )
        }
        // consume matrepr and get back array
        matrepr.retrieve_array().unwrap()
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
pub fn psd_dist(mata: &Array2<f32>, matb: &Array2<f32>) -> f32 {
    assert_eq!(mata.shape(), matb.shape());
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
            trab += mata[[i, j]] * mata[[j, i]];
        }
    }
    let d2 = tra + trb - 2.0 * trab;
    log::debug!("d2 = {:.3e}", d2);
    assert!(d2 >= 0.);
    d2.sqrt()
}

#[cfg(test)]

mod tests {

    use super::*;
    use rand::Rng;
    use rand_distr::Uniform;

    use cpu_time::ProcessTime;
    use std::time::SystemTime;

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
            for j in 0..dim {
                xsi = rng.sample(unif);
                data[i].push(xsi);
            }
        }
        // hnsw
        let ef_construct = 25;
        let nb_connection = 10;
        let start = ProcessTime::now();
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
        let cdc_point_5 = cdc.get_cdc_at_point(5);
        let cdc_point_6 = cdc.get_cdc_at_point(6);
        //
        let d_5_6 = psd_dist(&cdc_point_5, &cdc_point_6);
    }

    // TODO: test with nbdata > 5000 to check csr
    #[test]
    fn test_cdc_csr() {
        println!("\n\n test_cdc_csr");
        std::process::exit(1);
    }
}
