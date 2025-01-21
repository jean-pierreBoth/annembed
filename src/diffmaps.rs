//!  (Kind of) Diffusion maps embedding.
//!
//! This module (presently) computes a diffusion embedding for the kernel constructed from nearest neighbours
//! stored in a Hnsw structure, see in module [embedder](crate::embedder).  
//! In particular the kernel sets the diagonal to 0 and nearest neighbour weight to 1.
//!
//!

use num_traits::cast::FromPrimitive;
use num_traits::Float;

use std::collections::HashMap;

use quantiles::ckms::CKMS;
use rayon::prelude::*;

use ndarray::{Array1, Array2, Axis};
use sprs::{CsMat, TriMatBase};

use hnsw_rs::prelude::*;

use crate::embedder::*;
use crate::fromhnsw::{kgraph::KGraph, kgraph_from_hnsw_all};
//use crate::fromhnsw::*;
use crate::graphlaplace::*;
use crate::tools::{nodeparam::*, svdapprox::*};

#[derive(Copy, Clone)]
pub struct DiffusionParams {
    /// dimension of embedding
    asked_dim: usize,
    /// exponent of sampling law. By default we use 1/2
    alfa: f32,
    /// embedding time
    t: Option<f32>,
} // end of DiffusionParams

impl DiffusionParams {
    pub fn new(asked_dim: usize, t_opt: Option<f32>) -> Self {
        DiffusionParams {
            asked_dim,
            alfa: 0.5,
            t: t_opt,
        }
    }
    /// get embedding time
    pub fn get_t(&self) -> Option<f32> {
        self.t
    }

    //
    /// modify the default alfa See Lafon paper.
    /// naural values are 0. , 1/2 and 1.
    pub fn set_alfa(&mut self, alfa: f32) {
        if alfa > 1. || alfa < 0. {
            println!("alfa ")
        }
        self.alfa = alfa;
    }

    pub fn get_alfa(&self) -> f32 {
        return self.alfa;
    }

    pub fn get_embedding_dimension(&self) -> usize {
        self.asked_dim
    }
} // end of DiffusionParams

pub struct DiffusionMaps {
    /// parameters to use
    params: DiffusionParams,
    /// node parameters coming from graph transformation
    _node_params: Option<NodeParams>,
    //
    #[allow(unused)]
    laplacian: Option<GraphLaplacian>,
    /// can store svd result
    svd: Option<SvdResult<f32>>,
} // end of DiffusionMaps

impl DiffusionMaps {
    /// iitialization from NodeParams
    pub fn new(params: DiffusionParams) -> Self {
        DiffusionMaps {
            params,
            _node_params: None,
            laplacian: None,
            svd: None,
        }
    }

    /// do the whole work chain :graph conversion from hnsw structure, NodeParams transformation
    /// T is the type on which distances in Hnsw are computed,  
    /// F is f32 or f64 depending on how diffusions Maps is to be computed.
    #[deprecated = "use embed_from_hnsw"]
    pub fn embed_hnsw<T, D, F>(&mut self, hnsw: &Hnsw<T, D>) -> Array2<F>
    where
        D: Distance<T> + Send + Sync,
        T: Clone + Send + Sync,
        F: Float + FromPrimitive + std::marker::Sync + Send + std::fmt::UpperExp + std::iter::Sum,
    {
        //
        let knbn = hnsw.get_max_nb_connection();
        let kgraph = kgraph_from_hnsw_all::<T, D, F>(hnsw, knbn as usize).unwrap();
        // get NodeParams. CAVEAT to_proba_edges apply initial shift!!
        let nodeparams = to_proba_edges::<F>(&kgraph, 1., 2.);
        get_dmap_embedding::<F>(&nodeparams, self.params.asked_dim, self.params.get_t())
    } // end embed_hnsw

    /// Return laplacian from hnsw nearest neighbours.
    /// If store is true, the laplacian is stored in structure DiffusionsMaps for future use
    /// F is float type we want the result in
    #[allow(unused)]
    pub(crate) fn laplacian_from_hnsw<T, D, F>(
        &mut self,
        hnsw: &Hnsw<T, D>,
        store: bool,
    ) -> GraphLaplacian
    where
        T: Clone + Send + Sync,
        D: Distance<T> + Send + Sync,
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::ops::DivAssign
            + Into<f64>,
    {
        let knbn = hnsw.get_max_nb_connection();
        let kgraph = kgraph_from_hnsw_all::<T, D, F>(hnsw, knbn as usize).unwrap();
        // get NodeParams.
        let nodeparams = self.to_dmap_nodeparams::<F>(&kgraph);
        let laplacian = self.compute_laplacian(&nodeparams, self.params.get_alfa());
        if store {
            self.laplacian = Some(laplacian.clone())
        }
        laplacian
    }

    pub(crate) fn compute_laplacian(
        &self,
        initial_space: &NodeParams,
        alfa: f32,
    ) -> GraphLaplacian {
        //
        log::info!(
            "in GraphLaplacian::compute_laplacian, using alfa : {:.2e}",
            alfa
        );
        //
        let nbnodes = initial_space.get_nb_nodes();
        // get stats
        let max_nbng = initial_space.get_max_nbng();
        let node_params = initial_space;
        // TODO define a threshold for dense/sparse representation
        if nbnodes <= FULL_MAT_REPR {
            log::debug!("get_laplacian using full matrix");
            let mut transition_proba = Array2::<f32>::zeros((nbnodes, nbnodes));
            // we loop on all nodes, for each we want nearest neighbours, and get scale of distances around it
            for i in 0..node_params.params.len() {
                // remind to index each request
                let node_param = node_params.get_node_param(i);
                // recall : self.edge are used here (See to_dmap_nodeparams)
                for j in 0..node_param.edges.len() {
                    let edge = node_param.edges[j];
                    transition_proba[[i, edge.node]] = edge.weight;
                } // end of for j
            } // end for i
            log::trace!("full matrix initialized");
            // now we symetrize the graph by taking mean
            // The UMAP formula (p_i+p_j - p_i *p_j) implies taking the non null proba when one proba is null,
            // so UMAP initialization is more packed.
            let mut symgraph = (&transition_proba + &transition_proba.view().t()) * 0.5;
            // now we go to the symetric weighted laplacian D^-1/2 * G * D^-1/2 but get rid of the I - ...
            // cf Yan-Jordan Fast Approximate Spectral Clustering ACM-KDD 2009
            //  compute sum of row and renormalize. See Lafon-Keller-Coifman
            // Diffusions Maps appendix B
            // IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,VOL. 28, NO. 11,NOVEMBER 2006
            //
            let diag = symgraph.sum_axis(Axis(1));
            for i in 0..nbnodes {
                let mut row = symgraph.row_mut(i);
                for j in 0..nbnodes {
                    row[[j]] /= (diag[[i]] * diag[[j]]).powf(alfa);
                }
            }
            //
            log::trace!("\n allocating full matrix laplacian");
            return GraphLaplacian::new(MatRepr::from_array2(symgraph), diag);
        } else {
            log::debug!("Embedder using csr matrix");
            // now we must construct a CsrMat to store the symetrized graph transition probablity to go svd.
            // and initialize field initial_space with some NodeParams
            let mut edge_list = HashMap::<(usize, usize), f32>::with_capacity(nbnodes * max_nbng);
            for i in 0..node_params.params.len() {
                let node_param = node_params.get_node_param(i);
                for j in 0..node_param.edges.len() {
                    let edge = node_param.edges[j];
                    edge_list.insert((i, edge.node), node_param.edges[j].weight);
                } // end of for j
            }
            // now we iter on the hasmap symetrize the graph, and insert in triplets transition_proba
            let mut diagonal = Array1::<f32>::zeros(nbnodes);
            let mut rows = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
            let mut cols = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
            let mut values = Vec::<f32>::with_capacity(nbnodes * 2 * max_nbng);

            for ((i, j), val) in edge_list.iter() {
                assert!(i != j);
                let sym_val;
                if let Some(t_val) = edge_list.get(&(*j, *i)) {
                    sym_val = (val + t_val) * 0.5;
                } else {
                    sym_val = *val;
                }
                rows.push(*i);
                cols.push(*j);
                values.push(sym_val);
                diagonal[*i] += sym_val;
                //
                rows.push(*j);
                cols.push(*i);
                values.push(sym_val);
                diagonal[*j] += sym_val;
            }
            // as in FULL Representation we avoided the I diagnoal term which cancels anyway
            // Now we reset non diagonal terms to D^-1/2 G D^-1/2  i.e  val[i,j]/(D[i]*D[j])^1/2
            for i in 0..rows.len() {
                let row = rows[i];
                let col = cols[i];
                if row != col {
                    values[i] /= (diagonal[row] * diagonal[col]).powf(alfa);
                }
            }
            //
            log::trace!("allocating csr laplacian");
            let laplacian = TriMatBase::<Vec<usize>, Vec<f32>>::from_triplets(
                (nbnodes, nbnodes),
                rows,
                cols,
                values,
            );
            let csr_mat: CsMat<f32> = laplacian.to_csr();
            return GraphLaplacian::new(MatRepr::from_csrmat(csr_mat), diagonal);
        } // end case CsMat
          //
    }

    /// dmap specific edge proba compuatitons
    pub(crate) fn to_dmap_nodeparams<F>(&self, kgraph: &KGraph<F>) -> NodeParams
    where
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::ops::DivAssign
            + Into<f64>,
    {
        let nb_nodes = kgraph.get_nb_nodes();
        let mut nodeparams = Vec::<NodeParam>::with_capacity(nb_nodes);
        //
        let neighbour_hood = kgraph.get_neighbours();
        // compute a scale around each node, mean scale and quantiles on scale
        let scales: Vec<F> = neighbour_hood
            .par_iter()
            .map(|edges| self.get_scales_around_node(edges))
            .collect();
        // collect scales quantiles
        let mut scales_q: CKMS<f64> = CKMS::<f64>::new(0.001);
        let mut local_scales = Vec::<F>::with_capacity(scales.len());
        for s in &scales {
            scales_q.insert((*s).into());
            local_scales.push(*s);
        }

        println!("\n\n dmap scales quantiles at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}",
        scales_q.query(0.05).unwrap().1, scales_q.query(0.5).unwrap().1,
        scales_q.query(0.95).unwrap().1, scales_q.query(0.99).unwrap().1);
        println!();
        // get median of local scales and use it as global scale
        let median_scale: F = F::from(scales_q.query(0.5).unwrap().1).unwrap();
        assert!(median_scale > F::zero());
        // we keep ratio of local scale to median scale for possible kernel weighting
        for s in &mut local_scales {
            *s /= median_scale;
        }
        //
        // now we have scales we can remap edge length to weights.
        // TODO: no shift but could add drift with respect to local_scales variations
        let remap_weight =
            |w: F, scale: f32| (-((w.to_f32().unwrap()).max(0.) / scale).powf(2.)).exp();
        // now we loop on all nodes
        for i in 0..nb_nodes {
            let neighbours = &neighbour_hood[i];
            // get rid of case where all neighbours have dist 0 to current node (It happens in Higgs.data!!!)
            let mut all_equal = false;
            let last_n = neighbours
                .iter()
                .rfind(|&n| n.weight.to_f32().unwrap() > 0.);
            if last_n.is_none() {
                // means all distances are 0! (encountered in Higgs Boson bench)
                all_equal = true;
            } else {
                let last_e_w = last_n.unwrap().weight;
                if last_e_w <= neighbours[0].weight {
                    all_equal = true;
                }
            }
            // we add each node as a neighbour of itself to enfore ergodicity !!
            let nb_edges = 1 + neighbours.len();
            let mut edges = Vec::<OutEdge<f32>>::with_capacity(nb_edges);
            if all_equal {
                // all neighbours will have
                let proba: f32 = 1. / (nb_edges as f32);
                let self_edge = OutEdge::new(i, proba);
                edges.push(self_edge);
                for n in neighbours {
                    edges.push(OutEdge::new(n.node, proba));
                }
            } else {
                let self_edge = OutEdge::<f32>::new(i, 1.);
                edges.push(self_edge);
                let node_scale: f32 = (local_scales[i] * median_scale).to_f32().unwrap();
                let mut sum: f32 = 0.;
                for n in neighbours {
                    let weight: f32 = remap_weight(n.weight, node_scale).max(PROBA_MIN);
                    let edge = OutEdge::<f32>::new(n.node, weight);
                    edges.push(edge);
                    sum += weight;
                }
                // TODO: we adjust self_edge and renormalize
                //                edges[0].weight = edges[1].weight;
                sum += edges[0].weight;
                for e in &mut edges {
                    e.weight /= sum;
                }
            }
            // allocate a NodeParam and keep track of real scale of node
            let nodep = NodeParam::new(local_scales[i].to_f32().unwrap(), edges);
            nodeparams.push(nodep);
        }
        NodeParams::new(nodeparams, kgraph.get_max_nbng())
    } // end to_dmap_nodeparams

    // computes scale (mean dist) around a point
    pub(crate) fn get_scales_around_node<F>(&self, out_edges: &[OutEdge<F>]) -> F
    where
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign,
    {
        let sum = out_edges.iter().map(|e| e.weight).sum::<F>();
        sum / F::from(out_edges.len()).unwrap()
    }

    /// returns gr  ph laplacian if already computed and stored in structure
    #[allow(unused)]
    pub(crate) fn get_laplacian(&mut self) -> Option<&mut GraphLaplacian> {
        self.laplacian.as_mut()
    }

    /// returns svd result computed in dmap embedding
    pub fn get_svd_res(&self) -> Option<&SvdResult<f32>> {
        self.svd.as_ref()
    }

    /// Do the whole work chain :graph conversion from hnsw structure, NodeParams transformation.  
    /// T is the type on which distances in Hnsw are computed,
    /// F is f32 or f64 depending on how diffusions Maps is to be computed.  
    /// The svd result are stored in the DiffusionMaps structure and accessible with the functions
    /// [Self::get_svd_res()]
    pub fn embed_from_hnsw<T, D, F>(
        &mut self,
        hnsw: &Hnsw<T, D>,
        asked_dim: usize,
        t_opt: Option<f32>,
    ) -> Array2<F>
    where
        D: Distance<T> + Send + Sync,
        T: Clone + Send + Sync,
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::ops::DivAssign
            + Into<f64>,
    {
        let store = true;
        if self.get_laplacian().is_none() {
            let _ = self.laplacian_from_hnsw::<T, D, F>(hnsw, store);
        }
        let laplacian = self.get_laplacian().unwrap();
        log::debug!("got laplacian, going to svd ... asked_dim :  {}", asked_dim);
        let svd_res: SvdResult<f32> = laplacian.do_svd(asked_dim + 25).unwrap();
        //
        // As we used a laplacian and probability transitions we eigenvectors corresponding to lower eigenvalues
        let lambdas = svd_res.get_sigma().as_ref().unwrap();
        // singular vectors are stored in decrasing order according to lapack for both gesdd and gesvd.
        if lambdas.len() > 2 && lambdas[1] > lambdas[0] {
            panic!("svd spectrum not decreasing");
        }
        // we examine spectrum
        // our laplacian is without the term I of I-G , we use directly G symetrized so we consider upper eigenvalues
        log::info!(
            " first 3 eigen values {:.2e} {:.2e} {:2e}",
            lambdas[0],
            lambdas[1],
            lambdas[2]
        );
        // get info on spectral gap
        log::info!(
            " last eigenvalue computed rank {} value {:.2e}",
            lambdas.len() - 1,
            lambdas[lambdas.len() - 1]
        );
        //
        log::debug!("keeping columns from 1 to : {}", asked_dim);
        // We get U at index in range first_non_zero-max_dim..first_non_zero
        let u = svd_res.get_u().as_ref().unwrap();
        log::debug!("u shape : nrows: {} ,  ncols : {} ", u.nrows(), u.ncols());
        if u.ncols() < asked_dim {
            log::warn!(
                "asked dimension  : {} svd obtained less than asked for : {}",
                asked_dim,
                u.ncols()
            );
        }
        let real_dim = asked_dim.min(u.ncols());
        // we can get svd from approx range so that nrows and ncols can be number of nodes!
        let mut embedded = Array2::<F>::zeros((u.nrows(), real_dim));
        // according to theory (See Luxburg or Lafon-Keller diffusion maps) we must go back to eigen vectors of rw laplacian.
        // Appendix A of Coifman-Lafon Diffusion Maps. Applied Comput Harmonical Analysis 2006.
        // moreover we must get back to type F
        let normalized_lambdas = lambdas / (*lambdas)[0];
        let time = match t_opt {
            Some(t) => t,
            _ => 5.0f32.min(0.9f32.ln() / (normalized_lambdas[2] / normalized_lambdas[1]).ln()),
        };
        log::info!(
            "DiffusionMaps::embed_from_hnsw applying dmap time {:.2e}",
            time
        );
        let sum_diag = laplacian.degrees.iter().sum::<f32>();
        for i in 0..u.nrows() {
            let row_i = u.row(i);
            let weight_i = (laplacian.degrees[i] / sum_diag).sqrt();
            for j in 0..real_dim {
                // divide j value by diagonal and convert to F. take l_{i}^{t} as in dmap
                embedded[[i, j]] =
                    F::from_f32(normalized_lambdas[j + 1].powf(time) * row_i[j + 1] / weight_i)
                        .unwrap();
            }
        }
        log::debug!("DiffusionMaps::embed_from_hnsw ended");
        //
        self.svd = Some(svd_res.clone());
        //
        embedded
    }
} // end of impl DiffusionsMaps

//=====================================================================================================================

// this function initialize and returns embedding by a svd (or else?)
// We are intersested in first eigenvalues (excpeting 1.) of transition probability matrix
// i.e last non null eigenvalues of laplacian matrix!!
// The time used is the one in argument in t_opt if not None.
// If t_opt is none the time is compute so that $ (\lambda_{2}/\lambda_{1})^t \less 0.9 $
pub(crate) fn get_dmap_embedding<F>(
    initial_space: &NodeParams,
    asked_dim: usize,
    t_opt: Option<f32>,
) -> Array2<F>
where
    F: Float + FromPrimitive,
{
    //
    assert!(asked_dim >= 2);
    // get eigen values of normalized symetric lapalcian
    let mut laplacian = get_laplacian(initial_space);
    //
    log::debug!("got laplacian, going to svd ... asked_dim :  {}", asked_dim);
    let svd_res = laplacian.do_svd(asked_dim + 25).unwrap();
    // As we used a laplacian and probability transitions we eigenvectors corresponding to lower eigenvalues
    let lambdas = svd_res.get_sigma().as_ref().unwrap();
    // singular vectors are stored in decrasing order according to lapack for both gesdd and gesvd.
    if lambdas.len() > 2 && lambdas[1] > lambdas[0] {
        panic!("svd spectrum not decreasing");
    }
    // we examine spectrum
    // our laplacian is without the term I of I-G , we use directly G symetrized so we consider upper eigenvalues
    log::info!(
        " first 3 eigen values {:.2e} {:.2e} {:2e}",
        lambdas[0],
        lambdas[1],
        lambdas[2]
    );
    // get info on spectral gap
    log::info!(
        " last eigenvalue computed rank {} value {:.2e}",
        lambdas.len() - 1,
        lambdas[lambdas.len() - 1]
    );
    //
    log::debug!("keeping columns from 1 to : {}", asked_dim);
    // We get U at index in range first_non_zero-max_dim..first_non_zero
    let u = svd_res.get_u().as_ref().unwrap();
    log::debug!("u shape : nrows: {} ,  ncols : {} ", u.nrows(), u.ncols());
    if u.ncols() < asked_dim {
        log::warn!(
            "asked dimension  : {} svd obtained less than asked for : {}",
            asked_dim,
            u.ncols()
        );
    }
    let real_dim = asked_dim.min(u.ncols());
    // we can get svd from approx range so that nrows and ncols can be number of nodes!
    let mut embedded = Array2::<F>::zeros((u.nrows(), real_dim));
    // according to theory (See Luxburg or Lafon-Keller diffusion maps) we must go back to eigen vectors of rw laplacian.
    // Appendix A of Coifman-Lafon Diffusion Maps. Applied Comput Harmonical Analysis 2006.
    // moreover we must get back to type F
    let normalized_lambdas = lambdas / (*lambdas)[0];
    let time = match t_opt {
        Some(t) => t,
        _ => 5.0f32.min(0.9f32.ln() / (normalized_lambdas[2] / normalized_lambdas[1]).ln()),
    };
    log::info!("get_dmap_initial_embedding applying dmap time {:.2e}", time);
    let sum_diag = laplacian.degrees.iter().sum::<f32>();
    for i in 0..u.nrows() {
        let row_i = u.row(i);
        let weight_i = (laplacian.degrees[i] / sum_diag).sqrt();
        for j in 0..real_dim {
            // divide j value by diagonal and convert to F. take l_{i}^{t} as in dmap
            embedded[[i, j]] =
                F::from_f32(normalized_lambdas[j + 1].powf(time) * row_i[j + 1] / weight_i)
                    .unwrap();
        }
    }
    log::debug!("ended get_dmap_initial_embedding");
    embedded
} // end of get_dmap_initial_embedding

//======================================================================================================================

/// This function runs a parallel insertion of rows of an `Array2<T>` into a  Hnsw<T,D>.  
/// The hnsw structure must have chosen main parameters as the number of connection and layers, but
/// be empty.   
/// Returns number of point inserted if success.
pub fn array2_insert_hnsw<T, D>(data: &Array2<T>, hnsw: &mut Hnsw<T, D>) -> Result<usize, usize>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    //
    if hnsw.get_nb_point() > 0 {
        log::error!(
            "array2_insert_hnsw , insertion on non empty hnsw structure, nb point : {}",
            hnsw.get_nb_point()
        );
        return Err(1);
    }
    // we do parallel insertion by blocks of size blocksize
    let blocksize = 10000;
    let (nb_row, _) = data.dim();

    let nb_block = nb_row / blocksize;
    for i in 0..nb_block {
        let start = i * blocksize;
        let end = i * blocksize + blocksize - 1;
        let to_insert = (start..=end)
            .map(|n| (data.row(n).to_slice().unwrap(), n))
            .collect();
        hnsw.parallel_insert_slice(&to_insert);
    }
    let start = nb_block * blocksize;
    let to_insert = (start..nb_row)
        .map(|n| (data.row(n).to_slice().unwrap(), n))
        .collect();
    hnsw.parallel_insert_slice(&to_insert);
    //
    Ok(hnsw.get_nb_point())
} // end of array2_insert_hnsw

//=======================================================================

#[cfg(test)]
#[allow(unused)]
mod tests {

    use super::*;
    use statrs::function::erf::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // generate data as paragraph 4 of Harlim a Berry Variable Bandwith diffusion Kernels
    // Appl. Comput. Harmon. Anal. 40 (2016) 68–96
    fn generate_1d_gaussian(nbdata: usize) -> Vec<f32> {
        let delta = 1. / (nbdata + 1) as f64;
        let mut v = Vec::<f32>::with_capacity(nbdata);
        for i in 1..nbdata {
            let arg = 2. * delta * i as f64 - 1.;
            let d = (2.0_f64.sqrt() * erf_inv(arg)) as f32;
            if !d.is_normal() {
                log::error!("float problem arg = {}, d = {:?}", arg, d);
                panic!();
            } else {
                v.push(d);
            }
        }
        v
    }

    #[test]
    fn harlim_4() {
        log_init_test();
        //
        let nb_data = 4000;
        let data = generate_1d_gaussian(nb_data);
        // do dmap embedding, laplacian computation
        let dtime = 0.2;
        let dparams = DiffusionParams::new(4, Some(dtime));
        // hnsw definition
        let mut hnsw = Hnsw::<f32, DistL2>::new(64, nb_data, 16, 200, DistL2::default());
        hnsw.set_keeping_pruned(true);
        //
        for (i, d) in data.iter().enumerate() {
            hnsw.insert((&[*d], i));
        }
        log::info!("hnsw insertion done");
        //
        //
        let mut diffusion_map = DiffusionMaps::new(dparams);
        let emmbedded = diffusion_map.embed_from_hnsw::<f32, DistL2, f32>(&mut hnsw, 10, Some(0.2));
        let svd_res = diffusion_map.get_svd_res().unwrap();
        // get left eigen vectors array dimension is ()
        let left_u = svd_res.get_u().as_ref().unwrap();
        log::info!("left eigenvector dim : {:?}", left_u.dim());
        //
        log::info!("harlim_4 got embedding of size : {:?}", emmbedded.dim());
        // eigenvectors are Hermite polynomials
        let (vec_size, nb_vec) = emmbedded.dim();
        let dump_size = 40;
        let xmin = &data[0];
        let xmax = data.last().unwrap();
        log::info!("xmin = {:.3e}, xmax = {:.3e}", xmin, xmax);
        let gap = (nb_data - 1) / dump_size;
        let mut x = xmin;
        for i in 0..6 {
            let mut j = 0;
            log::info!("vec of rank i : {}", i);
            println!("vec of rank i = {}", i);
            println!(" x     v ");
            while x < xmax && j * gap < data.len() {
                let x = data[j * gap];
                let v = left_u[[j, i]];
                println!("{:.3e} , {:.3e}, ", x, v);
                j = j + 1;
            }
        }
        // compare with H3(x) = 1./sqrt(6.) * (x*x*x - 3*x)
        let emmbedded = diffusion_map.embed_hnsw::<f32, DistL2, f32>(&mut hnsw);
    } // end of harlim_4
} // end of mod tests
