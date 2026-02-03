#![allow(clippy::doc_overindented_list_items)]

//!  Variable bandwidth diffusion maps embedding.
//!
//! This module (presently) computes a diffusion embedding for the kernel constructed from nearest neighbours
//! stored in a Hnsw structure, see in module [embedder](crate::embedder).  
//! The scale used in the kernel is dependant on points as described in Berry and Harlim paper
//!
//! Bibliography:
//! - *Diffusion Maps*. Coifman Lafon Appl. Comput. Harmon. Anal. 21 (2006) 5–30
//! - *Self-Tuning Spectral Clustering*  Zelkin-Manor Perrona NIPS 2004
//! - *From graph to manifold Laplacian: The convergence rate*. Singer Appl. Comput. Harmon. Anal. 21 (2006)
//! - *Variables bandwith diffusion kernels* Berry and Harlim. Appl. Comput. Harmon. Anal. 40 (2016) 68–96
//!
//!  Details are discussed in [params](DiffusionParams)

use num_traits::Float;
use num_traits::cast::FromPrimitive;
use std::sync::atomic::{AtomicU32, Ordering};

use indexmap::IndexSet;
use quantiles::ckms::CKMS;
use rayon::prelude::*;
use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};
use sprs::{CsMat, TriMatBase};

use crate::embedder::*;
use crate::fromhnsw::{kgraph::KGraph, kgraph_from_hnsw_all};
use crate::graphlaplace::*;

#[allow(unused)]
use crate::tools::{clip, kdumap::*, matrepr::*, nodeparam::*, svdapprox::*};

use anyhow::Result;
use hnsw_rs::prelude::*;

#[cfg_attr(doc, katexit::katexit)]
///
/// The main parameters to initialize the structure are:
///  - the dimension of the embedding.
///  - the time of the embedding. By default it is computed using the decay of eigenvalues of the laplacian
///  - the number of neighbours to used in the comoputation of the laplacian.  
///     By default it is deduced by the number neighbours used in hnsw
///     with a limitation up to 16 (as the hnsw can require a large number of connection). To limit the cpu time it is possible to reduce it.
///     A good range is between 8 and 12.
///
///   The kernel uses variable bandwith of the form:
///     $$ K_{\epsilon}(x,y)  = h \left( \frac{\| x - y \|^{2}}{\epsilon \rho(x) \rho(y)} \right)  $$
///     with $\rho$ a scale function computed as mean L2 distance of a node to its neigbours. $h$ is taken to be the $exp(-x^{2}) $ function
///
///   If the scales were constant the corresponding laplacian is
///      $$ L f = \Delta f + (2 - 2 \alpha) \nabla f . \frac{\nabla  q}{q} $$
///      with $q$ being the sampling density. (See Coifman Lafon)
///
///  - alfa  
///      So setting alfa = 1. cancels the  data potential effect of data sampling density variation.
///
///  - beta is useful when the scale is not constant we get from Berry-Harlim paper corollary 1  
///
///    If d is the "intrinsic" dimension of the data the Laplacian we get is:
///    $$ L f = \Delta f + (2 - 2 \alpha) \nabla f . \frac{\nabla  q}{q} + (d+2) \  \nabla f . \frac{\nabla  \rho}{\rho} $$
///     We can estimate density $ q$ and then postulate  scale  from the relation  $$ \rho = q^{\beta} $$ with  $ \beta \lt 0 $.
///     We then get :
///    $$ L f = \Delta f + c_{1} \nabla f . \frac{\nabla  q}{q} $$ with $ c_{1} = 2 - 2 \alpha + \beta ( 2 + d) $
///
///    As we need to keep $ c_{1} \ge 0 $ and  we have $ \beta \lt 0 $ we must choose $  \alpha \gt \beta + 0.1 $ to ensure error control (See Harlim-Berry).  
///    For $ \beta = 0 $ ,  $ \alpha = 0.5 $ or $ \alpha = 1. $ are standard choices.  
///    For $\beta \ne 0$ , $ \beta \in \[ -0.1, -0.5\]$ and $ \alpha = 0.5 $ are standard choices.
#[derive(Copy, Clone, Debug)]
pub struct DiffusionParams {
    /// dimension of embedding
    asked_dim: usize,
    /// kernel biaising exponent of sampling law. By default we use 0.
    alfa: f32,
    /// exponent for going from density to scales
    beta: f32,
    /// The epsil parameter see doc above
    epsil: f32,
    /// embedding time
    t: Option<f32>,
    /// number of neighbour used in the laplacian graph. Useful if we want to keep less neighbours than hnsw has.
    gnbn_opt: Option<usize>,
    //
    h_layer: Option<usize>,
} // end of DiffusionParams

impl DiffusionParams {
    /// arguments are:
    /// - embedding dimension
    /// - optional diffusion time
    /// - optional number of neighbours used in laplacian discretisation
    /// - alfa is set to 0.5 and beta to  -0.1, epsil to 2.
    pub fn new(asked_dim: usize, t_opt: Option<f32>, g_opt: Option<usize>) -> Self {
        DiffusionParams {
            asked_dim,
            alfa: 0.5,
            beta: -0.1,
            epsil: 2.0f32,
            t: t_opt,
            gnbn_opt: g_opt,
            h_layer: None,
        }
    }
    /// get embedding time
    pub fn get_time(&self) -> Option<f32> {
        self.t
    }

    /// get dimension
    pub fn get_data_dim(&self) -> usize {
        self.asked_dim
    }

    pub fn get_gnbn(&self) -> Option<usize> {
        self.gnbn_opt
    }
    //
    /// modify the default alfa See Lafon paper.
    /// natural values are 0. , 1/2 and 1.
    pub fn set_alfa(&mut self, alfa: f32) {
        let alfa_min = -2.;
        let alfa_sup = 1.;
        if !(alfa_min..=alfa_sup).contains(&alfa) {
            self.alfa = alfa.max(alfa_min).min(alfa_sup);
            log::warn!(
                "alfa should be in [{:.3e} , {:.3e}], setting lafa to {:.3e} ",
                alfa_min,
                alfa_sup,
                self.alfa
            );
            return;
        }
        self.alfa = alfa;
    }

    /// set beta, must be negative in range -1. 0.
    /// As beta -> 0, we go nearer to fixed bandwidth
    pub fn set_beta(&mut self, beta: f32) {
        if (-1.01..=0.).contains(&beta) {
            self.beta = beta;
        } else {
            log::warn!(
                "not changing beta, beta should be in -1,0 Usual values are 0. -0.5 see doc "
            );
        }
    }

    /// should be between 0.5 and 4, can be slightly smaller in variable bandwidth case ( beta < 0.). Default to
    pub fn set_epsil(&mut self, epsil: f32) {
        let epsil_min = 0.50f32;
        let epsil_max: f32 = 4.0f32;
        //
        self.epsil = epsil.min(epsil_max).max(epsil_min);
        log::info!(
            "setting epsil to {:.3e}, should be between 0.5 and 4, can be slightly smaller in variable bandwidth case ( beta < 0.)",
            epsil
        );
    }

    /// set desired number of neighbours used in Kgraph (must be less than in Hnsw construction)
    pub fn set_gnbn(&mut self, nbn: usize) {
        self.gnbn_opt = Some(nbn);
    }

    pub fn set_hlayer(&mut self, layer: usize) {
        self.h_layer = Some(layer);
    }

    pub fn get_alfa(&self) -> f32 {
        self.alfa
    }

    pub fn get_beta(&self) -> f32 {
        self.beta
    }

    pub fn get_epsil(&self) -> f32 {
        self.epsil
    }

    /// returns layer above which diffusion map will run (default is 0)
    pub fn get_hlayer(&self) -> usize {
        self.h_layer.unwrap_or_default()
    }

    pub fn get_embedding_dimension(&self) -> usize {
        self.asked_dim
    }

    pub fn set_embedding_dimension(&mut self, asked_dim: usize) {
        self.asked_dim = asked_dim;
    }

    /// build variable density default parameters
    /// beta is set to -0.5
    pub fn build_with_variable_bandwidth() -> Self {
        DiffusionParams {
            asked_dim: 2,
            alfa: 0.5,
            beta: -0.1,
            epsil: 1.5f32,
            t: Some(5.),
            gnbn_opt: Some(12),
            h_layer: None,
        }
    }

    /// aflfa = 1, beta = 0
    pub fn build_with_fixed_bandwidth() -> Self {
        DiffusionParams {
            asked_dim: 2,
            alfa: 1.,
            beta: 0.,
            epsil: 2.0f32,
            t: Some(5.),
            gnbn_opt: Some(16),
            h_layer: None,
        }
    }
} // end of DiffusionParams

/// build with
impl Default for DiffusionParams {
    fn default() -> Self {
        DiffusionParams {
            asked_dim: 2,
            alfa: 1.,
            beta: 0.,
            epsil: 2.0f32,
            t: Some(5.),
            gnbn_opt: Some(12),
            h_layer: None,
        }
    }
}

impl std::fmt::Display for DiffusionParams {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // display main parameters
        write!(
            f,
            "alfa: {:.3e}, beta: {:.3e}, epsil : {:.3e}",
            self.alfa, self.beta, self.epsil
        )
    }
}

/// The algorithm implements:
///  *Variables bandwith diffusion kernels* Berry and Harlim. Appl. Comput. Harmon. Anal. 40 (2016) 68–96.
///
/// A random svd approximation is used to run on large number of items (See module [crate::tools::svdapprox])  
pub struct DiffusionMaps {
    /// parameters to use
    params: DiffusionParams,
    /// node parameters coming from graph transformation
    _node_params: Option<NodeParams>,
    // ratio from local_scale / median_scale deduced from distance
    normed_scales: Option<Array1<f32>>,
    // mean scale
    mean_scale: f32,
    // scales deduced from density and beta arg (beta < 0)
    beta_scales: Option<Array1<f32>>,
    //
    q_density: Option<Vec<f32>>,
    //
    laplacian: Option<GraphLaplacian>,
    /// to keep track of rank DataId conversion
    index: Option<IndexSet<DataId>>,
} // end of DiffusionMaps

impl DiffusionMaps {
    /// iitialization from NodeParams
    pub fn new(params: DiffusionParams) -> Self {
        DiffusionMaps {
            params,
            _node_params: None,
            normed_scales: None,
            mean_scale: 0.,
            beta_scales: None,
            q_density: None,
            laplacian: None,
            index: None,
        }
    }

    pub fn get_local_scales(&self) -> &Array1<f32> {
        assert!(self.normed_scales.is_some());
        if self.beta_scales.is_some() {
            self.beta_scales.as_ref().unwrap()
        } else {
            self.normed_scales.as_ref().unwrap()
        }
    }

    // returns mean scale (based on original l2 distances between nodes)
    pub fn get_mean_scale(&self) -> f32 {
        self.mean_scale
    }

    /// returns gr  ph laplacian if already computed and stored in structure
    #[allow(unused)]
    pub(crate) fn get_laplacian(&mut self) -> Option<&mut GraphLaplacian> {
        self.laplacian.as_mut()
    }

    /// returns svd result computed in dmap embedding
    pub fn get_svd_res(&self) -> Option<&SvdResult<f32>> {
        match &self.laplacian {
            Some(laplacian) => laplacian.svd_res.as_ref(),
            _ => None,
        }
    }

    pub fn get_index(&self) -> Option<&IndexSet<DataId>> {
        self.index.as_ref()
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
        let kgraph_res: anyhow::Result<KGraph<F>> =
            kgraph_from_hnsw_all::<T, D, F>(hnsw, knbn as usize);
        if kgraph_res.is_err() {
            panic!(
                "kgraph_from_hnsw_all failed {:?}",
                kgraph_res.err().unwrap()
            );
        };
        let kgraph = kgraph_res.unwrap();
        // get NodeParams. CAVEAT to_proba_edges apply initial shift!!
        let nodeparams = to_proba_edges::<F>(&kgraph, 1., 2.);
        get_dmap_embedding::<F>(&nodeparams, self.params.asked_dim, self.params.get_time())
    } // end embed_hnsw

    /// Return laplacian from hnsw nearest neighbours.
    /// F is float type we want the result in
    pub(crate) fn laplacian_from_hnsw<T, D, F>(
        &mut self,
        hnsw: &Hnsw<T, D>,
        dparams: &DiffusionParams,
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
        let gnbn = dparams.get_gnbn().unwrap_or(16);
        // hnsw can have large max_nb_connection (typically 64), we set a bound
        let knbn = hnsw.get_max_nb_connection().min(gnbn as u8);
        let kgraph_res = kgraph_from_hnsw_all::<T, D, F>(hnsw, knbn as usize);
        if kgraph_res.is_err() {
            panic!(
                "kgraph_from_hnsw_all failed {:?}",
                kgraph_res.err().unwrap()
            );
        }
        let kgraph = kgraph_res.unwrap();
        let sampling_size = 10000;
        let dim_stat = kgraph.estimate_intrinsic_dim_2nn(sampling_size);
        if let Ok(dim_stat) = dim_stat {
            log::info!(
                "\n DiffusionMaps::laplacian_from_hnsw estimate_intrinsic_dim_2nn dimension estimation with nbpoints : {}, dim : {:.5e} \n",
                sampling_size,
                dim_stat,
            );
            println!(
                " estimate_intrinsic_dim_2nn dimension estimation with nbpoints : {}, dim : {:.5e}",
                sampling_size, dim_stat
            );
        }
        // we store indexset to be able to go back from index (in embedding) to dataId (in hnsw) as kgrap will be deleted
        self.index = Some(kgraph.get_indexset().clone());
        //
        self.laplacian_from_kgraph(&kgraph)
    }

    // to be called by embed_from_kgraph
    pub(crate) fn laplacian_from_kgraph<F>(&mut self, kgraph: &KGraph<F>) -> GraphLaplacian
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
        // we store indexset to be able to go back from index (in embedding) to dataId (in hnsw) as kgrap will be deleted
        if self.index.is_none() {
            self.index = Some(kgraph.get_indexset().clone());
        }
        // get NodeParams. fill fields local_scale, mean_scale and  beta_scales if necessary
        let nbng = if let Some(asked_nbng) = self.params.get_gnbn() {
            asked_nbng.min(kgraph.get_max_nbng())
        } else {
            kgraph.get_max_nbng()
        };
        let nodeparams = self.compute_dmap_nodeparams::<F>(kgraph, nbng);
        //
        self.compute_laplacian(&nodeparams, self.params.get_alfa())
    } // end of laplacian_from_kgraph

    // transform nodeparams to a kernel.
    // We apply alfa parameter to possibly swap from Laplace-Beltrami to Ornstein-Uhlenbeck
    // as in Coifman-Lafon 2006.
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
        // compute local_scales
        let local_scale = self.normed_scales.as_ref().unwrap();
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
            // First we need to symetrize the graph.
            let mut symgraph = (&transition_proba + &transition_proba.view().t()) * 0.5;
            //
            // now we go to the symetric weighted laplacian D^-1/2 * G * D^-1/2 but get rid of the I - ...
            // We use Coifman-Lafon notatio,.    Lafon-Keller-Coifman
            // Diffusions Maps appendix B
            // IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,VOL. 28, NO. 11,NOVEMBER 2006
            //
            // compute q_alfa which is a proxy for density of data, then we use alfa for possible reweight for density
            let mut q = symgraph.sum_axis(Axis(1));
            let q_mean = q.sum() / max_nbng as f32;
            // scale normalization
            q /= q_mean;
            let mut degrees = Array1::<f32>::zeros(q.len());
            for i in 0..nbnodes {
                let mut row = symgraph.row_mut(i);
                for j in 0..nbnodes {
                    row[[j]] /= (q[[i]] * q[[j]]).powf(alfa);
                }
                degrees[[i]] = row.sum();
                //
            }
            // now we normalize rows according to D^-1/2 * G * D^-1/2. See Berry-Harlim P 82
            let symetrization_weights = degrees.sqrt();
            for i in 0..nbnodes {
                let mut row = symgraph.row_mut(i);
                for j in 0..nbnodes {
                    row[[j]] /= symetrization_weights[[i]] * symetrization_weights[[j]];
                }
                if log::log_enabled!(log::Level::Debug) {
                    // check normalization
                    let mut check = 0.0f32;
                    for j in 0..nbnodes {
                        check += row[[j]] * symetrization_weights[[j]];
                    }
                    check /= symetrization_weights[[i]];
                    if (check - 1.0).abs() > 1.0e-3 {
                        log::debug!(" check = {:.3e}", check);
                        panic!("bad normalization");
                    }
                }
            }
            //
            log::trace!("\n allocating full matrix laplacian");
            GraphLaplacian::new(
                MatRepr::from_array2(symgraph),
                symetrization_weights,
                Some(local_scale.clone()),
            )
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
            let mut q = Array1::<f32>::zeros(nbnodes);
            let mut rows = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
            let mut cols = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
            let mut values = Vec::<f32>::with_capacity(nbnodes * 2 * max_nbng);

            for ((i, j), val) in edge_list.iter() {
                let sym_val;
                if let Some(t_val) = edge_list.get(&(*j, *i)) {
                    // we are in proba mode, if both direction take max proba
                    sym_val = val.max(*t_val);
                } else {
                    sym_val = *val;
                }
                rows.push(*i);
                cols.push(*j);
                values.push(sym_val);
                q[*i] += sym_val;
                //
                rows.push(*j);
                cols.push(*i);
                values.push(sym_val);
                q[*j] += sym_val;
            }
            // scale normalization. We get something like a kernel density estimate
            let q_mean = q.sum() / max_nbng as f32;
            // scale normalization
            q /= q_mean;
            let mut degrees = Array1::<f32>::zeros(q.len());
            //
            // as in FULL Representation we avoided the I diagnoal term which cancels anyway
            // Now we apply density weighting according to alfa
            for i in 0..rows.len() {
                let row = rows[i];
                let col = cols[i];
                values[i] /= (q[row] * q[col]).powf(alfa);
            }
            // now we normalize rows
            // degrees correspond to q_{epsil, alfa} in Harlim Berry
            //
            for (i, v) in &mut values.iter().enumerate() {
                let row = rows[i];
                degrees[row] += v;
            }
            let symetrization_weights = degrees.sqrt();
            for i in 0..values.len() {
                let row = rows[i];
                let col = cols[i];
                values[i] /= symetrization_weights[row] * symetrization_weights[col];
            }
            log::trace!("allocating csr laplacian");
            let laplacian = TriMatBase::<Vec<usize>, Vec<f32>>::from_triplets(
                (nbnodes, nbnodes),
                rows,
                cols,
                values,
            );
            let csr_mat: CsMat<f32> = laplacian.to_csr();
            GraphLaplacian::new(
                MatRepr::from_csrmat(csr_mat),
                symetrization_weights,
                Some(local_scale.clone()),
            )
        } // end case CsMat
        //
    }

    // building block of scales_to_nodeparams to build nodeparams in //
    fn build_node_param<F, FUN>(
        &self,
        node: usize,
        neighbours: &Vec<OutEdge<F>>,
        local_scales: &[F],
        scale_to_kernel: FUN,
        nb_weight_too_low: &AtomicU32,
    ) -> NodeParam
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
        FUN: Send + Sync + Fn(F, f32, f32) -> f32,
    {
        // no isolated points !
        if neighbours.is_empty() {
            log::error!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            log::error!("encountered an isolated point, exiting ");
            std::process::exit(1);
        }
        // get rid of case where all neighbours have dist 0 to current node (It happens in Higgs.data!!!)
        let mut all_equal = false;
        let last_n = neighbours
            .iter()
            .rfind(|&n| n.weight.to_f32().unwrap() > 0.);
        //
        if let Some(edge) = last_n {
            let last_e_w = edge.weight;
            if last_e_w <= neighbours[0].weight {
                all_equal = true;
            }
        } else {
            // means all distances are 0! (encountered in Higgs Boson bench)
            all_equal = true;
        }
        // we add each node as a neighbour of itself to enforce ergodicity !!
        let nb_edges = 1 + neighbours.len();
        let mut edges = Vec::<OutEdge<f32>>::with_capacity(nb_edges);
        if all_equal {
            log::warn!(
                "all equal for node {}, weight {:.3e}, nb_neighbours {}",
                node,
                neighbours[0].weight.into(),
                neighbours.len()
            );
            // all neighbours will have
            let proba: f32 = 1. / (nb_edges as f32);
            let self_edge = OutEdge::new(node, proba);
            edges.push(self_edge);
            for n in neighbours {
                edges.push(OutEdge::new(n.node, proba));
            }
        } else {
            let self_edge = OutEdge::<f32>::new(node, 1.);
            edges.push(self_edge);
            let _shift = neighbours[0].weight.to_f32().unwrap();
            let from_scale = local_scales[node];
            // Recall no shift but could add drift with respect to local_scales variations
            for n in neighbours {
                let to_scale = local_scales[n.node];
                let local_scale = (to_scale * from_scale).sqrt();
                let mut weight: f32 = scale_to_kernel(n.weight, 0., local_scale.to_f32().unwrap());
                if weight < PROBA_MIN {
                    weight = weight.max(PROBA_MIN);
                    nb_weight_too_low.fetch_add(1, Ordering::SeqCst);
                }
                let edge = OutEdge::<f32>::new(n.node, weight);
                edges.push(edge);
            }
            // TODO: we adjust self_edge
            //                edges[0].weight = 1. / nb_edges as f32;
            if nb_edges > 1 {
                edges[0].weight = edges[1].weight;
            } else {
                edges[0].weight = 1. / 2.;
            }
        }
        //
        NodeParam::new(local_scales[node].to_f32().unwrap(), edges)
    }

    // This function is called by Self::compute_dmap_nodeparams and Self::density_to_kernel
    // Given scales (rho in Harlim) it computes nodeparams.
    // it is called once with scales deduced from distances and possibly once more with scales
    // coming from density**beta if beta < 0 where it modulates first mean scale estimated  from distance with
    // relative density**beta
    //
    // remap node params expressing distance to proba knowing scale to adopt
    // scales must not be normalized by their mean to keep in par with scales of distances. (So epsil keeps around 1.)
    //
    // Args:
    //  - local_scales is array of l2 ditances to node neighbours
    //  - scale_to_kernel : kernel function
    //
    // The function returns Nodeparams corresponding to new kernel and associated density (computed as mean transition proba)
    fn scales_to_nodeparams<F, FUN>(
        &self,
        kgraph: &KGraph<F>,
        local_scales: &[F],
        scale_to_kernel: &FUN,
    ) -> NodeParams
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
        FUN: Send + Sync + Fn(F, f32, f32) -> f32,
    {
        log::debug!("in DiffusionMaps::scales_to_kernel");
        //
        let nb_nodes = kgraph.get_nb_nodes();
        let neighbour_hood = kgraph.get_neighbours();
        //
        //let mut nodeparams = Vec::<NodeParam>::with_capacity(nb_nodes);
        //
        // now we have scales we can remap edge length to weights.
        // we choose epsil to put weight on at least 5 neighbours when no shift
        // TODO: we can now parallelize with Atomic on nb_weight_too_low
        let nb_weight_too_low = AtomicU32::new(0);
        // now we loop on all nodes
        let nodeparams: Vec<NodeParam> = (0..nb_nodes)
            .into_par_iter()
            .map(|i| {
                self.build_node_param(
                    i,
                    &neighbour_hood[i],
                    local_scales,
                    scale_to_kernel,
                    &nb_weight_too_low,
                )
            })
            .collect();
        //
        let low_weight = nb_weight_too_low.into_inner() as f32
            / (kgraph.get_max_nbng() * nodeparams.len()) as f32;
        if low_weight > 0. {
            log::info!(
                "to_dmap_nodeparams: proba of weight < {:.2e} = {:.2e}, possibly increase epsil",
                PROBA_MIN,
                low_weight
            );
        }
        //
        NodeParams::new(nodeparams, kgraph.get_max_nbng())
    }

    /// dmap specific edge proba compuatitons
    /// - compute basic transition kernel, with global scaling from l2 distances
    /// - estimate density
    /// - re estimate scale from density with from_density_to_new_scales
    ///
    pub(crate) fn compute_dmap_nodeparams<F>(
        &mut self,
        kgraph: &KGraph<F>,
        nbng: usize,
    ) -> NodeParams
    where
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::ops::DivAssign
            + std::iter::Sum
            + Into<f64>,
    {
        log::info!(
            "Diffusion computing kernels with using alfa : {:.2e} , beta : {:.2e}; epsil : {:.2e}",
            self.params.get_alfa(),
            self.params.get_beta(),
            self.params.get_epsil()
        );
        //
        let neighbour_hood = kgraph.get_neighbours();
        let nbgh_size = kgraph.get_max_nbng().min(nbng);
        log::info!(
            "compute_dmap_nodeparams kgraph nbng : {}, using size {}",
            kgraph.get_max_nbng(),
            nbgh_size
        );
        // compute a scale around each node, mean scale and quantiles on scale
        let mut local_scales: Vec<F> = neighbour_hood
            .par_iter()
            .map(|edges| self.get_dist_l2_from_node(edges, nbgh_size))
            .collect();
        // collect scales quantiles
        let scales_q = self.get_quantiles("scales quantiles first pass", &local_scales);
        let scale_width =
            (scales_q.query(0.999).unwrap().1 / scales_q.query(0.001).unwrap().1) as f32;
        let scale_median = scales_q.query(0.5).unwrap().1;
        log::info!(
            "compute_dmap_nodeparams knbn : {}, scale_max/min: {:.2e} scale_median : {:.2e}",
            nbgh_size,
            scale_width,
            scale_median
        );
        // normalize , get rid of zero scales (if all neighbours are at same position of point ! It happen!)
        //
        let sum = local_scales.iter().fold(F::zero(), |acc, s| acc + (*s));
        assert!(!sum.is_nan());
        let mean = sum / F::from(local_scales.len()).unwrap();
        ///////        let median = F::from(scale_median).unwrap();
        assert!(mean > F::zero());
        for d in &mut local_scales {
            if *d <= F::zero() {
                *d = mean;
            }
        }
        // now we have scales we can remap edge length to weights.
        // we choose epsil to put weight on at least 5 neighbours when no shift
        let exponent = 2.0f32;
        //
        let scales_f: Array1<f32> =
            Array1::<f32>::from_iter(local_scales.iter().map(|s| ((*s) / mean).to_f32().unwrap()));
        self.mean_scale = mean.to_f32().unwrap();
        let _ = self.get_quantiles(
            "normalized scales quantiles from first pass",
            scales_f.as_slice().unwrap(),
        );
        self.normed_scales = Some(scales_f);
        //  See documentation for epsil
        let epsil = self.params.get_epsil().sqrt();
        //
        let beta = self.params.get_beta();
        if beta > 0. {
            log::error!("beta cannot be > 0.");
            std::process::exit(1);
        }
        let remap_weight = |w: F, shift: f32, scale: f32| {
            let arg = ((w.to_f32().unwrap() - shift) / (epsil * scale)).powf(exponent);
            (-arg).exp()
        };
        //
        log::info!("using beta : {:.3e}", beta);
        if beta < 0. {
            let nodeparams = self.scales_to_nodeparams(kgraph, &local_scales, &remap_weight);
            let beta_scales = self.kernel0_to_density(beta, &nodeparams);
            let beta_scales_f: Vec<F> = beta_scales.iter().map(|s| F::from(*s).unwrap()).collect();
            let nodeparams = self.scales_to_nodeparams(kgraph, &beta_scales_f, &remap_weight);
            self.beta_scales = Some(beta_scales);
            nodeparams
        } else {
            // beta = 0 means we keep scale constant! so it is in fact fixed bandwidth
            let local_scales = vec![mean; local_scales.len()];
            self.scales_to_nodeparams(kgraph, &local_scales, &remap_weight)
        }
    } // end compute_dmap_nodeparams

    // Can be called by compute_dmap_nodeparams if we require restimation of scale in function of density (beta w 0.)
    // from nodeparams we can estimate density and reset scales depending upon beta.
    //
    // stores estimated density in field q_density and return new scales adjusted to previous mean scale
    fn kernel0_to_density(&mut self, beta: f32, initial_space: &NodeParams) -> Array1<f32> {
        //
        log::info!("using beta : {:.3e}", beta);
        //
        let nbnodes = initial_space.get_nb_nodes();
        // get stats
        let max_nbng = initial_space.get_max_nbng();
        let node_params = initial_space;
        // compute local_scales
        // TODO define a threshold for dense/sparse representation
        let q: Array1<f32> = if nbnodes <= FULL_MAT_REPR {
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
            // First we need to symetrize the graph.
            let symgraph = (&transition_proba + &transition_proba.view().t()) * 0.5;
            //
            // now we go to the symetric weighted laplacian D^-1/2 * G * D^-1/2 but get rid of the I - ...
            // We use Coifman-Lafon notatio,.    Lafon-Keller-Coifman
            // Diffusions Maps appendix B
            // IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,VOL. 28, NO. 11,NOVEMBER 2006
            //
            // compute q_alfa which is a proxy for density of data, then we use alfa for possible reweight for density
            let mut q = symgraph.sum_axis(Axis(1)) / max_nbng as f32;
            let q_mean = q.sum() / q.len() as f32;
            // scale normalization
            q /= q_mean;
            q
            // dump quantiles
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
            let mut q = Array1::<f32>::zeros(nbnodes);
            let mut rows = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
            let mut cols = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
            let mut values = Vec::<f32>::with_capacity(nbnodes * 2 * max_nbng);

            for ((i, j), val) in edge_list.iter() {
                let sym_val;
                if let Some(t_val) = edge_list.get(&(*j, *i)) {
                    // we are in proba mode, if both direction take max proba
                    sym_val = val.max(*t_val);
                } else {
                    sym_val = *val;
                }
                rows.push(*i);
                cols.push(*j);
                values.push(sym_val);
                q[*i] += sym_val;
                //
                rows.push(*j);
                cols.push(*i);
                values.push(sym_val);
                q[*j] += sym_val;
            }
            // scale normalization. We get something like a kernel density estimate
            q /= max_nbng as f32;
            let q_mean = q.sum() / q.len() as f32;
            q /= q_mean;
            q
        }; // end case CsMat
        // now we have an estimate of density scaled around 1.
        // We can recompute local scales with d**beta modulating initial mean scale
        let q_beta: Vec<f32> = q.iter().map(|d| (*d).powf(beta)).collect();
        let _ = self.get_quantiles("density**beta", &q_beta);
        //
        let mut scales = Array1::<f32>::from_vec(q_beta);
        scales *= self.mean_scale;
        // we dump scale quantiles
        let _ = self.get_quantiles(
            "scale deduced from density**beta",
            scales.as_slice().unwrap(),
        );
        //
        self.q_density = Some(q.to_vec());
        //
        scales
    }

    //

    // just dump some quantiles (mostly scales and densities)
    pub(crate) fn get_quantiles<F>(&self, message: &str, values: &[F]) -> CKMS<f64>
    where
        F: Float + Into<f64>,
    {
        log::info!("\n\n {}", message);
        let mut quant_densities: CKMS<f64> = CKMS::<f64>::new(0.001);
        for q in values {
            quant_densities.insert((*q).into());
        }
        log::info!(
            "quantiles at 0.001 : {:.2e}, 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.999 : {:.2e}",
            quant_densities.query(0.001).unwrap().1,
            quant_densities.query(0.05).unwrap().1,
            quant_densities.query(0.5).unwrap().1,
            quant_densities.query(0.95).unwrap().1,
            quant_densities.query(0.990).unwrap().1
        );
        log::debug!("");
        //
        quant_densities
    }

    // computes scale (mean L1 norm of dist) around a point
    // we compute mean of dist to first neighbour around a point given outgoing edges and graph
    #[allow(unused)]
    pub(crate) fn get_dist_l1_node<F>(
        &self,
        kgraph: &KGraph<F>,
        out_edges: &[OutEdge<F>],
        nbng_used: usize,
    ) -> F
    where
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::iter::Sum,
    {
        // determnine mean distance to nearest neighbour at local scale, reason why we need kgraph as argument.
        let rho_x = out_edges[0].weight;
        let mut rho_y_s = Vec::<F>::with_capacity(out_edges.len() + 1);
        //
        for neighbour in out_edges.iter().take(nbng_used) {
            let y_i = neighbour.node; // y_i is a NodeIx = usize
            rho_y_s.push(kgraph.get_neighbours()[y_i][0].weight);
        } // end of for i
        //
        rho_y_s.push(rho_x);
        let size = rho_y_s.len();
        if size > 0 {
            rho_y_s.into_iter().sum::<F>() / F::from(size).unwrap()
        } else {
            F::zero()
        }
    } // end of get_dist_around_node

    //

    // computes scale (mean norm of dist) around a point
    // we compute L2 mean of distance from a node
    pub(crate) fn get_dist_l2_from_node<F>(&self, out_edges: &[OutEdge<F>], nbng: usize) -> F
    where
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::iter::Sum,
    {
        //
        let dist2: F = out_edges
            .iter()
            .take(nbng)
            .map(|e| e.weight * e.weight)
            .sum();
        //
        if !out_edges.is_empty() {
            (dist2 / F::from(out_edges.len()).unwrap()).sqrt()
        } else {
            F::zero()
        }
    }

    // useful if we have already hnsw.
    // Note that embedded data are not reindexed to DataId
    pub fn embed_from_kgraph<F>(
        &mut self,
        kgraph: &KGraph<F>,
        dparams: &DiffusionParams,
    ) -> Result<Array2<F>>
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
        log::info!("in DiffusionMaps::embed_from_kgraph");
        //
        let asked_dim = dparams.get_data_dim();
        let t_opt = dparams.get_time();
        let mut laplacian = self.laplacian_from_kgraph::<F>(kgraph);
        let embedded = self
            .embed_from_laplacian::<F>(&mut laplacian, asked_dim, t_opt)
            .unwrap();
        // now we can store laplacian
        self.laplacian = Some(laplacian);
        //
        Ok(embedded)
    } // end of embed_from_kgraph

    pub(crate) fn embed_from_hnsw_intern<T, D, F>(
        &mut self,
        hnsw: &Hnsw<T, D>,
        dparams: &DiffusionParams,
    ) -> Result<Array2<F>>
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
        let asked_dim = dparams.get_data_dim();
        let t_opt = dparams.get_time();
        //
        let mut laplacian = self.laplacian_from_hnsw::<T, D, F>(hnsw, dparams);
        let embedded = self
            .embed_from_laplacian::<F>(&mut laplacian, asked_dim, t_opt)
            .unwrap();
        // now we can store laplacian
        self.laplacian = Some(laplacian);
        //
        Ok(embedded)
    }

    /// Do the whole work chain graph conversion from hnsw structure, NodeParams transformation and svd.
    /// T is the type on which distances in Hnsw are computed,
    /// F is f32 or f64 depending on how diffusions Maps is to be computed.  
    /// The svd result are stored in the DiffusionMaps structure and accessible with the functions
    /// [Self::get_svd_res()].  
    /// Note that returned data are reindexed! (come with original Id)
    pub fn embed_from_hnsw<T, D, F>(
        &mut self,
        hnsw: &Hnsw<T, D>,
        dparams: &DiffusionParams,
    ) -> Result<Array2<F>>
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
        // we get embedded data without reindexation
        let embedded = self.embed_from_hnsw_intern(hnsw, dparams).unwrap();
        // and we reindex
        let embedded_reindexed = self.reindex_embedding(&embedded);
        //
        Ok(embedded_reindexed)
    } // end of embed_from_hnsw

    //

    // once we have laplacian get compute eigenvectors and weight them with time and eigenvalues
    // This function returns an embeding in the graph indexing. To get back to DataId as sent in Hnsw.
    // Use [Self::reindex_embedding()] to get back to original data indexing
    fn embed_from_laplacian<F>(
        &self,
        laplacian: &mut GraphLaplacian,
        asked_dim: usize,
        t_opt: Option<f32>,
    ) -> Result<Array2<F>>
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
        //
        log::debug!("got laplacian, going to svd ... asked_dim :  {}", asked_dim);
        let svd_res = laplacian.do_svd(asked_dim + 15);
        if svd_res.is_err() {
            log::error!("embed_from_laplacian call of laplacian.do_svd failed");
            panic!("embed_from_laplacian call of laplacian.do_svd failed");
        } else {
            log::debug!("do_svd returns OK");
        }
        let svd_res = svd_res.unwrap();
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
            " first 5 eigen values {:.2e} {:.2e} {:.2e} {:.2e}  {:.2e} ",
            lambdas[0],
            lambdas[1],
            lambdas[2],
            lambdas[3],
            lambdas[4],
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
        let real_dim = asked_dim.min(u.ncols() - 1);
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
        // we must renormalize by
        let sum_diag = laplacian.normalizer.iter().sum::<f32>() / laplacian.normalizer.len() as f32;
        let scales = self.normed_scales.as_ref().unwrap();
        let max_coord = 10.0f32;
        for i in 0..u.nrows() {
            let row_i = u.row(i);
            let weight_i = scales[i] * (laplacian.normalizer[i] / sum_diag).sqrt();
            for j in 0..real_dim {
                // divide j value by diagonal and convert to F. take l_{i}^{t} as in dmap
                let clipped_val = crate::tools::clip::clip(
                    normalized_lambdas[j + 1].powf(time) * row_i[j + 1] / weight_i,
                    max_coord,
                );
                embedded[[i, j]] = F::from_f32(clipped_val).unwrap();
            }
        }
        log::debug!("DiffusionMaps::embed_from_hnsw ended");
        //
        laplacian.svd_res = Some(svd_res);
        //
        Ok(embedded)
    }

    /// This function reindex embedding according to original indexation
    /// returns embedded data reindexed by DataId. This requires the DataId to be contiguous from 0 to nbdata.  
    pub fn reindex_embedding<F>(&self, embedded: &Array2<F>) -> Array2<F>
    where
        F: Float,
    {
        //
        let (nbrow, dim) = embedded.dim();
        let mut reindexed = Array2::<F>::zeros((nbrow, dim));
        //
        let index = self.get_index().unwrap();
        //
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (One week bug!)
        for i in 0..nbrow {
            let row = embedded.row(i);
            let origin_id = index.get_index(i).unwrap();
            for j in 0..dim {
                reindexed[[*origin_id, j]] = row[j];
            }
        }
        reindexed
    } // end of get_embedding_reindexed
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
    let sum_diag = laplacian.normalizer.iter().sum::<f32>() / laplacian.normalizer.len() as f32;
    for i in 0..u.nrows() {
        let row_i = u.row(i);
        let weight_i = (laplacian.normalizer[i] / sum_diag).sqrt();
        for j in 0..real_dim {
            // divide j value by diagonal and convert to F. take l_{i}^{t} as in dmap
            embedded[[i, j]] =
                F::from_f32(normalized_lambdas[j + 1].powf(time) * row_i[j + 1] / weight_i)
                    .unwrap();
        }
    }
    log::debug!("ended get_dmap_embedding");
    embedded
} // end of get_dmap_embedding

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
    use crate::tools::io::write_csv_labeled_array2;
    use crate::utils::mnistio::*;
    use anyhow::anyhow;
    use cpu_time::ProcessTime;
    use ndarray::s;
    use statrs::function::erf::*;
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime};

    const MNIST_FASHION_DIR: &str = "/home/jpboth/Data/ANN/Fashion-MNIST/";
    const MNIST_DIGITS_DIR: &str = "/home/jpboth/Data/ANN/MNIST/";

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
    fn dmap_digits() {
        log_init_test();
        //
        log::info!("running mnist_digits");
        //
        let mnist_data = load_mnist_train_data(MNIST_DIGITS_DIR).unwrap();
        let labels = mnist_data.get_labels().to_vec();
        let images = mnist_data.get_images();
        // convert images as vectors
        let (_, _, nbimages) = images.dim();
        let mut images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        //
        for k in 0..nbimages {
            let v: Vec<f32> = images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32)
                .collect();
            images_as_v.push(v);
        }
        //
        // do dmap embedding, laplacian computation
        let dtime = 5.;
        let gnbn: Option<usize> = Some(12);
        let mut dparams: DiffusionParams = DiffusionParams::new(2, Some(dtime), gnbn);
        dparams.set_alfa(0.5);
        dparams.set_beta(-0.1);
        dparams.set_epsil(2.0);
        println!("DiffusionParams: {}", dparams);
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        // hnsw definition
        let mut hnsw = Hnsw::<f32, DistL2>::new(32, images_as_v.len(), 16, 200, DistL2::default());
        hnsw.set_keeping_pruned(true);
        //
        // we must pay fortran indexation once!. transform image to a vector
        let data_with_id: Vec<(&Vec<f32>, usize)> =
            images_as_v.iter().zip(0..images_as_v.len()).collect();
        hnsw.parallel_insert(&data_with_id);
        // dmaps
        let mut diffusion_map = DiffusionMaps::new(dparams);
        let emmbedded_res = diffusion_map.embed_from_hnsw::<f32, DistL2, f32>(&mut hnsw, &dparams);
        if emmbedded_res.is_err() {
            log::error!("embedding failed");
            panic!("dmap_fashion failed");
        };
        //
        log::info!(
            " dmap embed sys time {:.2e} s, cpu time : {}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_start.elapsed().as_secs()
        );
        // dump
        log::info!("dumping initial embedding in csv file");
        let mut csv_w = csv::Writer::from_path("mnist_digits_dmap.csv").unwrap();
        let _res = write_csv_labeled_array2(&mut csv_w, labels.as_slice(), &emmbedded_res.unwrap());
        csv_w.flush().unwrap();
    }

    #[test]
    fn dmap_fashion() {
        log_init_test();
        //
        log::info!("running mnist_fashion");
        //
        let fashion_train_data = load_mnist_train_data(MNIST_FASHION_DIR).unwrap();
        let mut labels = fashion_train_data.get_labels().to_vec();
        let images = fashion_train_data.get_images();
        // convert images as vectors
        let (_, _, nbimages) = images.dim();
        let mut images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        //
        for k in 0..nbimages {
            let v: Vec<f32> = images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32)
                .collect();
            images_as_v.push(v);
        }
        // load test data
        // ===============
        let fashion_test_data = load_mnist_test_data(MNIST_FASHION_DIR).unwrap();
        labels.append(&mut fashion_test_data.get_labels().to_vec());
        let images = fashion_test_data.get_images();
        // convert images as vectors
        let (_, _, nbimages) = images.dim();
        //
        for k in 0..nbimages {
            let v: Vec<f32> = images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32)
                .collect();
            images_as_v.push(v);
        }
        //
        // do dmap embedding, laplacian computation
        let dtime = 5.;
        let gnbn = 12;
        let mut dparams: DiffusionParams = DiffusionParams::new(2, Some(dtime), Some(gnbn));
        dparams.set_alfa(0.5);
        dparams.set_beta(-0.1);
        dparams.set_epsil(2.0);
        println!("DiffusionParams: {}", dparams);
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        // hnsw definition
        let mut hnsw = Hnsw::<f32, DistL2>::new(64, images_as_v.len(), 16, 200, DistL2::default());
        // folloqing is necessary
        hnsw.set_keeping_pruned(true);
        //
        // we must pay fortran indexation once!. transform image to a vector
        let data_with_id: Vec<(&Vec<f32>, usize)> =
            images_as_v.iter().zip(0..images_as_v.len()).collect();
        hnsw.parallel_insert(&data_with_id);
        // dmaps
        let mut diffusion_map = DiffusionMaps::new(dparams);
        let emmbedded_res = diffusion_map.embed_from_hnsw::<f32, DistL2, f32>(&mut hnsw, &dparams);
        if emmbedded_res.is_err() {
            log::error!("embedding failed");
            panic!("dmap_fashion failed");
        };
        //
        log::info!(
            " dmap embed sys time {:.2e} s, cpu time : {}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_start.elapsed().as_secs()
        );
        // dump
        log::info!("dumping initial embedding in csv file");
        let mut csv_w = csv::Writer::from_path("mnist_fashion_dmap.csv").unwrap();
        let _res = write_csv_labeled_array2(&mut csv_w, labels.as_slice(), &emmbedded_res.unwrap());
        csv_w.flush().unwrap();
    } // end of dmap_fashion
} // end of mod tests
