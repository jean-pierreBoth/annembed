//!  Diffusion maps embedding.
//!
//! This module (presently) computes a diffusion embedding for the kernel constructed from nearest neighbours
//! stored in a Hnsw structure, see in module [embedder](crate::embedder).  
//!
//! Bibilography
//!   - *Diffusion Maps*. Coifman Lafon Appl. Comput. Harmon. Anal. 21 (2006) 5–30
//!   - *Self-Tuning Spectral Clustering*  Zelkin-Manor Perrona NIPS 2004
//!   - *From graph to manifold Laplacian: The convergence rate*. Singer Appl. Comput. Harmon. Anal. 21 (2006)
//!   - *Variables bandwith diffusion kernels* Berry and Harlim. Appl. Comput. Harmon. Anal. 40 (2016) 68–96

use num_traits::cast::FromPrimitive;
use num_traits::Float;

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
use crate::tools::{clip, kdumap::*, nodeparam::*, svdapprox::*};
use anyhow::Result;
use hnsw_rs::prelude::*;

// TODO: katex doc
/// The parameters are:
///  - the dimension of the embedding.
///  - the time of the embedding. By default it is computed using the decay of eigenvalues of the laplacian
///  - the number of neighbours to use in the comoputation of the laplacian.  
///     By default it is deduced by the number neighbours used in hnsw
///     with a limitation up to 16 (as the hnsw can require a large number of connection). To limit the cpu time it is possible to reduce it.
///     A good range is between 8 and 12.
///  - alfa
///  - beta
#[derive(Copy, Clone)]
pub struct DiffusionParams {
    /// dimension of embedding
    asked_dim: usize,
    /// kernel biaising exponent of sampling law. By default we use 0.
    alfa: f32,
    /// exponent for going from density to scales
    beta: f32,
    /// embedding time
    t: Option<f32>,
    /// number of neighbour used in the laplacian graph.
    gnbn_opt: Option<usize>,
} // end of DiffusionParams

impl DiffusionParams {
    /// arguments are:
    /// - embedding dimension
    /// - optional diffusion time
    /// - optional number of neighbours used in laplacian discretisation
    pub fn new(asked_dim: usize, t_opt: Option<f32>, g_opt: Option<usize>) -> Self {
        DiffusionParams {
            asked_dim,
            alfa: 1.,
            beta: 0.,
            t: t_opt,
            gnbn_opt: g_opt,
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
        return self.gnbn_opt;
    }
    //
    /// modify the default alfa See Lafon paper.
    /// natural values are 0. , 1/2 and 1.
    pub fn set_alfa(&mut self, alfa: f32) {
        if !(-1.01..=1.).contains(&alfa) {
            println!("not changing alfa, alfa should be in [-1. , 1.] ");
            return;
        }
        self.alfa = alfa;
    }

    // set beta, must be negative in range -1. 0.
    pub fn set_beta(&mut self, beta: f32) {
        if (-1.01..=0.).contains(&beta) {
            self.beta = beta;
        } else {
            println!("not changing beta, beta should be in -1,0 Usual values are 0. -0.5 see doc ");
            return;
        }
    }

    pub fn get_alfa(&self) -> f32 {
        self.alfa
    }

    pub fn get_beta(&self) -> f32 {
        self.beta
    }

    pub fn get_embedding_dimension(&self) -> usize {
        self.asked_dim
    }
} // end of DiffusionParams

/// The algorithm implements:
///  *Variables bandwith diffusion kernels* Berry and Harlim. Appl. Comput. Harmon. Anal. 40 (2016) 68–96.
///
/// A random svd approximation is used to run on large number of items (See module [crate::tools::svdapprox])  
pub struct DiffusionMaps {
    /// parameters to use
    params: DiffusionParams,
    /// node parameters coming from graph transformation
    _node_params: Option<NodeParams>,
    // ratio from local_scale / median_scale
    normed_scales: Option<Array1<f32>>,
    // mean scale
    mean_scale: f32,
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
            q_density: None,
            laplacian: None,
            index: None,
        }
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
        // get NodeParams.
        let (nodeparams, mut q_density) =
            self.compute_dmap_nodeparams::<F>(kgraph, kgraph.get_max_nbng());
        // compute local_scales
        let mut local_scale: Array1<f32> = nodeparams.params.iter().map(|n| n.scale).collect();
        let mut mean_scale: f32 = local_scale.iter().sum();
        mean_scale /= local_scale.len() as f32;
        for l in &mut local_scale {
            *l /= mean_scale;
        }
        //
        for i in 0..q_density.len() {
            q_density[i] /= local_scale[i]
        }
        self.get_quantiles("final q_density/scale", &q_density);
        //
        self.q_density = Some(q_density);
        self.normed_scales = Some(local_scale);
        self.mean_scale = mean_scale;
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
        // TODO:
        let scale_renormalization = true;
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
            // scale normalization
            q /= local_scale;
            let mut degrees = Array1::<f32>::zeros(q.len());
            for i in 0..nbnodes {
                let mut row = symgraph.row_mut(i);
                for j in 0..nbnodes {
                    row[[j]] /= (q[[i]] * q[[j]]).powf(alfa);
                }
                degrees[[i]] = row.sum();
            }
            // now we normalize rows according to D^-1/2 * G * D^-1/2
            for i in 0..nbnodes {
                let mut row = symgraph.row_mut(i);
                for j in 0..nbnodes {
                    row[[j]] /= (degrees[[i]] * degrees[[j]]).sqrt();
                    if scale_renormalization {
                        row[[j]] /= local_scale[i] * local_scale[j];
                    }
                }
            }
            //
            log::trace!("\n allocating full matrix laplacian");
            GraphLaplacian::new(MatRepr::from_array2(symgraph), degrees)
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
            q /= local_scale;
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
            for i in 0..values.len() {
                let row = rows[i];
                let col = cols[i];
                values[i] /= (degrees[row] * degrees[col]).sqrt();
                if scale_renormalization {
                    values[i] /= local_scale[row] * local_scale[col];
                }
            }
            log::trace!("allocating csr laplacian");
            let laplacian = TriMatBase::<Vec<usize>, Vec<f32>>::from_triplets(
                (nbnodes, nbnodes),
                rows,
                cols,
                values,
            );
            let csr_mat: CsMat<f32> = laplacian.to_csr();
            GraphLaplacian::new(MatRepr::from_csrmat(csr_mat), degrees)
        } // end case CsMat
          //
    }

    // remap node params expressing distance to proba knowing scale to adopt
    // scales must not be normalized by their mean
    // The function returns Nodeparams corresponding to new kernel and associated density (computed as mean transition proba)
    fn scales_to_kernel<F>(
        &self,
        kgraph: &KGraph<F>,
        local_scales: &Vec<F>,
        scale_ratio: f32,
        scale_to_kernel: &dyn Fn(F, f32, f32) -> f32,
    ) -> (NodeParams, Vec<f32>)
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
        log::info!("in DiffusionMaps::scales_to_kernel");
        //
        let nb_nodes = kgraph.get_nb_nodes();
        let neighbour_hood = kgraph.get_neighbours();
        //
        let mut nodeparams = Vec::<NodeParam>::with_capacity(nb_nodes);
        // we keep local scale to possible kernel weighting
        let mut q_density: Vec<f32> = Vec::<f32>::with_capacity(nb_nodes);
        //
        // now we have scales we can remap edge length to weights.
        // we choose epsil to put weight on at least 5 neighbours when no shift
        // TODO: depend on absence of shift
        let mut nb_weight_to_low = 0usize;
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
            // we add each node as a neighbour of itself to enforce ergodicity !!
            let nb_edges = 1 + neighbours.len();
            let mut edges = Vec::<OutEdge<f32>>::with_capacity(nb_edges);
            if all_equal {
                log::warn!("all equal for node {}", i);
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
                let mut sum: f32 = 0.;
                let _shift = neighbours[0].weight.to_f32().unwrap();
                let from_scale = local_scales[i];
                // TODO: no shift but could add drift with respect to local_scales variations
                for n in neighbours {
                    let to_scale = local_scales[n.node];
                    let local_scale = (to_scale * from_scale).sqrt();
                    let mut weight: f32 =
                        scale_to_kernel(n.weight, 0., scale_ratio * local_scale.to_f32().unwrap());
                    if weight < PROBA_MIN {
                        weight = weight.max(PROBA_MIN);
                        nb_weight_to_low += 1;
                    }
                    let edge = OutEdge::<f32>::new(n.node, weight);
                    edges.push(edge);
                    sum += weight;
                }
                // TODO: we adjust self_edge
                edges[0].weight = edges[1].weight / 2.;
                sum += edges[0].weight;
                q_density.push(sum / edges.len() as f32);
            }
            // allocate a NodeParam and keep track of real scale of node
            let nodep = NodeParam::new(local_scales[i].to_f32().unwrap(), edges);
            nodeparams.push(nodep);
        }
        //
        log::info!(
            "to_dmap_nodeparams: proba of weight < {:.2e} = {:.2e}",
            PROBA_MIN,
            nb_weight_to_low as f32 / nodeparams.len() as f32
        );
        //
        (
            NodeParams::new(nodeparams, kgraph.get_max_nbng()),
            q_density,
        )
    }

    /// dmap specific edge proba compuatitons
    /// - compute basic transition kernel, with global scaling from l2 distances
    /// - estimate density
    /// - re estimate scale from density with from_density_to_new_scales
    ///
    pub(crate) fn compute_dmap_nodeparams<F>(
        &self,
        kgraph: &KGraph<F>,
        nbng: usize,
    ) -> (NodeParams, Vec<f32>)
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
            "Diffusion computing kernels with using alfa : {:.2e} , beta : {:.2e}",
            self.params.get_alfa(),
            self.params.get_beta()
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
        let local_scales: Vec<F> = neighbour_hood
            .par_iter()
            //            .map(|edges| self.get_dist_around_node(kgraph, edges, nbgh_size))
            .map(|edges| self.get_dist_l2_from_node(edges, nbgh_size))
            .collect();
        // collect scales quantiles
        let scales_q = self.get_quantiles("scales quantiles first pass", &local_scales);
        println!();
        // we keep local scale to possible kernel weighting
        //
        // now we have scales we can remap edge length to weights.
        // we choose epsil to put weight on at least 5 neighbours when no shift
        // TODO: depend on absence of shift
        let exponent = 2.0f32;
        let epsil = (scales_q.query(0.99).unwrap().1 / scales_q.query(0.01).unwrap().1) as f32;
        log::info!(
            "compute_dmap_nodeparams knbn : {}, epsil : {:.2e}",
            nbgh_size,
            epsil
        );
        // from scales to proba
        let remap_weight = |w: F, shift: f32, scale: f32| {
            let arg = ((w.to_f32().unwrap() - shift) / (epsil * scale)).powf(exponent);
            (-arg).exp()
        };
        let (_, q_density) = self.scales_to_kernel(kgraph, &local_scales, epsil, &remap_weight);
        self.get_quantiles("density quantiles first pass", &q_density);
        //
        let (nodeparams, q_density) = self.density_to_kernel(
            kgraph,
            nbgh_size,
            &local_scales,
            &q_density,
            self.params.get_beta(),
            &remap_weight,
        );
        //
        self.get_quantiles("density quantiles second pass", &q_density);
        //
        (nodeparams, q_density)
    } // end compute_dmap_nodeparams

    // after a first guess for scales get re-estimate scales from q_densities
    // computed from scales estimated via get_dist_l2_from_node
    // beta parameter described in harlim-berry paper. Should be < 0.
    fn density_to_kernel<F>(
        &self,
        kgraph: &KGraph<F>,
        nbng: usize,
        local_scales_step0: &Vec<F>,
        q_densities: &Vec<f32>,
        beta: f32,
        remap_weight: &dyn Fn(F, f32, f32) -> f32,
    ) -> (NodeParams, Vec<f32>)
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
        log::info!("in DiffusionMaps::density_to_kernel, beta : {:.2e}", beta);
        //
        let nbgh_size = kgraph.get_max_nbng().min(nbng);
        log::info!(
            "compute_dmap_nodeparams kgraph nbng : {}, using size {}",
            kgraph.get_max_nbng(),
            nbgh_size
        );
        //
        // compute a scale around each node, mean scale and quantiles on scale
        let sum_scale_step0: f32 = local_scales_step0.iter().map(|s| *s).sum::<F>().into() as f32;
        let mean_scale_step0 = sum_scale_step0 / local_scales_step0.len() as f32;
        //
        log::info!("mean scale step 0: {:.2e}", mean_scale_step0);
        //
        let local_scales: Vec<F> = q_densities
            .par_iter()
            .map(|d| F::from_f32(mean_scale_step0 * d.powf(beta)).unwrap())
            .collect();
        // collect scales quantiles
        let scale_q = self.get_quantiles::<F>("scales quantiles second pass", &local_scales);
        let epsil = (scale_q.query(0.99).unwrap().1 / scale_q.query(0.01).unwrap().1) as f32;
        log::info!(
            "density_to_kernel :nbgh_size {}, epsil : {:.2e}",
            nbgh_size,
            epsil
        );
        // compute a scale around each node, mean scale and quantiles on scale
        let sum_scale: f32 = local_scales.iter().map(|s| *s).sum::<F>().into() as f32;
        let mean_scale = sum_scale / local_scales.len() as f32;
        log::info!("mean scale step 1 : {:.2e}", mean_scale);
        // Now we define kernel with new scales
        let (nodeparams, q_density) =
            self.scales_to_kernel(kgraph, &local_scales, epsil, &remap_weight);
        // got here we have local scales, densities and kernel corresponding to densities
        //
        (nodeparams, q_density)
    } // end of from_density_to_new_scales

    //

    // just dump some quantiles (mostly scales and densities)
    pub(crate) fn get_quantiles<F>(&self, message: &str, values: &Vec<F>) -> CKMS<f64>
    where
        F: Float + Into<f64>,
    {
        log::info!("\n\n {}", message);
        let mut quant_densities: CKMS<f64> = CKMS::<f64>::new(0.001);
        for q in values {
            quant_densities.insert((*q).into());
        }
        println!("quantiles at 0.01 : {:.2e}, 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}",
        quant_densities.query(0.01).unwrap().1,
        quant_densities.query(0.05).unwrap().1, quant_densities.query(0.5).unwrap().1,
        quant_densities.query(0.95).unwrap().1, quant_densities.query(0.99).unwrap().1);
        println!();
        //
        quant_densities
    }

    // computes scale (mean norm of dist) around a point
    // we compute mean of dist to first neighbour around a point given outgoing edges and graph
    #[allow(unused)]
    pub(crate) fn get_dist_around_node<F>(
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
        rho_y_s.into_iter().sum::<F>() / F::from(size).unwrap()
    } // end of get_dist_around_node

    //

    // computes scale (mean norm of dist) around a point
    // we compute L2 mean of distance from a node
    #[allow(unused)]
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
        (dist2 / F::from(out_edges.len()).unwrap()).sqrt()
    }

    // useful if we have already hnsw.
    // Note that embedded data are not reindexed to DataId
    pub(crate) fn embed_from_kgraph<F>(
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
    /// Note that
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
        let embedded = self.embed_from_hnsw_intern(&hnsw, dparams).unwrap();
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
        let svd_res = laplacian.do_svd(asked_dim + 25);
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
        // we must renormalize by
        let sum_diag = laplacian.degrees.iter().sum::<f32>();
        let scales = self.normed_scales.as_ref().unwrap();
        for i in 0..u.nrows() {
            let row_i = u.row(i);
            let weight_i = scales[i] * (laplacian.degrees[i] / sum_diag).sqrt();
            for j in 0..real_dim - 1 {
                // divide j value by diagonal and convert to F. take l_{i}^{t} as in dmap
                embedded[[i, j]] =
                    F::from_f32(normalized_lambdas[j + 1].powf(time) * row_i[j + 1] / weight_i)
                        .unwrap();
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
        let dtime = 1.;
        let gnbn: usize = 16;
        let mut dparams: DiffusionParams = DiffusionParams::new(4, Some(dtime), Some(gnbn));
        dparams.set_alfa(1.);
        dparams.set_beta(-1.);
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
        println!(
            " dmap embed time {:.2e} s, cpu time : {}",
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
        let dtime = 1.;
        let gnbn = 16;
        let mut dparams: DiffusionParams = DiffusionParams::new(20, Some(dtime), Some(gnbn));
        dparams.set_alfa(1.);
        dparams.set_beta(-1.);
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
        println!(
            " dmap embed time {:.2e} s, cpu time : {}",
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
