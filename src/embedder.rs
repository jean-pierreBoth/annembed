//! Embedding from GraphK
//!
//! The embedding is based on the graph (see [KGraph]) extracted from the Hnsw structure.  
//! Edges out a given point are given an exponential weight scaled related to the distance their neighbour.
//! This weight is modulated locally by a scale parameter computed by the mean of the distance of a point to
//! its nearest neighbour observed locally around each point.  
//!
//! **A more complete description of the model used can be found in module embedparams with hints to
//! initialize parameters**.
//!
//!  To go through the entropy optimization the type F defining the probability of edges must satisfy:  
//!
//!  F: Float + NumAssign + std::iter::Sum + num_traits::cast::FromPrimitive + Send + Sync + ndarray::ScalarOperand
//!  
//!  in fact it is f32 or f64.

use num_traits::{Float, NumAssign};

use ndarray::{Array1, Array2, ArrayView1};
// use ndarray_linalg::{Lapack, Scalar};
use lax::Lapack;

use crate::tools::io::write_csv_labeled_array2;
use csv::Writer;
use quantiles::ckms::CKMS; // we could use also greenwald_khanna

// threading needs
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::Arc;

use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use rand_distr::WeightedAliasIndex;
use rand_distr::{Distribution, Normal};

use indexmap::set::*;

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use crate::diffmaps::*;
use crate::embedparams::*;
use crate::fromhnsw::{kgproj::*, kgraph::kgraph_from_hnsw_all, kgraph::KGraph};
use crate::tools::{dichotomy::*, kdumap::*, nodeparam::*, clip::clip};
use hnsw_rs::prelude::*;

/// do not consider probabilities under PROBA_MIN, thresolded!!
pub(crate) const PROBA_MIN: f32 = 1.0E-5;

// to be used in emdedded space so small dimension. no need for simd and
#[inline]
fn distl2<F>(a: &[F], b: &[F]) -> F
where
    F: Float + ndarray::ScalarOperand + Send + Sync + std::iter::Sum,
{
    assert_eq!(a.len(), b.len());
    let norm: F = a
        .iter()
        .zip(b.iter())
        .map(|t| (*t.0 - *t.1) * (*t.0 - *t.1))
        .sum();
    num_traits::Float::sqrt(norm)
}

struct DistL2F;

impl<F> Distance<F> for DistL2F
where
    F: Float + ndarray::ScalarOperand + std::iter::Sum + Send + Sync,
{
    fn eval(&self, va: &[F], vb: &[F]) -> f32 {
        distl2::<F>(va, vb).to_f32().unwrap()
    } // end of compute
} // end of impl block

//=====================================================================================

/// The structure corresponding to the embedding process.
/// It must be initialized by the graph extracted from Hnsw according to the choosen strategy
/// and the asked dimension for embedding.
pub struct Embedder<'a, F> {
    /// graph constrcuted with fromhnsw module
    kgraph: Option<&'a KGraph<F>>,
    /// projection
    hkgraph: Option<&'a KGraphProjection<F>>,
    /// parameters
    parameters: EmbedderParams,
    /// contains edge probabilities according to the probabilized graph constructed before laplacian symetrization
    /// It is this representation that is used for cross entropy optimization!
    initial_space: Option<NodeParams>,
    /// initial embedding (option for degugging analyzing)
    initial_embedding: Option<Array2<F>>,
    /// final embedding
    embedding: Option<Array2<F>>,
} // end of Embedder

impl<'a, F> Embedder<'a, F>
where
    F: Float + Lapack + ndarray::ScalarOperand + Send + Sync + Into<f64>,
{
    /// constructor from a graph and asked embedding dimension
    pub fn new(kgraph: &'a KGraph<F>, parameters: EmbedderParams) -> Self {
        Embedder::<F> {
            kgraph: Some(kgraph),
            hkgraph: None,
            parameters,
            initial_space: None,
            initial_embedding: None,
            embedding: None,
        }
    } // end of new

    /// construction from a hierarchical graph
    pub fn from_hkgraph(
        graph_projection: &'a KGraphProjection<F>,
        parameters: EmbedderParams,
    ) -> Self {
        Embedder::<F> {
            kgraph: None,
            hkgraph: Some(graph_projection),
            parameters,
            initial_space: None,
            initial_embedding: None,
            embedding: None,
        }
    } // end of from_hkgraph

    pub fn get_asked_dimension(&self) -> usize {
        self.parameters.asked_dim
    }

    pub fn get_scale_rho(&self) -> f64 {
        self.parameters.scale_rho
    }

    pub fn get_b(&self) -> f64 {
        self.parameters.b
    }

    pub fn get_grad_step(&self) -> f64 {
        self.parameters.grad_step
    }

    pub fn get_nb_grad_batch(&self) -> usize {
        self.parameters.nb_grad_batch
    }

    /// get neighbourhood size used in embedding
    pub fn get_kgraph_nbng(&self) -> usize {
        if self.kgraph.is_some() {
            self.kgraph.as_ref().unwrap().get_max_nbng()
        } else if self.hkgraph.is_some() {
            self.hkgraph.as_ref().unwrap().get_nbng()
        } else {
            0
        }
    }

    /// dispatch to one_step embed or hierarchical embedding
    pub fn embed(&mut self) -> Result<usize, usize> {
        if self.kgraph.is_some() {
            log::info!("doing one step embedding");
            self.one_step_embed()
        } else {
            log::info!("doing 2 step embedding");
            self.h_embed()
        }
    } // end of embed

    /// do hierarchical embedding on GraphPrrojection
    pub fn h_embed(&mut self) -> Result<usize, usize> {
        if self.hkgraph.is_none() {
            log::error!("Embedder::h_embed , graph projection is none");
            return Err(1);
        }
        log::debug!("in h_embed");
        // one_step embed of the small graph.
        let graph_projection = self.hkgraph.as_ref().unwrap();
        log::info!(" embedding first (small) graph");
        let mut first_step_parameters = self.parameters;
        first_step_parameters.nb_grad_batch =
            self.parameters.grad_factor * self.parameters.nb_grad_batch;
        log::info!("nb initial batch : {}", first_step_parameters.nb_grad_batch);
        first_step_parameters.grad_step = 1.;
        let mut embedder_first_step =
            Embedder::new(graph_projection.get_small_graph(), first_step_parameters);
        let cpu_start = ProcessTime::now();
        let sys_start = SystemTime::now();
        let res_first = embedder_first_step.one_step_embed();
        if res_first.is_err() {
            log::error!("Embedder::h_embed first step failed");
            return res_first;
        }
        println!(
            " first step embedding sys time(ms) {:.2e} cpu time(ms) {:.2e}",
            sys_start.elapsed().unwrap().as_millis(),
            cpu_start.elapsed().as_millis()
        );
        // get initial embedding
        let large_graph = graph_projection.get_large_graph();
        log::info!("computing proba edges for large graph ...");
        self.initial_space = Some(to_proba_edges(
            large_graph,
            self.parameters.scale_rho as f32,
            self.parameters.beta as f32,
        ));
        let nb_nodes_large = large_graph.get_nb_nodes();
        let first_embedding = embedder_first_step.get_embedded().unwrap();
        // use projection to initialize large graph
        let quant = graph_projection.get_projection_distance_quant();
        if quant.count() > 0 {
            println!(" projection distance quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
                        quant.query(0.05).unwrap().1, quant.query(0.5).unwrap().1,
                        quant.query(0.95).unwrap().1, quant.query(0.99).unwrap().1);
        };
        let dim = self.parameters.get_dimension();
        let mut second_step_init = Array2::<F>::zeros((nb_nodes_large, dim));
        log::info!("doing projection");
        let (nb_nodes_small, _) = first_embedding.dim();
        // we were cautious on indexation so we can do:
        let mut rng = thread_rng();
        for i in 0..nb_nodes_small {
            for j in 0..dim {
                second_step_init[[i, j]] = first_embedding[[i, j]];
            }
        }
        let median_dist = quant.query(0.5).unwrap().1;
        // we sample a random position around first embedding position
        // TODO: amount of clipping
        let normal = Normal::<f32>::new(0., 1.0).unwrap();
        for i in nb_nodes_small..nb_nodes_large {
            let projected_edge = graph_projection.get_projection_by_nodeidx(&i);
            // we get looseness around projected point depend on where we are in quantiles on distance
            let ratio = projected_edge.weight.to_f32().unwrap() / median_dist;
            let correction = (ratio / dim as f32).sqrt();
            for j in 0..dim {
                let clipped_correction = clip(correction * normal.sample(&mut rng), 2.);
                second_step_init[[i, j]] = first_embedding[[projected_edge.node, j]]
                    + F::from(clipped_correction).unwrap();
            }
        }
        log::debug!("projection done");
        //
        self.initial_embedding = Some(second_step_init);
        // cross entropy optimize
        log::info!("optimizing second step");
        let embedding_res =
            self.entropy_optimize(&self.parameters, self.initial_embedding.as_ref().unwrap());
        //
        println!(
            " first + second step embedding sys time(s) {:.2e} cpu time(s) {:.2e}",
            sys_start.elapsed().unwrap().as_secs(),
            cpu_start.elapsed().as_secs()
        );
        //
        match embedding_res {
            Ok(embedding) => {
                self.embedding = Some(embedding);
                Ok(1)
            }
            _ => {
                log::error!("Embedder::embed : embedding optimization failed");
                Err(1)
            }
        }
    } // end of h_embed

    /// do the embedding
    pub fn one_step_embed(&mut self) -> Result<usize, usize> {
        //
        log::info!("doing 1 step embedding");
        self.parameters.log();
        let graph_to_embed = self.kgraph.unwrap();
        // construction of initial neighbourhood, scales and proba of edges from distances.
        // we will need  initial_space representation for graph laplacian and in cross entropy optimization

        // we can initialize embedding with diffusion maps or pure random.
        let mut initial_embedding;
        if self.parameters.dmap_init {
            // initial embedding via diffusion maps, in this case we have to have a coherent box normalization with random case
            //
            let cpu_start = ProcessTime::now();
            let sys_start = SystemTime::now();
            let dmapnew = false;
            //
            if dmapnew {
                log::info!("using new dmaps");
                let dtime = 5.;
                let gnbn = 16;
                let mut dparams: DiffusionParams = DiffusionParams::new(10, Some(dtime), Some(gnbn));
                dparams.set_alfa(1.);
                dparams.set_beta(-1.);
                let mut diffusion_map = DiffusionMaps::new(dparams);
               initial_embedding = diffusion_map.embed_from_kgraph::<F>(graph_to_embed, &dparams).unwrap();
            }
            else {
                log::info!("using old dmaps");
                self.initial_space = Some(to_proba_edges(
                graph_to_embed,
                self.parameters.scale_rho as f32,
                self.parameters.beta as f32,
                ));  
                initial_embedding = get_dmap_embedding(
                    self.initial_space.as_ref().unwrap(),
                    self.parameters.get_dimension(),
                    None,
                );
            }
            
            println!(
                " dmap initialization sys time(ms) {:.2e} cpu time(ms) {:.2e}",
                sys_start.elapsed().unwrap().as_millis(),
                cpu_start.elapsed().as_millis()
            );
            set_data_box(&mut initial_embedding, F::from(10.).unwrap());
        } else {
            // if we use random initialization we must have a box size coherent with renormalizes scales, so box size is 1.
            initial_embedding = self.get_random_init(1.);
        }
        //
        self.initial_space = Some(to_proba_edges(
            graph_to_embed,
            self.parameters.scale_rho as f32,
            self.parameters.beta as f32,
        ));
        let embedding_res = self.entropy_optimize(&self.parameters, &initial_embedding);
        // optional store dump initial embedding
        self.initial_embedding = Some(initial_embedding);
        //
        match embedding_res {
            Ok(embedding) => {
                self.embedding = Some(embedding);
                Ok(1)
            }
            _ => {
                log::error!("Embedder::embed : embedding optimization failed");
                Err(1)
            }
        }
    } // end embed

    /// At the end returns the embedded data as Matrix.
    /// The row of the matrix corresponds to the embedded dat vectors but after reindexation of DataId
    /// to ensure a contiguous indexation.  
    /// To get a matrix with row corresponding to DataId if they were already contiguous for 0 to nbdata use
    /// function  get_embedded_reindexed to get the permutation/reindexation unrolled!
    pub fn get_embedded(&self) -> Option<&Array2<F>> {
        self.embedding.as_ref()
    }

    /// returns embedded data reindexed by DataId. This requires the DataId to be contiguous from 0 to nbdata.  
    ///  See [crate::fromhnsw::kgraph::KGraph::get_idx_from_dataid]
    pub fn get_embedded_reindexed(&self) -> Array2<F> {
        let emmbedded = self.embedding.as_ref().unwrap();
        let (nbrow, dim) = emmbedded.dim();
        let mut reindexed = Array2::<F>::zeros((nbrow, dim));
        //
        let kgraph = if self.hkgraph.is_some() {
            self.hkgraph.as_ref().unwrap().get_large_graph()
        } else {
            self.kgraph.as_ref().unwrap()
        };
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (One week bug!)
        for i in 0..nbrow {
            let row = emmbedded.row(i);
            let origin_id = kgraph.get_data_id_from_idx(i).unwrap();
            for j in 0..dim {
                reindexed[[*origin_id, j]] = row[j];
            }
        }
        reindexed
    }

    /// **return the embedded vector corresponding to original data vector corresponding to data_id**
    /// This methods fails if data_id do not exist. Use KGraph.get_data_id_from_idx to check before if necessary.
    pub fn get_embedded_by_dataid(&self, data_id: &DataId) -> ArrayView1<F> {
        // we must get data index as stored in IndexSet
        let kgraph = if self.hkgraph.is_some() {
            self.hkgraph.as_ref().unwrap().get_large_graph()
        } else {
            self.kgraph.as_ref().unwrap()
        };
        let data_idx = kgraph.get_idx_from_dataid(data_id).unwrap();
        self.embedding.as_ref().unwrap().row(data_idx)
    } // end of get_data_embedding

    /// **get embedding of a given node index after reindexation by the embedding to index in [0..nb_nodes]**
    pub fn get_embedded_by_nodeid(&self, node: NodeIdx) -> ArrayView1<F> {
        self.embedding.as_ref().unwrap().row(node)
    }

    /// returns the initial embedding. Same remark as for method get_embedded. Storage is optional TODO
    pub fn get_initial_embedding(&self) -> Option<&Array2<F>> {
        self.initial_embedding.as_ref()
    }

    pub fn get_initial_embedding_reindexed(&self) -> Array2<F> {
        //
        let emmbedded = self.initial_embedding.as_ref().unwrap();
        let (nbrow, dim) = emmbedded.dim();
        let mut reindexed = Array2::<F>::zeros((nbrow, dim));
        //
        let kgraph = if self.hkgraph.is_some() {
            self.hkgraph.as_ref().unwrap().get_large_graph()
        } else {
            self.kgraph.as_ref().unwrap()
        };
        //
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (One week bug!)
        for i in 0..nbrow {
            let row = emmbedded.row(i);
            let origin_id = kgraph.get_data_id_from_idx(i).unwrap();
            for j in 0..dim {
                reindexed[[*origin_id, j]] = row[j];
            }
        }
        reindexed
    } // end of get_initial_embedding_reindexed

    /// get random initialization in a square of side size
    fn get_random_init(&self, size: f32) -> Array2<F> {
        log::trace!("Embedder get_random_init with size {:.2e}", size);
        //
        let nb_nodes = self.initial_space.as_ref().unwrap().get_nb_nodes();
        let mut initial_embedding = Array2::<F>::zeros((nb_nodes, self.get_asked_dimension()));
        let unif = Uniform::<f32>::new(-size / 2., size / 2.);
        let mut rng = thread_rng();
        for i in 0..nb_nodes {
            for j in 0..self.get_asked_dimension() {
                initial_embedding[[i, j]] = F::from(rng.sample(unif)).unwrap();
            }
        }
        //
        initial_embedding
    } // end of get_random_init

    /// For each node we in the orginal kgraph compute the transformed neighbourhood info in the embedded space.
    /// Precisely: for each node n1 in initial space, for each neighbour n2 of n1, we compute l2dist of
    /// embedded points corresponding to n1, n2.
    /// So we have an embedded edgewhich is not always an edge in kgraph computed from embedded data.  
    /// The conservation of edges through the embedding is a measure of neighborhood conservation.  
    /// Function returns for each node , increasing sorted distances (L2) in embedded space to its neighbours in original space.
    fn get_transformed_kgraph(&self) -> Option<Vec<(usize, Vec<OutEdge<F>>)>> {
        // we check we have kgraph
        let kgraph: &KGraph<F>;
        if self.kgraph.is_some() {
            kgraph = self.kgraph.unwrap();
            log::info!("found kgraph");
        } else if self.hkgraph.is_some() {
            kgraph = self.hkgraph.as_ref().unwrap().get_large_graph();
            log::info!("found kgraph from projection");
        } else {
            log::info!("could not find kgraph");
            std::process::exit(1);
        }
        // we loop on kgraph nodes, loop on edges of node, get extremity id , converts to index, compute embedded distance and sum
        let neighbours = kgraph.get_neighbours();
        //
        let transformed_neighbours: Vec<(usize, Vec<OutEdge<F>>)> = (0..neighbours.len())
            .into_par_iter()
            .map(|n| -> (usize, Vec<OutEdge<F>>) {
                let node_embedded = self.get_embedded_by_nodeid(n);
                let mut transformed_neighborhood =
                    Vec::<OutEdge<F>>::with_capacity(neighbours[n].len());
                let mut node_edge_length = F::max_value();
                for edge in &neighbours[n] {
                    let ext_embedded = self.get_embedded_by_nodeid(edge.node);
                    // now we can compute distance for corresponding edge in embedded space. We must use L2
                    node_edge_length = distl2(
                        node_embedded.as_slice().unwrap(),
                        ext_embedded.as_slice().unwrap(),
                    )
                    .min(node_edge_length);
                    transformed_neighborhood.push(OutEdge::<F>::new(edge.node, node_edge_length));
                }
                // sort transformed_neighborhood
                transformed_neighborhood
                    .sort_unstable_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());
                (n, transformed_neighborhood)
            })
            .collect();
        // We need to ensure that parallel iter did not permute data.
        for (i, item) in transformed_neighbours.iter().enumerate() {
            assert_eq!(i, item.0);
        }
        Some(transformed_neighbours)
    } // end of get_transformed_kgraph

    /// compute hnsw and kgraph from embedded data and get maximal edge length by node
    /// Returns for each node, distance of nbng'th neighbours to node.
    /// This function can require much memory for large number of nodes with large nbng
    fn get_max_edge_length_embedded_kgraph(&self, nbng: usize) -> Option<Vec<(usize, f64)>> {
        let embedding = self.embedding.as_ref().unwrap();
        // TODO use the same parameters as the hnsw given to kgraph, and adjust ef_c accordingly
        let max_nb_connection = nbng;
        let ef_c = 64;
        // compute hnsw
        let nb_nodes = embedding.nrows();
        let nb_layer = 16.min((nb_nodes as f32).ln().trunc() as usize);
        let mut hnsw =
            Hnsw::<F, DistL2F>::new(max_nb_connection, nb_nodes, nb_layer, ef_c, DistL2F {});
        hnsw.set_keeping_pruned(true);
        // need to store arrayviews to get a sufficient lifetime to call as_slice later
        let vectors: Vec<ArrayView1<F>> = (0..nb_nodes).map(|i| (embedding.row(i))).collect();
        let mut data_with_id = Vec::<(&[F], usize)>::with_capacity(nb_nodes);
        for (i, v) in vectors.iter().enumerate().take(nb_nodes) {
            data_with_id.push((v.as_slice().unwrap(), i));
        }
        hnsw.parallel_insert_slice(&data_with_id);
        // compute kgraph from hnsw and sum edge length
        let optimal_graph: anyhow::Result<KGraph<F>> = kgraph_from_hnsw_all(&hnsw, nbng);
        if optimal_graph.is_err() {
            log::error!("could not compute optimal graph");
            return None;
        }
        let optimal_graph = optimal_graph.unwrap();
        let optimal_max_edge = optimal_graph.compute_max_edge();
        Some(optimal_max_edge)
    } // end of get_max_edge_length_embedded_kgraph

    /// This function is an attempt to quantify the quality of the embedding using the graph projection defined in [KGraph].  
    ///  
    /// The graph projection defines a natural neighborhood of a point as the set of edges around a point. The size of this neighbourhood is related to
    /// the parameter *max_nb_connection* in Hnsw and roughly correspond to a number of neighbours between 2 and 3 times this parameter.  
    /// The argument *nbng* to this function should be set accordingly.
    ///   
    /// It tries to assess how neighbourhood of points in original space and neighbourhood in embedded space match.
    /// The size of neighbourhood considered is *nbng*.    
    ///
    /// In each neighbourhood of a point, taken as center in the initial space we:
    ///
    /// - count the number of its neighbours for which the distance to the center is less than the radius of the neighbourhood (of size *nbng*) in embedded space.
    ///   For neighbourhood that have a match , we give the mean number of matches.    
    ///   This quantify the conservation of neighborhoods through the embedding. The lower the number of neighbourhoods without a match and the higher
    ///   the mean number of matches, the better is the embedding.
    ///  
    /// - compute the length of embedded edges joining original neighbours to a node and provide quantiles summary.
    ///
    /// - compute the ratio of these edge length to the radius of the ball in embedded space corresponding to nbng 'th neighbours.
    ///   (question is: how much do we need to dilate the neighborhood in embedded space to retrieve the neighbours in original space?)  
    ///
    /// The quantiles on ratio these distance are then dumped. The lower the median (or the mean), the better is the embedding.
    ///   
    /// It gives a rough idea of the continuity of the embedding. The lesser the ratio the tighter the embedding.
    ///
    /// For example for the **fashion mnist** in the hierarchical case we get consistently (see examples):
    ///
    /// - With an embedding dimension of 2 and a target neighbourhood of 50:
    /// ```text
    ///  a guess at quality
    ///  neighbourhood size used in embedding : 6
    ///  nb neighbourhoods without a match : 20260,  mean number of neighbours conserved when match : 5.069e0
    ///  embedded radii quantiles at 0.05 : 3.15e-2 , 0.25 : 4.52e-2, 0.5 :  5.68e-2, 0.75 : 7.80e-2, 0.85 : 9.32e-2, 0.95 : 1.36e-1
    ///
    ///  statistics on conservation of neighborhood (of size nbng)
    ///  neighbourhood size used in target space : 50
    ///  quantiles on ratio : distance in embedded space of neighbours of origin space / distance of last neighbour in embedded space
    ///
    ///  quantiles at 0.05 : 5.40e-2 , 0.25 : 3.28e-1, 0.5 :  7.46e-1, 0.75 : 1.57e0, 0.85 : 2.38e0, 0.95 : 4.50e0
    /// ```
    ///
    /// So 20000 out of 70000 data points have no conserved neighbours, for the others points 5 out of 6 neighbours are retrived.
    /// 75% of neighbourhood is conserved within a radius increased by a factor 1.57.
    ///
    /// - With an embedding dimension of 15 and a target neighbourhood of 50:
    ///
    /// ```text
    /// a guess at quality
    /// neighbourhood size used in embedding : 6
    /// nb neighbourhoods without a match : 9124,  mean number of neighbours conserved when match : 5.585e0
    /// embedded radii quantiles at 0.05 : 5.55e-2 , 0.25 : 8.66e-2, 0.5 :  1.15e-1, 0.75 : 1.53e-1, 0.85 : 1.80e-1, 0.95 : 2.41e-1
    ///
    /// statistics on conservation of neighborhood (of size nbng)
    /// neighbourhood size used in target space : 50
    /// quantiles on ratio : distance in embedded space of neighbours of origin space / distance of last neighbour in embedded space
    /// quantiles at 0.05 : 5.03e-2 , 0.25 : 2.24e-1, 0.5 :  4.36e-1, 0.75 : 8.06e-1, 0.85 : 1.13e0, 0.95 : 1.92e0
    /// ```
    /// So 9000 out of 70000 data points have no conserved neighbours, for the others points 5.6 out of 6 neighbours are retrieved.
    /// 75% of neighbourhood is conserved within a radius increased by a factor 0.8.
    // called by external binary (see examples)
    #[allow(unused)]
    pub fn get_quality_estimate_from_edge_length(&self, nbng: usize) -> Option<f64> {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let quality = 0f64;
        // compute min edge length from initial kgraph
        let transformed_kgraph = self.get_transformed_kgraph();
        if transformed_kgraph.is_none() {
            log::error!("cannot ask for embedded quality before embedding");
            std::process::exit(1);
        }
        let transformed_kgraph = transformed_kgraph.unwrap();
        // compute max edge length from kgraph constructed from embedded points corresponding to nbng neighbours
        let max_edges_embedded = self.get_max_edge_length_embedded_kgraph(nbng);
        if max_edges_embedded.is_none() {
            log::error!("get_quality_estimate_from_edge_length : cannot compute mean edge length from embedded data");
            return None;
        }
        let max_edges_embedded = max_edges_embedded.unwrap();
        // now we can for each node see if best of propagated initial edges encounter ball in reconstructed kgraph from embedded data
        assert_eq!(max_edges_embedded.len(), transformed_kgraph.len());

        let mut embedded_radii = CKMS::<f64>::new(0.01);
        let mut ratio_dist_q = CKMS::<f64>::new(0.01);
        let nb_nodes = max_edges_embedded.len();
        let mut nodes_match = Vec::with_capacity(nb_nodes);
        let mut first_dist = Vec::with_capacity(nb_nodes);
        let mut mean_ratio = (0., 0usize);
        for i in 0..nb_nodes {
            // check we speak of same node
            assert_eq!(i, max_edges_embedded[i].0);
            assert_eq!(i, transformed_kgraph[i].0);
            nodes_match.push(0);
            embedded_radii.insert(max_edges_embedded[i].1);
            // how many transformed edges are in maximal neighborhood of size nbng?
            let neighbours = &transformed_kgraph[i].1;
            for e in neighbours {
                if e.weight.to_f64().unwrap() <= max_edges_embedded[i].1 {
                    nodes_match[i] += 1;
                }
                ratio_dist_q.insert(e.weight.to_f64().unwrap() / max_edges_embedded[i].1);
                mean_ratio.0 += e.weight.to_f64().unwrap() / max_edges_embedded[i].1;
            }
            mean_ratio.1 += neighbours.len();
            first_dist.push(neighbours[0].weight.to_f64().unwrap());
        }
        // some stats
        let nb_without_match = nodes_match
            .iter()
            .fold(0, |acc, x| if *x == 0 { acc + 1 } else { acc });
        let mean_nbmatch: f64 = nodes_match.iter().sum::<usize>() as f64
            / (nodes_match.len() - nb_without_match) as f64;
        println!("\n\n a guess at quality ");
        println!(
            "  neighbourhood size used in embedding : {}",
            self.get_kgraph_nbng()
        );
        println!("  nb neighbourhoods without a match : {},  mean number of neighbours conserved when match : {:.3e}", nb_without_match,  mean_nbmatch);
        println!("  embedded radii quantiles at 0.05 : {:.2e} , 0.25 : {:.2e}, 0.5 :  {:.2e}, 0.75 : {:.2e}, 0.85 : {:.2e}, 0.95 : {:.2e} \n", 
            embedded_radii.query(0.05).unwrap().1, embedded_radii.query(0.25).unwrap().1, embedded_radii.query(0.5).unwrap().1,
            embedded_radii.query(0.75).unwrap().1, embedded_radii.query(0.85).unwrap().1, embedded_radii.query(0.95).unwrap().1);
        //
        // The smaller the better!
        // we give quantiles on ratio : distance of neighbours in origin space / distance of last neighbour in embedded space
        println!("\n statistics on conservation of neighborhood (of size nbng)");
        println!("  neighbourhood size used in target space : {}", nbng);
        println!("  quantiles on ratio : distance in embedded space of neighbours of origin space / distance of last neighbour in embedded space");
        println!("  quantiles at 0.05 : {:.2e} , 0.25 : {:.2e}, 0.5 :  {:.2e}, 0.75 : {:.2e}, 0.85 : {:.2e}, 0.95 : {:.2e} \n", 
            ratio_dist_q.query(0.05).unwrap().1, ratio_dist_q.query(0.25).unwrap().1, ratio_dist_q.query(0.5).unwrap().1,
            ratio_dist_q.query(0.75).unwrap().1, ratio_dist_q.query(0.85).unwrap().1, ratio_dist_q.query(0.95).unwrap().1);

        let median_ratio = ratio_dist_q.query(0.5).unwrap().1;
        println!("\n quality index: ratio of distance to neighbours in origin space / distance to last neighbour in embedded space");
        println!(
            "  neighborhood are conserved in radius multiplied by median  : {:.2e}, mean {:.2e} ",
            median_ratio,
            mean_ratio.0 / mean_ratio.1 as f64
        );
        println!();
        //
        let mut csv_dist = Writer::from_path("first_dist.csv").unwrap();
        let _res = write_csv_labeled_array2(
            &mut csv_dist,
            first_dist.as_slice(),
            &self.get_embedded_reindexed(),
        );
        csv_dist.flush().unwrap();
        //
        let cpu_time: Duration = cpu_start.elapsed();
        log::info!(
            " quality estimation,  sys time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_time.as_secs()
        );
        //
        Some(quality)
    } // end of get_quality_estimate_from_edge_length

    // given neighbours of a node we choose scale to satisfy a normalization constraint.
    // p_i = exp[- beta * (d(x,y_i) - d(x, y_1)/ local_scale ]
    // We return beta/local_scale
    // as function is monotonic with respect to scale, we use dichotomy.
    #[allow(unused)]
    fn get_scale_from_umap(&self, norm: f64, neighbours: &[OutEdge<F>]) -> (f32, Vec<f32>) {
        // p_i = exp[- beta * (d(x,y_i)/ local_scale) ]
        let nbgh = neighbours.len();
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut dist = neighbours
            .iter()
            .map(|n| n.weight.to_f32().unwrap())
            .collect::<Vec<f32>>();
        //
        let f = |beta: f32| {
            dist.iter()
                .map(|d| (-(d - rho_x) * beta).exp())
                .sum::<f32>()
        };
        // f is decreasing
        // TODO we could also normalize as usual?
        let beta = dichotomy_solver(false, f, 0f32, f32::MAX, norm as f32).unwrap();
        // reuse rho_y_s to return proba of edge
        for d in dist.iter_mut() {
            *d = (-(*d - rho_x) * beta).exp();
        }
        // in this state neither sum of proba adds up to 1 neither is any entropy (Shannon or Renyi) normailed.
        (1. / beta, dist)
    } // end of get_scale_from_umap

    pub fn get_nb_nodes(&self) -> usize {
        self.initial_space.as_ref().unwrap().get_nb_nodes()
    }

    // minimize divergence between embedded and initial distribution probability
    // We use cross entropy as in Umap. The edge weight function must take into acccount an initial density estimate and a scale.
    // The initial density makes the embedded graph asymetric as the initial graph.
    // The optimization function thus should try to restore asymetry and local scale as far as possible.
    // returns the embedded data after restauration of the original indexation/identification of datas! (time consuming bug)
    fn entropy_optimize(
        &self,
        params: &EmbedderParams,
        initial_embedding: &Array2<F>,
    ) -> Result<Array2<F>, String> {
        //
        log::debug!("in Embedder::entropy_optimize");
        //
        if self.initial_space.is_none() {
            log::error!("Embedder::entropy_optimize : initial_space not constructed, exiting");
            return Err(String::from(
                " initial_space not constructed, no NodeParams",
            ));
        }
        let ce_optimization = EntropyOptim::new(
            self.initial_space.as_ref().unwrap(),
            params,
            initial_embedding,
        );
        // compute initial value of objective function
        let start = ProcessTime::now();
        let initial_ce = ce_optimization.ce_compute_threaded();
        let cpu_time: Duration = start.elapsed();
        println!(
            " initial cross entropy value {:.2e},  in time {:?}",
            initial_ce, cpu_time
        );
        // We manage some iterations on gradient computing
        let grad_step_init = params.grad_step;
        log::info!("grad_step_init : {:.2e}", grad_step_init);
        //
        log::debug!("in Embedder::entropy_optimize  ... gradient iterations");
        let nb_sample_by_iter = params.nb_sampling_by_edge * ce_optimization.get_nb_edges();
        //
        log::info!("\n optimizing embedding");
        log::info!(
            " nb edges {} , number of edge sampling by grad iteration {}",
            ce_optimization.get_nb_edges(),
            nb_sample_by_iter
        );
        log::info!(
            " nb iteration : {}  sampling size {} ",
            self.get_nb_grad_batch(),
            nb_sample_by_iter
        );
        let cpu_start = ProcessTime::now();
        let sys_start = SystemTime::now();
        for iter in 1..=self.get_nb_grad_batch() {
            // loop on edges
            let grad_step = grad_step_init * (1. - iter as f64 / self.get_nb_grad_batch() as f64);
            ce_optimization.gradient_iteration_threaded(nb_sample_by_iter, grad_step);
            //            let cpu_time: Duration = start.elapsed();
            //            log::debug!("ce after grad iteration time(ms) {:.2e} grad iter {:.2e}",  cpu_time.as_millis(), ce_optimization.ce_compute_threaded());
        }
        println!(
            " gradient iterations sys time(s) {:.2e} , cpu_time(s) {:.2e}",
            sys_start.elapsed().unwrap().as_secs(),
            cpu_start.elapsed().as_secs()
        );
        let final_ce = ce_optimization.ce_compute_threaded();
        println!(" final cross entropy value {:.2e}", final_ce);
        // return reindexed data (if possible)
        let dim = self.get_asked_dimension();
        let nbrow = self.get_nb_nodes();
        let mut reindexed = Array2::<F>::zeros((nbrow, dim));
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (One week bug!)
        for i in 0..nbrow {
            let row = ce_optimization.get_embedded_data(i);
            for j in 0..dim {
                reindexed[[i, j]] = row.read()[j];
            }
        }
        //
        Ok(reindexed)
        //
    } // end of entropy_optimize
} // end of impl Embedder

//==================================================================================================================

/// All we need to optimize entropy discrepancy
/// A list of edge with its weight, an array of scale for each origin node of an edge, proba (weight) of each edge
/// and coordinates in embedded_space with lock protection for //
struct EntropyOptim<'a, F> {
    /// initial space by neighbourhood of each node
    node_params: &'a NodeParams,
    /// for each edge , initial node, end node, proba (weight of edge) 24 bytes
    edges: Vec<(NodeIdx, OutEdge<f32>)>,
    /// embedded coordinates of each node, under RwLock to // optimization     nbnodes * (embedded dim * f32 + lock size))
    embedded: Vec<Arc<RwLock<Array1<F>>>>,
    /// embedded_scales
    embedded_scales: Vec<f32>,
    /// weighted array for sampling positive edges
    pos_edge_distribution: WeightedAliasIndex<f32>,
    /// embedding parameters
    params: &'a EmbedderParams,
} // end of EntropyOptim

impl<'a, F> EntropyOptim<'a, F>
where
    F: Float
        + NumAssign
        + std::iter::Sum
        + num_traits::cast::FromPrimitive
        + Send
        + Sync
        + ndarray::ScalarOperand,
{
    //
    pub fn new(
        node_params: &'a NodeParams,
        params: &'a EmbedderParams,
        initial_embed: &Array2<F>,
    ) -> Self {
        log::debug!("entering EntropyOptim::new");
        // TODO what if not the same number of neighbours!!
        let nbng = node_params.params[0].edges.len();
        let nbnodes = node_params.get_nb_nodes();
        let mut edges = Vec::<(NodeIdx, OutEdge<f32>)>::with_capacity(nbnodes * nbng);
        let mut edges_weight = Vec::<f32>::with_capacity(nbnodes * nbng);
        let mut initial_scales = Vec::<f32>::with_capacity(nbnodes);
        // construct field edges
        for i in 0..nbnodes {
            initial_scales.push(node_params.params[i].scale);
            for j in 0..node_params.params[i].edges.len() {
                edges.push((i, node_params.params[i].edges[j]));
                edges_weight.push(node_params.params[i].edges[j].weight);
            }
        }
        log::debug!("construction alias table for sampling edges..");
        let start = ProcessTime::now();
        let pos_edge_sampler = WeightedAliasIndex::new(edges_weight).unwrap();
        let cpu_time: Duration = start.elapsed();
        log::debug!(
            "constructied alias table for sampling edges.. , time : {:?}",
            cpu_time
        );
        // construct embedded, initial embed can be droped now
        let mut embedded = Vec::<Arc<RwLock<Array1<F>>>>::new();
        let nbrow = initial_embed.nrows();
        for i in 0..nbrow {
            embedded.push(Arc::new(RwLock::new(initial_embed.row(i).to_owned())));
        }
        // compute embedded scales
        let embedded_scales = estimate_embedded_scales_from_initial_scales(&initial_scales);
        // get qunatile on embedded scales
        let mut scales_q = CKMS::<f32>::new(0.001);
        for s in &embedded_scales {
            scales_q.insert(*s);
        }
        println!("\n\n embedded scales quantiles at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}",
        scales_q.query(0.05).unwrap().1, scales_q.query(0.5).unwrap().1, 
        scales_q.query(0.95).unwrap().1, scales_q.query(0.99).unwrap().1);
        println!();
        //
        EntropyOptim {
            node_params,
            edges,
            embedded,
            embedded_scales,
            pos_edge_distribution: pos_edge_sampler,
            params,
        }
        // construct field embedded
    } // end of new

    pub fn get_nb_edges(&self) -> usize {
        self.edges.len()
    } // end of get_nb_edges

    // return result as an Array2<F> cloning data to result to struct Embedder
    // We return data in rows as (re)indexed in graph construction after hnsw!!
    #[allow(unused)]
    fn get_embedded_raw(&self) -> Array2<F> {
        let nbrow = self.embedded.len();
        let nbcol = self.params.asked_dim;
        let mut embedding_res = Array2::<F>::zeros((nbrow, nbcol));
        // TODO version 0.15 provides move_into and push_row
        //
        for i in 0..nbrow {
            let row = self.embedded[i].read();
            for j in 0..nbcol {
                embedding_res[[i, j]] = row[j];
            }
        }
        embedding_res
    }

    // return result as an Array2<F> cloning data to return result to struct Embedder
    // Here we reindex data according to their original order. DataId must be contiguous and fill [0..nb_data]
    // for this method not to panic!
    #[allow(unused)]
    pub fn get_embedded_reindexed(&self, indexset: &IndexSet<NodeIdx>) -> Array2<F> {
        let nbrow = self.embedded.len();
        let nbcol = self.params.asked_dim;
        let mut embedding_res = Array2::<F>::zeros((nbrow, nbcol));
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (use One week bug!
        for i in 0..nbrow {
            let row = self.embedded[i].read();
            let origin_id = indexset.get_index(i).unwrap();
            for j in 0..nbcol {
                embedding_res[[*origin_id, j]] = row[j];
            }
        }
        embedding_res
    }

    /// row is here a NodeIdx (after reindexation of DataId), hence the not public interface
    fn get_embedded_data(&self, row: usize) -> Arc<RwLock<Array1<F>>> {
        Arc::clone(&self.embedded[row])
    }

    // computes croos entropy between initial space and embedded distribution.
    // necessary to monitor optimization
    #[allow(unused)]
    fn ce_compute(&self) -> f64 {
        log::trace!("\n entering EntropyOptim::ce_compute");
        //
        let mut ce_entropy = 0.;
        let b: f64 = self.params.b;

        for edge in self.edges.iter() {
            let node_i = edge.0;
            let node_j = edge.1.node;
            assert!(node_i != node_j);
            let weight_ij = edge.1.weight as f64;
            let weight_ij_embed = cauchy_edge_weight(
                &self.embedded[node_i].read(),
                self.embedded_scales[node_i] as f64,
                b,
                &self.embedded[node_j].read(),
            )
            .to_f64()
            .unwrap();
            if weight_ij_embed > 0. {
                ce_entropy += -weight_ij * weight_ij_embed.ln();
            }
            if weight_ij_embed < 1. {
                ce_entropy += -(1. - weight_ij) * (1. - weight_ij_embed).ln();
            }
            if !ce_entropy.is_finite() {
                log::debug!(
                    "weight_ij {} weight_ij_embed {}",
                    weight_ij,
                    weight_ij_embed
                );
                std::panic!();
            }
        }
        //
        ce_entropy
    } // end of ce_compute

    // threaded version for computing cross entropy between initial distribution and embedded distribution with Cauchy law.
    fn ce_compute_threaded(&self) -> f64 {
        log::trace!("\n entering EntropyOptim::ce_compute_threaded");
        //
        let b: f64 = self.params.b;
        let ce_entropy = self
            .edges
            .par_iter()
            .fold(
                || 0.0f64,
                |entropy: f64, edge| {
                    entropy + {
                        let node_i = edge.0;
                        let node_j = edge.1.node;
                        let weight_ij = edge.1.weight as f64;
                        let weight_ij_embed = cauchy_edge_weight(
                            &self.embedded[node_i].read(),
                            self.embedded_scales[node_i] as f64,
                            b,
                            &self.embedded[node_j].read(),
                        )
                        .to_f64()
                        .unwrap();
                        let mut term = 0.;
                        if weight_ij_embed > 0. {
                            term += -weight_ij * weight_ij_embed.ln();
                        }
                        if weight_ij_embed < 1. {
                            term += -(1. - weight_ij) * (1. - weight_ij_embed).ln();
                        }
                        term
                    }
                },
            )
            .sum::<f64>();
        //
        ce_entropy
    } // end of ce_compute_threaded

    // TODO : pass functions corresponding to edge_weight and grad_edge_weight as arguments to test others weight function
    /// This function optimize cross entropy for Shannon cross entropy
    fn ce_optim_edge_shannon(&self, threaded: bool, grad_step: f64)
    where
        F: Float
            + NumAssign
            + std::iter::Sum
            + num_traits::cast::FromPrimitive
            + ndarray::ScalarOperand,
    {
        //
        let edge_idx_sampled: usize;
        let mut y_i;
        let mut y_j;
        let node_j;
        let node_i;
        if threaded {
            edge_idx_sampled = thread_rng().sample(&self.pos_edge_distribution);
            node_i = self.edges[edge_idx_sampled].0;
            node_j = self.edges[edge_idx_sampled].1.node;
            y_i = self.get_embedded_data(node_i).read().to_owned();
            y_j = self.get_embedded_data(node_j).read().to_owned();
        }
        // end threaded
        else {
            edge_idx_sampled = thread_rng().sample(&self.pos_edge_distribution);
            node_i = self.edges[edge_idx_sampled].0;
            y_i = self.get_embedded_data(node_i).write().to_owned();
            node_j = self.edges[edge_idx_sampled].1.node;
            y_j = self.get_embedded_data(node_j).write().to_owned();
        };
        // get coordinate of node
        // we locks once and directly a write lock as conflicts should be small, many edges, some threads. see Recht Hogwild!
        let dim = self.params.asked_dim;
        let mut gradient = Array1::<F>::zeros(dim);
        //
        assert!(node_i != node_j);
        let weight = self.edges[edge_idx_sampled].1.weight as f64;
        assert!(weight <= 1.);
        let scale = self.embedded_scales[node_i] as f64;
        let b: f64 = self.params.b;
        // compute l2 norm of y_j - y_i
        let d_ij: f64 = y_i
            .iter()
            .zip(y_j.iter())
            .map(|(vi, vj)| (*vi - *vj) * (*vi - *vj))
            .sum::<F>()
            .to_f64()
            .unwrap();
        let d_ij_scaled = d_ij / (scale * scale);
        // this coeff is common for P and 1.-P part
        let coeff: f64 = if b != 1. {
            let cauchy_weight = 1. / (1. + d_ij_scaled.powf(b));
            2. * b * cauchy_weight * d_ij_scaled.powf(b - 1.) / (scale * scale)
        } else {
            let cauchy_weight = 1. / (1. + d_ij_scaled);
            2. * b * cauchy_weight / (scale * scale)
        };
        if d_ij_scaled > 0. {
            // repulsion annhinilate  attraction if P<= 1. / (alfa + 1). choose 0.1/ PROBA_MIN
            let alfa = (1. / PROBA_MIN) as f64;
            let coeff_repulsion = 1. / (d_ij_scaled * d_ij_scaled).max(alfa);
            // clipping makes each point i or j making at most half way to the other in case of attraction
            let coeff_ij =
                (grad_step * coeff * (-weight + (1. - weight) * coeff_repulsion)).max(-0.49);
            gradient = (&y_j - &y_i) * F::from(coeff_ij).unwrap();
            log::trace!(
                "norm attracting coeff {:.2e} gradient {:.2e}",
                coeff_ij,
                l2_norm(&gradient.view()).to_f64().unwrap()
            );
        }
        y_i -= &gradient;
        y_j += &gradient;
        *(self.get_embedded_data(node_j).write()) = y_j;
        // now we loop on negative sampling filtering out nodes that are either node_i or are in node_i neighbours.
        let asked_nb_neg = 5;
        let mut got_nb_neg = 0;
        let mut _nb_failed = 0;
        while got_nb_neg < asked_nb_neg {
            let neg_node: NodeIdx = thread_rng().gen_range(0..self.embedded_scales.len());
            if neg_node != node_i
                && neg_node != node_j
                && self
                    .node_params
                    .get_node_param(node_i)
                    .get_edge(neg_node)
                    .is_none()
            {
                // get a read lock, as neg_node is not the locked nodes node_i and node_j
                let neg_data = self.get_embedded_data(neg_node);
                let y_k;
                if let Some(lock_read) = neg_data.try_read() {
                    y_k = lock_read.to_owned();
                    got_nb_neg += 1;
                } else {
                    // to cover contention case... not seen
                    log::trace!("neg lock failed");
                    _nb_failed += 1;
                    continue;
                }
                // compute the common part of coeff as in function grad_common_coeff
                let d_ik: f64 = y_i
                    .iter()
                    .zip(y_k.iter())
                    .map(|(vi, vj)| (*vi - *vj) * (*vi - *vj))
                    .sum::<F>()
                    .to_f64()
                    .unwrap();
                let d_ik_scaled = d_ik / (scale * scale);
                let coeff = if b != 1. {
                    let cauchy_weight = 1. / (1. + d_ik_scaled.powf(b));
                    2. * b * cauchy_weight * d_ik_scaled.powf(b - 1.) / (scale * scale)
                } else {
                    let cauchy_weight = 1. / (1. + d_ik_scaled);
                    2. * b * cauchy_weight / (scale * scale)
                };
                // we know node_j is not in neighbour, we smooth repulsion for point with dist less than scale/4
                // the part of repulsion comming from coeff is less than 1/(scale * scale)
                //
                let alfa = 1. / 16.;
                if d_ik > 0. {
                    let coeff_repulsion = 1. / (d_ik_scaled * d_ik_scaled).max(alfa); // !!
                    let coeff_ik = (grad_step * coeff * coeff_repulsion).min(2.);
                    gradient = (&y_k - &y_i) * F::from_f64(coeff_ik).unwrap();
                    log::trace!(
                        "norm repulsive  coeff gradient {:.2e} {:.2e}",
                        coeff_ik,
                        l2_norm(&gradient.view()).to_f64().unwrap()
                    );
                }
                y_i -= &gradient;
            } // end node_neg is accepted
        } // end of loop on neg sampling
          // final update of node_i
        *(self.get_embedded_data(node_i).write()) = y_i;
    } // end of ce_optim_from_point

    #[allow(unused)]
    fn gradient_iteration(&self, nb_sample: usize, grad_step: f64) {
        for _ in 0..nb_sample {
            self.ce_optim_edge_shannon(false, grad_step);
        }
    } // end of gradient_iteration

    fn gradient_iteration_threaded(&self, nb_sample: usize, grad_step: f64) {
        (0..nb_sample)
            .into_par_iter()
            .for_each(|_| self.ce_optim_edge_shannon(true, grad_step));
    } // end of gradient_iteration_threaded
} // end of impl EntropyOptim

//===============================================================================================================


/// computes the weight of an embedded edge.
/// scale correspond at density observed at initial point in original graph (hence the asymetry)
fn cauchy_edge_weight<F>(initial_point: &Array1<F>, scale: f64, b: f64, other: &Array1<F>) -> F
where
    F: Float + std::iter::Sum + num_traits::FromPrimitive,
{
    let dist = initial_point
        .iter()
        .zip(other.iter())
        .map(|(i, f)| (*i - *f) * (*i - *f))
        .sum::<F>();
    let mut dist_f64 = dist.to_f64().unwrap() / (scale * scale);
    //
    dist_f64 = dist_f64.powf(b);
    //    assert!(dist_f64 > 0.);
    //
    let weight = 1. / (1. + dist_f64);
    let mut weight_f = F::from_f64(weight).unwrap();
    if !(weight_f < F::one()) {
        weight_f = F::one() - F::epsilon();
        log::trace!("cauchy_edge_weight fixing dist_f64 {:2.e}", dist_f64);
    }
    assert!(weight_f < F::one());
    assert!(weight_f.is_normal());
    weight_f
} // end of cauchy_edge_weight

fn l2_norm<F>(v: &ArrayView1<'_, F>) -> F
where
    F: Float + std::iter::Sum + num_traits::cast::FromPrimitive,
{
    //
    v.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt()
} // end of l2_norm

// in embedded space (in unit ball or unit box) the scale is chosen as the scale at corresponding point / divided by mean initial scales.
fn estimate_embedded_scales_from_initial_scales(initial_scales: &[f32]) -> Vec<f32> {
    log::trace!("estimate_embedded_scale_from_initial_scales");
    let mean_scale: f32 = initial_scales.iter().sum::<f32>() / (initial_scales.len() as f32);
    let scale_sup = 4.0; // CAVEAT seems we can go up to 4.
    let scale_inf = 1. / scale_sup;
    let width = 0.2;
    // We want embedded scae impact between 0.5 and 2 (amplitude 4) , we take into account the square in cauchy weight
    let embedded_scale: Vec<f32> = initial_scales
        .iter()
        .map(|&x| width * (x / mean_scale).min(scale_sup).max(scale_inf))
        .collect();
    //
    for (i, scale) in embedded_scale.iter().enumerate() {
        log::trace!("embedded scale for node {} : {:.2e}", i, scale);
    }
    //
    embedded_scale
} // end of estimate_embedded_scale_from_initial_scales

// renormalize data (center and enclose in a box of a given box size) before optimization of cross entropy
fn set_data_box<F>(data: &mut Array2<F>, box_size: F)
where
    F: Float
        + NumAssign
        + std::iter::Sum<F>
        + num_traits::cast::FromPrimitive
        + ndarray::ScalarOperand,
{
    let nbdata = data.nrows();
    let dim = data.ncols();
    //
    let mut means = Array1::<F>::zeros(dim);
    let mut max_max = F::zero();
    //
    for j in 0..dim {
        for i in 0..nbdata {
            means[j] += data[[i, j]];
        }
        means[j] /= F::from(nbdata).unwrap();
    }
    for j in 0..dim {
        for i in 0..nbdata {
            data[[i, j]] -= means[j];
            max_max = max_max.max(data[[i, j]].abs());
        }
    }
    //
    max_max /= box_size / F::from(2.).unwrap();
    for f in data.iter_mut() {
        *f /= max_max;
        assert!((*f).abs() <= box_size);
    }
} // end of set_data_box

#[cfg(test)]
#[allow(clippy::range_zip_with_len)]
mod tests {

    //    cargo test embedder  -- --nocapture

    use super::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[cfg(test)]
    fn gen_rand_data_f32(nb_elem: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::<Vec<f32>>::with_capacity(nb_elem);
        let mut rng = thread_rng();
        let unif = Uniform::<f32>::new(0., 1.);
        for i in 0..nb_elem {
            let val = 2. * i as f32 * rng.sample(unif);
            let v: Vec<f32> = (0..dim).map(|_| val * rng.sample(unif)).collect();
            data.push(v);
        }
        data
    } // end of gen_rand_data_f32

    #[test]
    fn mini_embed_full() {
        log_init_test();
        // generate datz
        let nb_elem = 500;
        let embed_dim = 20;
        let data = gen_rand_data_f32(nb_elem, embed_dim);
        let data_with_id = data
            .iter()
            .zip(0..data.len())
            .collect::<Vec<(&Vec<f32>, usize)>>();
        // hnsw construction
        let ef_c = 50;
        let max_nb_connection = 50;
        let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
        let mut hns =
            Hnsw::<f32, DistL1>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL1 {});
        // to enforce the asked number of neighbour
        hns.set_keeping_pruned(true);
        hns.parallel_insert(&data_with_id);
        hns.dump_layer_info();
        // go to kgraph
        let knbn = 10;
        log::info!("calling kgraph.init_from_hnsw_all");
        let kgraph: KGraph<f32> = kgraph_from_hnsw_all(&hns, knbn).unwrap();
        log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
        let _kgraph_stats = kgraph.get_kraph_stats();
        let mut embed_params = EmbedderParams::default();
        embed_params.asked_dim = 5;
        let mut embedder = Embedder::new(&kgraph, embed_params);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
    } // end of mini_embed_full
} // end of tests
