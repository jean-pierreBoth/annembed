//! umap-like embedding from GraphK
//! The embedding is based on the graph provided by hnsw.
//! Edges out a given point are given an exponential weight scaled with the distance to nearest neighbour
//! point in the vicinity of the point. This scale can be modulated by the scale_rho parameter.
//! 



//#![allow(dead_code)]

use num_traits::{Float, NumAssign};

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::{Lapack, Scalar};


use quantiles::{ckms::CKMS};     // we could use also greenwald_khanna

// threading needs
use rayon::prelude::*;
use parking_lot::{RwLock};
use std::sync::Arc;

use rand::{Rng, thread_rng};
use rand::distributions::{Uniform};
use rand_distr::{WeightedAliasIndex};

use indexmap::set::*;


use std::time::{Duration,SystemTime};
use cpu_time::ProcessTime;

use hnsw_rs::prelude::*;
use crate::fromhnsw::*;
use crate::embedparams::*;
use crate::nodeparam::*;
use crate::diffmaps::*;
use crate::tools::dichotomy::*;

/// do not consider probabilities under PROBA_MIN, thresolded!!
const PROBA_MIN: f32 = 1.0E-5;





//=====================================================================================



/// The structure corresponding to the embedding process. 
/// It must be initialized by the graph extracted from Hnsw according to the choosen strategy
/// and the asked dimension for embedding.
pub struct Embedder<'a,F> {
    /// graph constrcuted with fromhnsw module
    kgraph: Option<&'a KGraph<F>>,
    ///
    hkgraph: Option<&'a KGraphProjection<F>>,
    /// parameters
    parameters : EmbedderParams,
    /// contains edge probabilities according to the probabilized graph constructed before laplacian symetrization
    /// It is this representation that is used for cross entropy optimization!
    initial_space: Option<NodeParams>,
    /// initial embedding (option for degugging analyzing)
    initial_embedding : Option<Array2<F>>,
    /// final embedding
    embedding: Option<Array2<F>>,
} // end of Embedder


impl<'a,F> Embedder<'a,F>
where
    F: Float + Lapack + Scalar + ndarray::ScalarOperand + Send + Sync,
{
    /// constructor from a graph and asked embedding dimension
    pub fn new(kgraph : &'a KGraph<F>, parameters : EmbedderParams) -> Self {
        Embedder::<F>{kgraph : Some(kgraph), hkgraph : None, parameters , initial_space:None, initial_embedding : None, embedding:None}
    } // end of new


    /// construction from a hierarchical graph
    pub fn from_hkgraph(graph_projection : &'a KGraphProjection<F>, parameters : EmbedderParams) -> Self {
        Embedder::<F>{kgraph : None, hkgraph : Some(graph_projection), parameters , initial_space:None, initial_embedding : None, embedding:None}


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


    /// do the embedding
    pub fn embed(&mut self) -> Result<usize, usize> {
        //
        self.parameters.log();
        let graph_to_embed = self.kgraph.unwrap();
        // construction of initial neighbourhood, scales and proba of edges from distances.
        // we will need  initial_space representation for graph laplacian and in cross entropy optimization
        self.initial_space = Some(to_proba_edges(graph_to_embed, self.parameters.scale_rho as f32, self.parameters.beta as f32));
        // we can initialize embedding with diffusion maps or pure random.
        let mut initial_embedding;
        if self.parameters.dmap_init {
            // initial embedding via diffusion maps, in this case we have to have a coherent box normalization with random case
            let cpu_start = ProcessTime::now();
            let sys_start = SystemTime::now();
            initial_embedding = get_dmap_embedding(self.initial_space.as_ref().unwrap(), self.parameters.get_dimension());
            println!(" dmap initialization sys time(ms) {:.2e} cpu time(ms) {:.2e}", sys_start.elapsed().unwrap().as_millis(), cpu_start.elapsed().as_millis());
            set_data_box(&mut initial_embedding, 1.);
        }
        else {
        // if we use random initialization we must have a box size coherent with renormalizes scales, so box size is 1.
            initial_embedding = self.get_random_init(1.);
        }
        let embedding_res = self.entropy_optimize(&self.parameters, &initial_embedding);
        // optional store dump initial embedding
        self.initial_embedding = Some(initial_embedding);
        //
        match embedding_res {
            Ok(embedding) => {
                self.embedding = Some(embedding);
                return Ok(1);
            }
            _ => {
                log::info!("Embedder::embed : embedding optimization failed");
                return Err(1);
            }        
        }
    } // end embed


    /// At the end returns the embedded data as Matrix. 
    /// The row of the matrix corresponds to the embedded dat vectors but after reindexation of DataId
    /// to ensure a contiguous indexation.  
    /// To get a matrix with row corresponding to DataId if they were already contiguous for 0 to nbdata use
    /// function  get_embedded_reindexed to get the permutation/reindexation unrolled!
    pub fn get_embedded(&self) -> Option<&Array2<F>> {
        return self.embedding.as_ref();
    }

    /// returns embedded data reindexed by DataId. This requires the DataId to be contiguous from 0 to nbdata.  
    ///  See [crate::fromhnsw::KGraph::get_idx_from_dataid]
    pub fn get_embedded_reindexed(&self) -> Array2<F> {
        let emmbedded = self.embedding.as_ref().unwrap();
        let (nbrow, dim) = emmbedded.dim();
        let mut reindexed =  Array2::<F>::zeros((nbrow, dim));
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (One week bug!)
        for i in 0..nbrow {
            let row = emmbedded.row(i);
            let origin_id = self.kgraph.unwrap().get_data_id_from_idx(i).unwrap();
            for j in 0..dim {
                reindexed[[*origin_id,j]] = row[j];
            }
        }
        return reindexed;
    }    

    /// **return the embedded vector corresponding to original data vector corresponding to data_id**
    /// This methods fails if data_id do not exist. Use KGraph.get_data_id_from_idx to check before if necessary.
    pub fn get_embedded_by_dataid(&self, data_id : &DataId) -> ArrayView1<F> {
        // we must get data index as stored in IndexSet
        let kgraph = self.kgraph.unwrap();    // CAVEAT depends on processing state
        let data_idx = kgraph.get_idx_from_dataid(data_id).unwrap();
        self.embedding.as_ref().unwrap().row(data_idx)
    } // end of get_data_embedding


    /// **get embedding of a given node index after reindexation by the embedding to index in [0..nb_nodes]**
    pub fn get_embedded_by_nodeid(&self, node : NodeIdx) -> ArrayView1<F> {
        self.embedding.as_ref().unwrap().row(node)
    }

    
     /// returns the initial embedding. Same remark as for method get_embedded. Storage is optional TODO
     pub fn get_initial_embedding(&self) -> Option<&Array2<F>> {
        return self.initial_embedding.as_ref();
    }   

    pub fn get_initial_embedding_reindexed(&self) ->  Array2<F> {
        let emmbedded = self.initial_embedding.as_ref().unwrap();
        let (nbrow, dim) = emmbedded.dim();
        let mut reindexed =  Array2::<F>::zeros((nbrow, dim));
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (One week bug!)
        for i in 0..nbrow {
            let row = emmbedded.row(i);
            let origin_id = self.kgraph.unwrap().get_data_id_from_idx(i).unwrap();
            for j in 0..dim {
                reindexed[[*origin_id,j]] = row[j];
            }
        }
        return reindexed;
    }  // end of get_initial_embedding_reindexed



    /// get random initialization in a square of side size
    fn get_random_init(&mut self, size:f32) -> Array2<F> {
        log::trace!("Embedder get_random_init with size {:.2e}", size);
        //
        let nb_nodes = self.initial_space.as_ref().unwrap().get_nb_nodes();
        let mut initial_embedding = Array2::<F>::zeros((nb_nodes, self.get_asked_dimension()));
        let unif = Uniform::<f32>::new(-size/2. , size/2.);
        let mut rng = thread_rng();
        for i in 0..nb_nodes {
            for j in 0..self.get_asked_dimension() {
                initial_embedding[[i,j]] = F::from(rng.sample(unif)).unwrap();
            }
        }
        //
        initial_embedding
    }  // end of get_random_init




    // given neighbours of a node we choose scale to satisfy a normalization constraint.
    // p_i = exp[- beta * (d(x,y_i) - d(x, y_1)/ local_scale ]
    // We return beta/local_scale
    // as function is monotonic with respect to scale, we use dichotomy.
    #[allow(unused)]
    fn get_scale_from_umap(&self, norm: f64, neighbours: &Vec<OutEdge<F>>) -> (f32, Vec<f32>) {
        // p_i = exp[- beta * (d(x,y_i)/ local_scale) ]
        let nbgh = neighbours.len();
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut dist = neighbours
            .iter()
            .map(|n| n.weight.to_f32().unwrap())
            .collect::<Vec<f32>>();
        let f = |beta: f32| {
            dist.iter()
                .map(|d| (-(d - rho_x) * beta).exp())
                .sum::<f32>()
        };
        // f is decreasing
        // TODO we could also normalize as usual?
        let beta = dichotomy_solver(false, f, 0f32, f32::MAX, norm as f32).unwrap();
        // reuse rho_y_s to return proba of edge
        for i in 0..nbgh {
            dist[i] = (-(dist[i] - rho_x) * beta).exp();
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
    fn entropy_optimize(&self, params : &EmbedderParams, initial_embedding : &Array2<F>) -> Result<Array2<F>, String> {
        //
        log::info!("in Embedder::entropy_optimize");
        //
        if self.initial_space.is_none() {
            log::error!("Embedder::entropy_optimize : initial_space not constructed, exiting");
            return Err(String::from(" initial_space not constructed, no NodeParams"));
        }
        let ce_optimization = EntropyOptim::new(self.initial_space.as_ref().unwrap(), params, initial_embedding);
        // compute initial value of objective function
        let start = ProcessTime::now();
        let initial_ce = ce_optimization.ce_compute();
        let cpu_time: Duration = start.elapsed();
        println!(" initial cross entropy value {:.2e},  in time {:?}", initial_ce, cpu_time);
        // We manage some iterations on gradient computing
        let grad_step_init = params.grad_step;
        log::info!("grad_step_init : {:.2e}", grad_step_init);
        //
        log::debug!("in Embedder::entropy_optimize  ... gradient iterations");
        let nb_sample_by_iter = params.nb_sampling_by_edge * ce_optimization.get_nb_edges();
        //
        log::info!("\n optimizing embedding");
        log::info!(" nb edges {} , number of edge sampling by grad iteration {}", ce_optimization.get_nb_edges(), nb_sample_by_iter);
        log::info!(" nb iteration : {}  sampling size {} ", self.get_nb_grad_batch(), nb_sample_by_iter);
        let start = ProcessTime::now();
        for iter in 1..=self.get_nb_grad_batch() {
            // loop on edges
            let grad_step = grad_step_init * (1.- iter as f64/self.get_nb_grad_batch() as f64);
            let start = ProcessTime::now();
            ce_optimization.gradient_iteration_threaded(nb_sample_by_iter, grad_step);
            let cpu_time: Duration = start.elapsed();
//            println!("ce after grad iteration time {:?} grad iter {:.2e}",  cpu_time, ce_optimization.ce_compute_threaded());
            log::debug!("ce after grad iteration time {:?} grad iter {:.2e}",  cpu_time, ce_optimization.ce_compute_threaded());
        }
        let cpu_time: Duration = start.elapsed();
        println!(" gradient iterations cpu_time {:?}",  cpu_time);
        let final_ce = ce_optimization.ce_compute();
        println!(" final cross entropy value {:.2e}", final_ce);
        // return reindexed data (if possible)
        let dim = self.get_asked_dimension();
        let nbrow = self.get_nb_nodes();
        let mut reindexed =  Array2::<F>::zeros((nbrow, dim));
        // TODO version 0.15 provides move_into and push_row
        // Here we must not forget that to interpret results we must go
        // back from indexset to original points (One week bug!)
        for i in 0..nbrow {
            let row = ce_optimization.get_embedded_data(i);
            for j in 0..dim {
                reindexed[[i,j]] = row.read()[j];
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
    edges : Vec<(NodeIdx, OutEdge<f32>)>,
    /// scale for each node
    initial_scales : Vec<f32>,
    /// embedded coordinates of each node, under RwLock to // optimization     nbnodes * (embedded dim * f32 + lock size))
    embedded : Vec<Arc<RwLock<Array1<F>>>>,
    /// embedded_scales
    embedded_scales : Vec<f32>,
    /// weighted array for sampling positive edges
    pos_edge_distribution : WeightedAliasIndex<f32>,
    /// embedding parameters
    params : &'a EmbedderParams,
} // end of EntropyOptim




impl <'a, F> EntropyOptim<'a,F> 
    where F: Float + NumAssign + std::iter::Sum + num_traits::cast::FromPrimitive + Send + Sync + ndarray::ScalarOperand {
    //
    pub fn new(node_params : &'a NodeParams, params: &'a EmbedderParams, initial_embed : &Array2<F>) -> Self {
        log::info!("entering EntropyOptim::new");
        // TODO what if not the same number of neighbours!!
        let nbng = node_params.params[0].edges.len();
        let nbnodes = node_params.get_nb_nodes();
        let mut edges = Vec::<(NodeIdx, OutEdge<f32>)>::with_capacity(nbnodes*nbng);
        let mut edges_weight = Vec::<f32>::with_capacity(nbnodes*nbng);
        let mut initial_scales =  Vec::<f32>::with_capacity(nbnodes);
        // construct field edges
        for i in 0..nbnodes {
            initial_scales.push(node_params.params[i].scale);
            for j in 0..node_params.params[i].edges.len() {
                edges.push((i, node_params.params[i].edges[j]));
                edges_weight.push(node_params.params[i].edges[j].weight);
            }
        }
        log::info!("construction alias table for sampling edges..");
        let start = ProcessTime::now();
        let pos_edge_sampler = WeightedAliasIndex::new(edges_weight).unwrap();
        let cpu_time: Duration = start.elapsed();
        log::info!("constructied alias table for sampling edges.. , time : {:?}", cpu_time);
        // construct embedded, initial embed can be droped now
        let mut embedded = Vec::<Arc<RwLock<Array1<F>>>>::new();
        let nbrow  = initial_embed.nrows();
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
        println!("");  
        //
        EntropyOptim { node_params,  edges, initial_scales, embedded, embedded_scales, 
                            pos_edge_distribution : pos_edge_sampler,
                            params : params}
        // construct field embedded
    }  // end of new 

    pub fn get_nb_edges(&self) -> usize {
        self.edges.len()
    } // end of get_nb_edges

    // return result as an Array2<F> cloning data to result to struct Embedder
    // We return data in rows as (re)indexed in graph construction after hnsw!!
    #[allow(unused)]
    fn get_embedded_raw(& self) -> Array2<F> {
        let nbrow = self.embedded.len();
        let nbcol = self.params.asked_dim;
        let mut embedding_res = Array2::<F>::zeros((nbrow, nbcol));
        // TODO version 0.15 provides move_into and push_row
        // 
        for i in 0..nbrow {
            let row = self.embedded[i].read();
            for j in 0..nbcol {
                embedding_res[[i,j]] = row[j];
            }
        }
        return embedding_res;
    }


    // return result as an Array2<F> cloning data to return result to struct Embedder
    // Here we reindex data according to their original order. DataId must be contiguous and fill [0..nb_data]
    // for this method not to panic!
#[allow(unused)]
    pub fn get_embedded_reindexed(& self, indexset: &IndexSet<NodeIdx>) -> Array2<F> {
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
                embedding_res[[*origin_id,j]] = row[j];
            }
        }
        return embedding_res;
    }


    /// row is here a NodeIdx (after reindexation of DataId), hence the not public interface
    fn get_embedded_data(&self, row : usize) -> Arc<RwLock<Array1<F>>> {
        Arc::clone(&self.embedded[row])
    }

    // computes croos entropy between initial space and embedded distribution. 
    // necessary to monitor optimization
    fn ce_compute(&self) -> f64 {
        log::trace!("\n entering EntropyOptim::ce_compute");
        //
        let mut ce_entropy = 0.;
        let b : f64 = self.params.b;

        for edge in self.edges.iter() {
            let node_i = edge.0;
            let node_j = edge.1.node;
            assert!(node_i != node_j);
            let weight_ij = edge.1.weight as f64;
            let weight_ij_embed = cauchy_edge_weight(&self.embedded[node_i].read(), 
                    self.embedded_scales[node_i] as f64, b,
                    &self.embedded[node_j].read()).to_f64().unwrap();
            if weight_ij_embed > 0. {
                ce_entropy += -weight_ij * weight_ij_embed.ln();
            }
            if weight_ij_embed < 1. {
                ce_entropy += - (1. - weight_ij) * (1. - weight_ij_embed).ln();
            }            
            if !ce_entropy.is_finite() {
                log::debug!("weight_ij {} weight_ij_embed {}", weight_ij, weight_ij_embed);
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
        let b : f64 = self.params.b;
        let ce_entropy = self.edges.par_iter()
            .fold( || 0.0f64, | entropy : f64, edge| entropy + {
                let node_i = edge.0;
                let node_j = edge.1.node;
                let weight_ij = edge.1.weight as f64;
                let weight_ij_embed = cauchy_edge_weight(&self.embedded[node_i].read(), 
                        self.initial_scales[node_i] as f64, b,
                        &self.embedded[node_j].read()).to_f64().unwrap();
                let mut term = 0.;
                if weight_ij_embed > 0. {
                    term += -weight_ij * weight_ij_embed.ln();
                }
                if weight_ij_embed < 1. {
                    term += - (1. - weight_ij) * (1. - weight_ij_embed).ln();
                }
                term
            })
            .sum::<f64>();
        return ce_entropy;
    }





    // TODO : pass functions corresponding to edge_weight and grad_edge_weight as arguments to test others weight function
    /// This function optimize cross entropy for Shannon cross entropy
    fn ce_optim_edge_shannon(&self, threaded : bool, grad_step : f64)
    where
        F: Float + NumAssign + std::iter::Sum + num_traits::cast::FromPrimitive + ndarray::ScalarOperand
    {
        // 
        let edge_idx_sampled : usize;
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
        } // end threaded
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
        let b : f64 = self.params.b;
        // compute l2 norm of y_j - y_i
        let d_ij : f64 = y_i.iter().zip(y_j.iter()).map(|(vi,vj)| (*vi-*vj)*(*vi-*vj)).sum::<F>().to_f64().unwrap();
        let d_ij_scaled = d_ij/(scale*scale);
        // this coeff is common for P and 1.-P part
        let coeff : f64;
        if b != 1. { 
            let cauchy_weight = 1./ (1. + d_ij_scaled.powf(b));
            coeff =  2. * b * cauchy_weight * d_ij_scaled.powf(b - 1.)/ (scale*scale);
        }
        else {
            let cauchy_weight = 1./ (1. + d_ij_scaled);
            coeff =  2. * b * cauchy_weight / (scale*scale);
        }
        if d_ij_scaled > 0. {
            // repulsion annhinilate  attraction if P<= 1. / (alfa + 1)
            let alfa = 100.;
            let coeff_repulsion = 1. / (d_ij_scaled*d_ij_scaled).max(alfa);
            // clipping makes each point i or j making at most half way to the other in case of attraction
            let coeff_ij = (grad_step * coeff * (- weight + (1.-weight) * coeff_repulsion)).max(-0.49);
            gradient = (&y_j - &y_i) * F::from(coeff_ij).unwrap();
            log::trace!("norm attracting coeff {:.2e} gradient {:.2e}", coeff_ij, l2_norm(&gradient.view()).to_f64().unwrap());
        }
        y_i -= &gradient;
        y_j += &gradient;
        *(self.get_embedded_data(node_j).write()) = y_j;
        // now we loop on negative sampling filtering out nodes that are either node_i or are in node_i neighbours.
        let asked_nb_neg = 5;
        let mut got_nb_neg = 0;
        let mut _nb_failed = 0;
        while got_nb_neg < asked_nb_neg {
            let neg_node : NodeIdx = thread_rng().gen_range(0..self.embedded_scales.len());
            if neg_node != node_i && neg_node != node_j && self.node_params.get_node_param(node_i).get_edge(neg_node).is_none() {
                // get a read lock, as neg_node is not the locked nodes node_i and node_j
                let neg_data = self.get_embedded_data(neg_node);
                let y_k;
                if let Some(lock_read) = neg_data.try_read() {
                    y_k = lock_read.to_owned();
                    got_nb_neg += 1;
                }
                else {  // to cover contention case... not seen
                    log::trace!("neg lock failed");
                    _nb_failed += 1;
                    continue;
                }
                // compute the common part of coeff as in function grad_common_coeff
                let d_ik : f64 = y_i.iter().zip(y_k.iter()).map(|(vi,vj)| (*vi-*vj)*(*vi-*vj)).sum::<F>().to_f64().unwrap();
                let d_ik_scaled = d_ik/(scale*scale);
                let coeff : f64;
                if b != 1. {
                    let cauchy_weight = 1./ (1. + d_ik_scaled.powf(b));
                    coeff = 2. * b * cauchy_weight * d_ik_scaled.powf(b - 1.)/ (scale*scale);
                }
                else {
                    let cauchy_weight = 1./ (1. + d_ik_scaled);
                    coeff = 2. * b * cauchy_weight/ (scale*scale);
                }
                // use the same clipping as attractive/repulsion case
                let alfa = 1./100.;
                if d_ik > 0. {
                    let coeff_repulsion = 1. /(d_ik_scaled * d_ik_scaled).max(alfa);  // !!
                    let coeff_ik =  (grad_step * coeff * coeff_repulsion).min(2.);
                    gradient = (&y_k - &y_i) * F::from_f64(coeff_ik).unwrap();
                    log::trace!("norm repulsive  coeff gradient {:.2e} {:.2e}", coeff_ik , l2_norm(&gradient.view()).to_f64().unwrap());
                }
                y_i -= &gradient;
            } // end node_neg is accepted
        }  // end of loop on neg sampling
        // final update of node_i
        *(self.get_embedded_data(node_i).write()) = y_i;
    } // end of ce_optim_from_point



#[allow(unused)]
    fn gradient_iteration(&self, nb_sample : usize, grad_step : f64) {
        for _ in 0..nb_sample {
            self.ce_optim_edge_shannon(false, grad_step);
        }
    } // end of gradient_iteration



    fn gradient_iteration_threaded(&self, nb_sample : usize, grad_step : f64) {
        (0..nb_sample).into_par_iter().for_each( |_| self.ce_optim_edge_shannon(true, grad_step));
    } // end of gradient_iteration_threaded
    
    
}  // end of impl EntropyOptim


//===============================================================================================================

use std::fmt::{Display,Debug,LowerExp,UpperExp}; 

// Construct the representation of graph as a collections of probability-weighted edges
// determines scales in initial space and proba of edges for the neighbourhood of every point.
// Construct node params for later optimization
// after this function Embedder structure do not need field kgraph anymore
// This function relies on get_scale_from_proba_normalisation function which construct proabability-weighted edge around each node.
// These 2 function are also the base of module dmap
//
fn to_proba_edges<F>(kgraph : & KGraph<F>, scale_rho : f32, beta : f32) -> NodeParams
    where F : Float + num_traits::cast::FromPrimitive + Display + Debug + LowerExp + UpperExp + Send + Sync {
    //
    let nbnodes = kgraph.get_nb_nodes();
    let mut perplexity_q : CKMS<f32> = CKMS::<f32>::new(0.001);
    let mut scale_q : CKMS<f32> = CKMS::<f32>::new(0.001);
    let mut weight_q :  CKMS<f32> = CKMS::<f32>::new(0.001);
    // get stats
    let mut node_params = Vec::<NodeParam>::with_capacity(nbnodes);
    // TODO can be // with rayon taking care of indexation
    let neighbour_hood = kgraph.get_neighbours();
    let mut max_nbng = 0;
    for i in 0..neighbour_hood.len() {
        let node_param = get_scale_from_proba_normalisation(kgraph, scale_rho, beta, &neighbour_hood[i]);
        scale_q.insert(node_param.scale);
        perplexity_q.insert(node_param.get_perplexity());
        // coose random edge to audit
        let j = thread_rng().gen_range(0..node_param.edges.len());
        weight_q.insert(node_param.edges[j].weight);
        max_nbng = node_param.edges.len().max(max_nbng);
        assert_eq!(node_param.edges.len(), neighbour_hood[i].len());
        node_params.push(node_param);
    }
    // dump info on quantiles
    println!("\n constructed initial space");
    println!("\n scales quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
    scale_q.query(0.05).unwrap().1, scale_q.query(0.5).unwrap().1, 
    scale_q.query(0.95).unwrap().1, scale_q.query(0.99).unwrap().1);
    //
    println!("\n edge weight quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
    weight_q.query(0.05).unwrap().1, weight_q.query(0.5).unwrap().1, 
    weight_q.query(0.95).unwrap().1, weight_q.query(0.99).unwrap().1);
    //
    println!("\n perplexity quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
    perplexity_q.query(0.05).unwrap().1, perplexity_q.query(0.5).unwrap().1, 
    perplexity_q.query(0.95).unwrap().1, perplexity_q.query(0.99).unwrap().1);
    println!("");    
    //
    NodeParams::new(node_params, max_nbng)
}  // end of construction of node params



// Simplest function where we know really what we do and why. 
// Given a graph, scale and exponent parameters transform a list of distance-edge to neighbours into a list of proba-edge.
// 
// Given neighbours of a node we choose scale to satisfy a normalization constraint.
// p_i = exp[- (d(x,y_i) - shift)/ local_scale)**beta] 
// with shift = d(x,y_0) and local_scale = mean_dist_to_first_neighbour * scale_rho
//  and then normalized to 1.
//
// We do not set an edge from x to itself. So we will have 0 on the diagonal matrix of transition probability.
// This is in accordance with t-sne, umap and so on. The main weight is put on first neighbour.
//
// This function returns the local scale (i.e mean distance of a point to its nearest neighbour)
// and vector of proba weight to nearest neighbours.
//
fn get_scale_from_proba_normalisation<F> (kgraph : & KGraph<F>, scale_rho : f32, beta : f32, neighbours: &Vec<OutEdge<F>>) -> NodeParam 
    where F : Float + num_traits::cast::FromPrimitive {
    //
//        log::trace!("in get_scale_from_proba_normalisation");
    let nbgh = neighbours.len();
    // determnine mean distance to nearest neighbour at local scale, reason why we need kgraph as argument.
    let rho_x = neighbours[0].weight.to_f32().unwrap();
    let mut rho_y_s = Vec::<f32>::with_capacity(neighbours.len() + 1);
    for i in 0..nbgh {
        let y_i = neighbours[i].node; // y_i is a NodeIx = usize
        rho_y_s.push(kgraph.neighbours[y_i][0].weight.to_f32().unwrap());
    } // end of for i
    rho_y_s.push(rho_x);
    let mean_rho = rho_y_s.iter().sum::<f32>() / (rho_y_s.len() as f32);
    // we set scale so that transition proba do not vary more than PROBA_MIN between first and last neighbour
    // exp(- (first_dist -last_dist)/scale) >= PROBA_MIN
    // TODO do we need some optimization with respect to this 1 ? as we have lambda for high variations
    let scale = scale_rho * mean_rho;
    // now we adjust scale so that the ratio of proba of last neighbour to first neighbour do not exceed epsil. CAVEAT
    let first_dist = neighbours[0].weight.to_f32().unwrap();
    let last_dist = neighbours[nbgh - 1].weight.to_f32().unwrap();
    assert!(first_dist > 0. && last_dist > 0.);
    assert!(last_dist >= first_dist);
    //
    let remap_weight = | w : F , shift : f32, scale : f32 , beta : f32| (-((w.to_f32().unwrap() - shift).max(0.)/ scale).pow(beta)).exp();
    //
    if last_dist > first_dist {
        //
        if remap_weight(F::from(last_dist).unwrap(), first_dist, scale, beta )/remap_weight(F::from(first_dist).unwrap(), first_dist, scale, beta) < PROBA_MIN.ln() {
            log::info!("too large variation of neighbours probablities , reduce beta");
            // we could rescale by augmenting scale... or impose an edge weight of PROBA_MIN...
        }
        let mut probas_edge = neighbours
            .iter()
            .map(|n| OutEdge::<f32>::new(n.node, remap_weight(n.weight, first_dist, scale, beta)) )
            .collect::<Vec<OutEdge<f32>>>();
        //
        log::trace!(" first dist {:2e} last dist {:2e}", first_dist, last_dist);
        log::trace!("scale : {:.2e} , first neighbour proba {:2e}, last neighbour proba {:2e} proba gap {:.2e}", scale, probas_edge[0].weight, 
                        probas_edge[probas_edge.len() - 1].weight,
                        probas_edge[probas_edge.len() - 1].weight / probas_edge[0].weight);
        assert!(probas_edge[probas_edge.len() - 1].weight / probas_edge[0].weight >= PROBA_MIN);
        let sum = probas_edge.iter().map(|e| e.weight).sum::<f32>();
        for i in 0..nbgh {
            probas_edge[i].weight = probas_edge[i].weight / sum;
        }
        return NodeParam::new(scale, probas_edge);
    } else {
        // all neighbours are at the same distance!
        let probas_edge = neighbours
            .iter()
            .map(|n| OutEdge::<f32>::new(n.node, 1.0 / nbgh as f32))
            .collect::<Vec<OutEdge<f32>>>();
        return NodeParam::new(scale, probas_edge);
    }
} // end of get_scale_from_proba_normalisation
    




// restrain value
#[allow(unused)]
fn clip<F>(f : F, max : f64) -> F 
    where     F: Float + num_traits::FromPrimitive  {
    let f_r = f.to_f64().unwrap();
    let truncated = if f_r > max {
        log::trace!("truncated >");
        max
    }
    else if f_r < -max {
        log::trace!("truncated <");
        -max
    }
    else {
        f_r
    };
    return F::from(truncated).unwrap();
}   // end clip



/// computes the weight of an embedded edge.
/// scale correspond at density observed at initial point in original graph (hence the asymetry)
fn cauchy_edge_weight<F>(initial_point: &Array1<F>, scale: f64, b : f64, other: &Array1<F>) -> F
where
    F: Float + std::iter::Sum + num_traits::FromPrimitive
{
    let dist = initial_point
        .iter()
        .zip(other.iter())
        .map(|(i, f)| (*i - *f) * (*i - *f))
        .sum::<F>();
    let mut dist_f64 = dist.to_f64().unwrap() / (scale*scale);
    //
    dist_f64 = dist_f64.powf(b);
    assert!(dist_f64 > 0.);
    //
    let weight =  1. / (1. + dist_f64);
    let mut weight_f = F::from_f64(weight).unwrap();
    if !(weight_f < F::one()) {
        weight_f = F::one() - F::epsilon();
        log::trace!("cauchy_edge_weight fixing dist_f64 {:2.e}", dist_f64);
    }
    assert!(weight_f < F::one());
    assert!(weight_f.is_normal());      
    return weight_f;
} // end of cauchy_edge_weight



#[allow(unused)]
fn l2_dist<F>(y1: &ArrayView1<'_, F> , y2 : &ArrayView1<'_, F>) -> F 
where F :  Float + std::iter::Sum + num_traits::cast::FromPrimitive {
    //
    y1.iter().zip(y2.iter()).map(|(v1,v2)| (*v1 - *v2) * (*v1- *v2)).sum()
}  // end of l2_dist




fn l2_norm<F>(v: &ArrayView1<'_, F>) -> F 
where F :  Float + std::iter::Sum + num_traits::cast::FromPrimitive {
    //
    v.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt()
}  // end of l2_norm






// in embedded space (in unit ball or unit box) the scale is chosen as the scale at corresponding point / divided by mean initial scales.
fn estimate_embedded_scales_from_initial_scales(initial_scales :&Vec<f32>) -> Vec<f32> {
    log::trace!("estimate_embedded_scale_from_initial_scales");
    let mean_scale : f32 = initial_scales.iter().sum::<f32>() / (initial_scales.len() as f32);
    let scale_sup = 2.0;  // CAVEAT seems we can go up to 4.
    let scale_inf = 1./scale_sup;
    // We want embedded scae impact between 0.5 and 2 (amplitude 4) , we take into account the square in cauchy weight
    let embedded_scale : Vec<f32> = initial_scales.iter().map(|&x| 0.2 * (x/mean_scale).min(scale_sup).max(scale_inf)).collect();
    //
    for i in 0..embedded_scale.len() {
        log::trace!("embedded scale for node {} : {:.2e}", i , embedded_scale[i]);
    }
    //
    embedded_scale
}  // end of estimate_embedded_scale_from_initial_scales


// renormalize data (center and enclose in a box of a given box size) before optimization of cross entropy
fn set_data_box<F>(data : &mut Array2<F>, box_size : f64) 
    where  F: Float +  NumAssign + std::iter::Sum<F> + num_traits::cast::FromPrimitive + Send + Sync + ndarray::ScalarOperand  {
    let nbdata = data.nrows();
    let dim = data.ncols();
    //
    let mut means = Array1::<F>::zeros(dim);
    let mut max_max = F::zero();
    //
    for j in 0..dim  {
        for i in 0..nbdata {
            means[j] += data[[i,j]];
        }
        means[j] /= F::from(nbdata).unwrap();
    }
    for j in 0..dim  {
        for i in 0..nbdata {
            data[[i,j]] = data[[i,j]] - means[j];
            max_max = max_max.max(data[[i,j]].abs());          
        }
    }
    //
    max_max /= F::from(0.5 * box_size).unwrap();
    for f in data.iter_mut()  {
        *f = (*f)/max_max;
        assert!((*f).abs() <= F::one());
    }    
}  // end of set_data_box



mod tests {

//    cargo test embedder  -- --nocapture


    #[allow(unused)]
    use super::*;


    use rand::distributions::{Uniform};


    // have a warning with and error without ?
    #[allow(unused)]
    use hnsw_rs::prelude::*;
    #[allow(unused)]
    use hnsw_rs::hnsw::Neighbour;

    #[allow(unused)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }




    #[allow(unused)]
    fn gen_rand_data_f32(nb_elem: usize , dim:usize) -> Vec<Vec<f32>> {
        let mut data = Vec::<Vec<f32>>::with_capacity(nb_elem);
        let mut rng = thread_rng();
        let unif =  Uniform::<f32>::new(0.,1.);
        for i in 0..nb_elem {
            let val = 2. * i as f32 * rng.sample(unif);
            let v :Vec<f32> = (0..dim).into_iter().map(|_|  val * rng.sample(unif)).collect();
            data.push(v);
        }
        data
    }
    
    #[test]
    fn mini_embed_full() {
        log_init_test();
        // generate datz
        let nb_elem = 500;
        let embed_dim = 20;
        let data = gen_rand_data_f32(nb_elem, embed_dim);
        let data_with_id = data.iter().zip(0..data.len()).collect();
        // hnsw construction
        let ef_c = 50;
        let max_nb_connection = 50;
        let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
        let mut hns = Hnsw::<f32, DistL1>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL1{});
        // to enforce the asked number of neighbour
        hns.set_keeping_pruned(true);
        hns.parallel_insert(&data_with_id);
        hns.dump_layer_info();
        // go to kgraph
        let knbn = 10;
        log::info!("calling kgraph.init_from_hnsw_all");
        let kgraph = kgraph_from_hnsw_all(&hns, knbn).unwrap();
        log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
        let _kgraph_stats = kgraph.get_kraph_stats();
        let mut embed_params = EmbedderParams::new();
        embed_params.asked_dim = 5;
        let mut embedder = Embedder::new(&kgraph, embed_params);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
    } // end of mini_embed_full



} // end of tests
