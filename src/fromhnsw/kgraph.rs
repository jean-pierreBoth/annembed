//! Get a very simple graph from hnsw to be used in kruksal algo and
//! neighborhood entropy computations
//! 
//! 

use anyhow::anyhow;

use num_traits::Float;
use num_traits::cast::FromPrimitive;

// to dump to ripser
use std::io::Write;

use indexmap::set::*;

use std::cmp::Ordering;

use rand::thread_rng;

use quantiles::ckms::CKMS;     // we could use also greenwald_khanna

use rayon::prelude::*;

use hnsw_rs::prelude::*;

use crate::tools::{dimension::*,nodeparam::*};
use rand::distributions::Distribution;

// morally F should be f32 and f64.  
// The solution from ndArray is F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Display + Debug + LowerExp + UpperExp + (ScalarOperand + LinalgScalar) + Send + Sync.   
// For edge weight we just need  F : FromPrimitive + Float + Display + Debug + LowerExp + UpperExp + Send + Sync 


//====================================================================================================


/// A structure to keep track of min and max distance to neighbour.
/// We keep assume that Nan are excluded once we have reached the point we need this.
struct RangeNghb<F:Float>(F, F);


/// We may need  some statistics on the graph:
///  - range: distance to nearest and farthest nodes of each node
///  - how many edges arrives in a node (in_degree)
///  - quantiles for the distance to nearest neighbour of nodes
pub struct KGraphStat<F:Float> {
    /// for each node, distances to nearest and farthest neighbours
    ranges : Vec<RangeNghb<F>>,
    /// incoming degrees
    in_degrees : Vec<u32>,
    /// mean incoming degree
    mean_in_degree : usize,
    /// max incoming degree. Useful to choose between Compressed Storage Mat or dense Array
    max_in_degree : usize,
    ///  We maintain quantiles on distances to first neighbours ad f32
    /// This can serve as an indicator on relative density around a point.
    min_radius_q : CKMS<f32>,
}  // end of KGraphStat


impl <F:Float> KGraphStat<F> {
    /// extract a density index on point defined by inverse of max distance of k-th Neighbour
    pub fn get_density_index(&self) -> Vec<F> {
        let density = self.ranges.iter().map(|x| x.1.recip()).collect();
        //
        return density;
    } // get_density_index

    /// return maximum in_degree. Useful to choose between CsMat or dense Array2 representation of graph
    pub fn get_max_in_degree(&self) -> usize {
        self.max_in_degree
    }

    /// return mean incoming degree of nodes
    pub fn get_mean_in_degree(&self) -> usize {
        self.mean_in_degree
    }
  
    /// returns incoming degrees
    pub fn get_in_degrees(&self) -> &Vec<u32> {
        &self.in_degrees
    }

    /// return radius at quantile
    pub fn get_radius_at_quantile(&self, frac:f64) -> f32 {
        if frac >=0. && frac<=1. {
            self.min_radius_q.query(frac).unwrap().1
        }
        else {
            // do we panic! ?
            0.
        }
    }
}  // end of impl block for KGraphStat



/// 
/// A very minimal graph for this crate.  
/// 
/// The graph comes from an k-nn search so we know the number of neighbours we have.
/// Edges out of a node are given a weitht of type F which must satisfy Ord, so the 
/// edges can be sorted.
/// 
/// The first initialization from hnsw is a full hnsw representation,
/// but it should be possible to select a layer to get a subsampling of data
/// or the whole children of a given node at any layer to get a specific region of the data.  
///  
/// Note: The point extracted from the Hnsw are given an index by the KGraph structure
/// as hnsw do not enforce client data_id to be in [0..nbpoints]
/// 
#[derive(Clone)]
pub struct KGraph<F> {
    /// max number of neighbours of each node. Note it can a little less than computed in Hnsw
    pub(crate) max_nbng : usize,
    /// number of nodes.
    /// If GraphK is initialized from the descendant of a point in Hnsw we do not know in advance the number of nodes!!
    pub(crate) nbnodes: usize,
    /// neighbours\[i\] contains the indexes of neighbours node i sorted by increasing weight edge!
    /// all node indexing is done after indexation in node_set
    pub(crate) neighbours : Vec<Vec<OutEdge<F>>>,
    /// to keep track of current node indexes.
    pub(crate) node_set : IndexSet<DataId>,
}   // end of struct KGraph





impl <F> KGraph<F> 
    where F : FromPrimitive + Float + std::fmt::UpperExp + Sync + Send + std::iter::Sum
{
    /// allocates a graph with expected size nbnodes and nbng neighbours 
    pub fn new() -> Self {
        let neighbours_init = Vec::<Vec<OutEdge<F>>>::new();
        KGraph {
            max_nbng : 0,
            nbnodes : 0,
            neighbours :  neighbours_init,
            node_set : IndexSet::new(),
        }
    }  // end of new

    /// get number of nodes of graph
    pub fn get_nb_nodes(&self) -> usize {
        self.nbnodes
    }

    /// get number of neighbour of each node
    pub fn get_max_nbng(&self) -> usize {
        self.max_nbng
    }

    /// returns a reference to Neighbourhood info
    pub fn get_neighbours(&self) -> &Vec<Vec<OutEdge<F>>> {
        &self.neighbours
    }

    /// get out edges from node given its index
    pub fn get_out_edges_by_idx(&self, node : NodeIdx) -> &Vec<OutEdge<F>> {
        &self.neighbours[node]
    }

    /// returns largest edge vector by node
    pub fn compute_max_edge(&self) -> Vec<(usize,f64)> {
        let neighbours = &self.neighbours;
        // TODO already sorted...
        let mut max_edge_length : Vec<(usize,f64)> = (0..neighbours.len()).into_par_iter().map( |n| -> (usize,f64) {
                let mut node_edge_length : f64 = 0.;
                for edge in  &neighbours[n] {
                    node_edge_length = node_edge_length.max(edge.weight.to_f64().unwrap());
                }
                return (n, node_edge_length);
            }
            ).collect();
        // in max_edge_length we have for each node its largest edge, but due to // iter nodes are to be reset in order!
        max_edge_length.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap()); 
        max_edge_length
    } // end of compute_max_edge


    /// given a DataId returns list of edges from corresponding point or None if error occurs
    pub fn get_out_edges_by_data_id(&self,  data_id: &DataId) -> Result<&Vec<OutEdge<F>>, anyhow::Error> {
        let idx = self.get_idx_from_dataid(data_id);
        if idx.is_none() {
            return Err(anyhow!("bad data_id"));
        }
        //
        let idx = idx.unwrap();
        return Ok(self.get_out_edges_by_idx(idx));
    } // end of get_out_edges_by_data_id


    /// estimate intrinsic dimension around a point given by its data_id.
    /// We implement the method described in :  
    ///     Maximum likelyhood estimation of intrinsic dimension.
    ///     Levina E. and Bickel P.J NIPS 2004.  [Levina-Bickel](https://www.stat.berkeley.edu/~bickel/mldim.pdf)
    /// 
    pub fn intrinsic_dim_at_data_id(&self, data_id : &DataId) -> Result<f64,anyhow::Error>   {
        //
        let edges_res = self.get_out_edges_by_data_id(data_id);
        if edges_res.is_err() {
            return Err(edges_res.err().unwrap());
        }
        let edges: &Vec<OutEdge<F>> = edges_res.unwrap();
        intrinsic_dimension_from_edges::<F>(edges)
    } // end of intrinsic_dim


    /// We implement the method described in :  
    ///     Maximum likelyhood estimation of intrinsic dimension.
    ///     Levina E. and Bickel P.J NIPS 2004.  [Levina-Bickel](https://www.stat.berkeley.edu/~bickel/mldim.pdf).  
    /// 
    /// We estimate dimension by sampling sampling_size points around which we estimate intrinsic
    /// dimension and returs mean and standard deviation if we do not encounter error.
    ///   
    /// **Note : As recommended in the Paper cited, the estimation needs more than 20 neighbours around each point.**
    ///        We provide an estimation even if this condition is not fulfilled but it is less robust.
    // TODO : get an histogram of dimensions
    pub fn estimate_intrinsic_dim(&self, sampling_size: usize) ->  Result<(f64,f64),anyhow::Error> {
        // we sample points, ignoring the probability to sample twice or more the ame point.
        // TODO sampling without replacement?
        let mut dims = Vec::<f64>::with_capacity(sampling_size);
        let nb_nodes = self.get_nb_nodes();
        let mut rng = thread_rng();
        let between = rand_distr::Uniform::from(0..nb_nodes);
        // TODO to be parallelized if necessary
        for _ in 0..sampling_size {
            let node = between.sample(&mut rng);
            let edges = &self.neighbours[node];
            let dim = intrinsic_dimension_from_edges(edges);
            if dim.is_ok() {
                dims.push(dim.unwrap());
            }
        }
        if dims.len() == 0 {
            log::error!("could not sample dimension");
            return Err(anyhow!("could not sample points"));
        }
        let mean_dim : f64 = dims.iter().sum::<f64>()/dims.len() as f64;
        let mut sigma = dims.iter().fold(0., |acc, d| acc + (d-mean_dim)*(d-mean_dim));
        sigma = (sigma/dims.len() as f64).sqrt();
        log::debug!(" mean dimension : {:.3e}, sigma : {:.3e}, nb_points used: {}", mean_dim, sigma, dims.len());
        return Ok((mean_dim, sigma));
    } // end of estimate_intrinsic_dim


    /// As data can come from hnsw with arbitrary data id not on [0..nb_data] we reindex
    /// them for array computation.  
    /// At the end we must provide a way to get back to original labels of data.
    /// 
    /// When we get embedded data as an `Array2<F>`, row i of data corresponds to
    /// the original data with label get_data_id_from_idx(i)
    pub fn get_data_id_from_idx(&self, index:usize) -> Option<&DataId> {
        return self.node_set.get_index(index)
    }

    /// get the index corresponding to a given DataId
    pub fn get_idx_from_dataid(&self, data_id: &DataId) -> Option<usize> {
        return self.node_set.get_index_of(data_id)
    }

    /// useful after embedding to get back to original indexes.
#[allow(unused)]
    pub(crate) fn get_indexset(&self) -> &IndexSet<DataId> {
        &self.node_set
    } // end of get_indexset



    /// dump a Graph in a format corresponding to sprs::TriMatI to serve as input to Bauer's ripser module.
    /// The dump corresponds to Ripser working on a distance matrix given in sparse format. See Ripser Code or Julia Ripserer
    /// We need to symetrize the matrix as we dump a distance matrix
    /// Note that ripser do not complain for no symetric data but Ripserer does 
    pub(crate) fn to_ripser_sparse_dist(&self, writer : &mut dyn Write) -> Result<(), anyhow::Error> {
        log::debug!("in to_ripser_sparse_dist");
        //
        for i in 0..self.nbnodes {
            for n in &self.neighbours[i] {
                write!(writer, "{} {} {:.5E}\n", i, n.node, n.weight)?;
                write!(writer, "{} {} {:.5E}\n", n.node, i, n.weight)?;
            }
        }
        //
        log::debug!("to_ripser_sparse_dist finished");
        Ok(()) 
    }  // end of to_ripser_sparse_dist




    /// Fills in KGraphStat from KGraph
    pub fn get_kraph_stats(&self) -> KGraphStat<F> {
        let mut in_degrees : Vec<u32> = (0..self.nbnodes).into_iter().map(|_| 0).collect();
        let mut ranges = Vec::<RangeNghb<F>>::with_capacity(self.nbnodes);
        //
        let mut max_max_r = F::zero();
        let mut min_min_r = F::max_value();
        //
        let mut quant = CKMS::<f32>::new(0.001);
        //
        for i in 0..self.neighbours.len() {
            if self.neighbours[i].len() > 0 {
                let min_r = self.neighbours[i][0].weight;
                let max_r = self.neighbours[i][self.neighbours[i].len()-1].weight;
                quant.insert(F::to_f32(&min_r).unwrap());
                //
                max_max_r = max_max_r.max(max_r);
                min_min_r = min_min_r.min(min_r);
                // compute in_degrees
                ranges.push(RangeNghb(min_r, max_r));
                for j in 0..self.neighbours[i].len() {
                    in_degrees[self.neighbours[i][j].node] += 1;
                }
           }
        }
        // dump some info
        let mut max_in_degree = 0;
        let mut mean_in_degree : f32 = 0.;
        for i in 0..in_degrees.len() {
            max_in_degree = max_in_degree.max(in_degrees[i]);
            mean_in_degree += in_degrees[i] as f32;
        }
        if in_degrees.len() > 0 {
            mean_in_degree /= in_degrees.len() as f32;
        }
        //
        println!("\n minimal graph statistics \n");
        println!("\t max in degree : {:.2e}", max_in_degree);
        println!("\t mean in degree : {:.2e}", mean_in_degree);
        println!("\t max max range : {:.2e} ", max_max_r.to_f32().unwrap());
        println!("\t min min range : {:.2e} ", min_min_r.to_f32().unwrap());
        if quant.count() > 0 {
            println!("min radius quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
                        quant.query(0.05).unwrap().1, quant.query(0.5).unwrap().1, 
                        quant.query(0.95).unwrap().1, quant.query(0.99).unwrap().1);
        }
        //
        KGraphStat{ranges, in_degrees, mean_in_degree : mean_in_degree.round() as usize, max_in_degree : max_in_degree as usize, 
                    min_radius_q : quant}
    }  // end of get_kraph_stats

} // end of block impl KGraph


/// initialization of a graph with expected number of neighbours nbng.  
/// 
/// This initialization corresponds to the case where use all points of the hnsw structure
/// see also *initialize_from_layer* and *initialize_from_descendants*.   
/// nbng is the maximal number of neighbours kept. The effective mean number can be less,
/// in this case use the Hnsw.set_keeping_pruned(true) to restrict pruning in the search.
///
pub fn kgraph_from_hnsw_all<T, D, F>(hnsw : &Hnsw<T,D>, nbng : usize) -> std::result::Result<KGraph<F>, usize> 
    where   T : Clone + Send + Sync, 
            D : Distance<T> + Send + Sync,
            F : Float + FromPrimitive {
    //
    log::debug!("entering kgraph_from_hnsw_all");
    //
    let max_nbng = nbng;
    let mut nb_point_below_nbng = 0;
    let mut mean_deficient_neighbour_size: usize = 0;   
    let mut minimum_nbng = nbng;
    let mut mean_nbng = 0u64;
    // We must extract the whole structure , for each point the list of its nearest neighbours and weight<F> of corresponding edge
    let max_nb_conn = hnsw.get_max_nb_connection() as usize;    // morally this the k of knn bu we have that for each layer
    // check consistency between max_nb_conn and nbng
    if max_nb_conn < nbng {
        log::info!("init_from_hnsw_all: number of neighbours must be less than hnsw max_nb_connection : {} ", max_nb_conn);
        println!("init_from_hnsw_all: number of neighbours must be less than hnsw max_nb_connection : {} ", max_nb_conn);
    }
    let point_indexation = hnsw.get_point_indexation();
    let nb_point = point_indexation.get_nb_point();
    let mut node_set = IndexSet::<DataId>::with_capacity(nb_point);
    // now we have nb_point we can allocate neighbour field, and we push vectors inside as we will fill in ordre we do not know!
    let mut neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point);
    for _i in 0..nb_point {
        neighbours.push(Vec::<OutEdge<F>>::new());
    }        
    //
    let point_indexation = hnsw.get_point_indexation();
    let mut point_iter = point_indexation.into_iter();
    while let Some(point) = point_iter.next() {
        // now point is an Arc<Point<F>>
        // point_id must be in 0..nb_point. CAVEAT This is not enforced as in petgraph. We should check that
        let point_id = point.get_origin_id();
        // remap _point_id
        let (index, _) = node_set.insert_full(point_id);
        //
        let neighbours_hnsw = point.get_neighborhood_id();
        // neighbours_hnsw contains neighbours in each layer
        // we flatten the layers and transfer neighbours to KGraph::_neighbours
        // possibly use a BinaryHeap?
        let nb_layer = neighbours_hnsw.len();
        let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn*nb_layer);
        for i in 0..nb_layer {
            for j in 0..neighbours_hnsw[i].len() {
                // remap id. nodeset enforce reindexation from 0 too nbnodes whatever the number of node will be
                let (neighbour_idx, _) = node_set.insert_full(neighbours_hnsw[i][j].get_origin_id());
                assert!(index != neighbour_idx);
                vec_tmp.push(OutEdge::<F>{ node : neighbour_idx, weight : F::from_f32(neighbours_hnsw[i][j].distance).unwrap()});
            }
        }
        vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
        assert!(vec_tmp.len() <= 1 || vec_tmp[0].weight <= vec_tmp[1].weight);    // temporary , check we did not invert order
        // keep only the asked size. Could we keep more ?
        if vec_tmp.len() < nbng {
            nb_point_below_nbng += 1;
            mean_deficient_neighbour_size += vec_tmp.len();
            log::trace!("neighbours must have {} neighbours, point {} got only {}", max_nbng, point_id, vec_tmp.len());
            if vec_tmp.len() == 0 {
                let p_id = point.get_point_id();
                log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer : {} ", p_id.0, p_id.1);
            }
        }
        vec_tmp.truncate(nbng);
        mean_nbng += vec_tmp.len() as u64;
        minimum_nbng = minimum_nbng.min(vec_tmp.len());
        //
        // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
        // =====================================================================================================================================
        //  
        neighbours[index] = vec_tmp;
    }
    let nbnodes = neighbours.len();
    assert_eq!(neighbours.len(), nb_point);
    log::trace!("KGraph::exiting init_from_hnsw_all");
    // now we can fill some statistics on density and incoming degrees for nodes!
    log::info!("mean number of neighbours obtained = {:.3e}, minimal number of neighbours {}", mean_nbng as f64 / nb_point as f64, minimum_nbng);
    if nb_point_below_nbng > 0 {
        log::info!("number of points with less than : {} neighbours = {},  mean size for deficient neighbourhhod {:.3e}", nbng, nb_point_below_nbng, 
                    mean_deficient_neighbour_size as f64/nb_point_below_nbng as f64 );
        }
    let mean_nbng = mean_nbng as f64 / nb_point as f64;
    if mean_nbng < nbng as f64 {
        log::warn!(" mean number of neighbours obtained : {:.3e}", mean_nbng);
        log::warn!(" possibly use hnsw.set_keeping_pruned(true)");
        println!(" mean number of neighbours obtained : {:.3e}", mean_nbng);
        println!(" possibly use hnsw.set_keeping_pruned(true)");
    }
    //
    Ok(KGraph{max_nbng, nbnodes, neighbours, node_set})
}   // end kgraph_from_hnsw_all



    /// extract points from layers (less populated) above a given layer (this provides sub sampling where each point has nbng neighbours.  
    /// 
    /// The number of neighbours asked for must be smaller than for init_from_hnsw_all as we do inspect only 
    /// a fraction of the points and a fraction of the neighbourhood of each point. (all the focus is inside a layer)
    pub fn kgraph_from_hnsw_layer<T, D, F>(hnsw : &Hnsw<T,D>, nbng : usize ,layer : usize) -> std::result::Result<KGraph<F>, usize> 
        where   T : Clone + Send + Sync, 
                D : Distance<T> + Send + Sync,
                F : Float + FromPrimitive {
        //
        log::trace!("init_from_hnsw_layer");
        //
        let max_nbng = nbng;
        let mut nb_point_below_nbng: usize = 0;
        let mut mean_deficient_neighbour_size: usize = 0;
        let mut minimum_nbng = nbng;
        let mut mean_nbng = 0u64;
        let max_nb_conn = hnsw.get_max_nb_connection() as usize;
        let max_level_observed = hnsw.get_max_level_observed() as usize;
        let mut nb_point = 0;
        for l in (layer..=max_level_observed).rev() {  
                nb_point += hnsw.get_point_indexation().get_layer_nb_point(l);
        }
        log::trace!("init_from_hnsw_layer down to layer {} collecting nbpoint : {}", layer, nb_point);
        // now we have nb_point we can allocate neighbour field, and we push vectors inside as we will fill in an order we do not know!
        let mut node_set = IndexSet::<DataId>::with_capacity(nb_point);
        let mut neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point);
        for _i in 0..nb_point {
            neighbours.push(Vec::<OutEdge<F>>::new());
        }
        let mut nb_point_collected = 0;
        //
        for l in (layer..=max_level_observed).rev() {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            //
            while let Some(point) = layer_iter.next() {
                // now point is an Arc<Point<F>>
                // point_id must be in 0..nb_point. CAVEAT This is not enforced as in petgraph. We should check that
                let origin_id = point.get_origin_id();
                let p_id = point.get_point_id();
                // remap _point_id
                let (index, _) = node_set.insert_full(origin_id);
                if index >= nb_point {
                    log::trace!("init_from_hnsw_layer point_id {} index {}", origin_id, index);
                    assert!(index < nb_point);
                }
                //
                let neighbours_hnsw = point.get_neighborhood_id();
                // get neighbours of point in the same layer
                // possibly use a BinaryHeap?
                let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn);
                // scan all neighbours in upper layer to keep 
                for m in layer..=max_level_observed {
                    for j in 0..neighbours_hnsw[m].len() {
                        let n_origin_id = neighbours_hnsw[m][j].get_origin_id();
                        let n_p_id = neighbours_hnsw[m][j].p_id;
                        if n_p_id.0 as usize >= l {
                            // remap id. nodeset enforce reindexation from 0 to nbpoint
                            let (neighbour_idx, _) = node_set.insert_full(n_origin_id);
                            vec_tmp.push(OutEdge::<F>{ node : neighbour_idx, weight : F::from_f32(neighbours_hnsw[m][j].distance).unwrap()});
                        }
                    } // end of for j
                } // end of for on m
                vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
                assert!(vec_tmp.len() <= 1 || vec_tmp[0].weight <= vec_tmp[1].weight);    // temporary , check we did not invert order
                // keep only the asked size. Could we keep more ?
                if vec_tmp.len() < nbng {
                    nb_point_below_nbng += 1;
                    mean_deficient_neighbour_size += vec_tmp.len();
                    log::trace!("neighbours must have {} neighbours, got only {}. layer {}  , pos in layer : {}", nbng, vec_tmp.len(),  p_id.0, p_id.1);
                    if vec_tmp.len() == 0 {
                        let p_id = point.get_point_id();
                        log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer : {} ", p_id.0, p_id.1);
                        node_set.swap_remove(&index);
                        continue;
                    }
                } 
                vec_tmp.truncate(nbng);
                nb_point_collected += 1;
                mean_nbng += vec_tmp.len() as u64;
                minimum_nbng = minimum_nbng.min(vec_tmp.len());
                // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
                neighbours[index] = vec_tmp;
            } // end of while
        }
        let nbnodes = neighbours.len();
        assert_eq!(nbnodes, nb_point);
        log::trace!("KGraph::exiting init_from_hnsw_layer");
        log::trace!("collected {} points", nb_point_collected);
        // now we can fill some statistics on density and incoming degrees for nodes!
        let mean_nbng = mean_nbng as f64 / nb_point_collected as f64;
        log::info!("mean number of neighbours obtained = {:.3e} minimal number of neighbours {}", mean_nbng, minimum_nbng);
        if nb_point_below_nbng > 0 {
            log::info!("number of points with less than : {} neighbours = {},  mean size for deficient neighbourhhod {:.3e}", nbng, 
                nb_point_below_nbng,  mean_deficient_neighbour_size as f64/nb_point_below_nbng as f64);
        }
        if mean_nbng < nbng as f64 {
            println!(" mean number of neighbours obtained : {:.3e}", mean_nbng);
            println!(" possibly use hnsw.reset_keeping_pruned(true)");
        }
        //
        Ok(KGraph{max_nbng, nbnodes, neighbours, node_set})
    } // end of init_from_hnsw_layer



//==========================================================================================





#[cfg(test)]
mod tests {

//    cargo test fromhnsw  -- --nocapture
//    cargo test  fromhnsw::tests::test_graph_projection -- --nocapture
//    RUST_LOG=annembed::fromhnsw=TRACE cargo test fromhnsw -- --nocapture

use super::*;

use std::fs::OpenOptions;
use std::path::Path;
use std::io::BufWriter;

use rand::distributions::Uniform;
use rand::prelude::*;

use crate::fromhnsw::hubness;

#[cfg(test)]
fn log_init_test() {
    let res = env_logger::builder().is_test(true).try_init();
    if res.is_err() {
        println!("could not init log");
    }
}  // end of log_init_test


fn gen_rand_data_f32(nb_elem: usize , dim:usize) -> Vec<Vec<f32>> {
    let mut data = Vec::<Vec<f32>>::with_capacity(nb_elem);
    let mut rng = thread_rng();
    let unif =  Uniform::<f32>::new(0.,1.);
    for i in 0..nb_elem {
        let val = 10. * i as f32 * rng.sample(unif);
        let v :Vec<f32> = (0..dim).into_iter().map(|_|  val * rng.sample(unif)).collect();
        data.push(v);
    }
    data
} // end of gen_rand_data_f32


/// test conversion of full hnsw to KGraph and dimension estimation. 
/// mean intrinsic dimension should be around 30 as it is the dimension we use generate random unifrom data 
#[test]
fn test_full_hnsw() {
    //
    log_init_test();
    //
    let nb_elem = 20000;
    let dim = 30;
    let knbn = 20;
    //
    log::debug!("test_full_hnsw");
    println!("\n\n test_serial nb_elem {:?}", nb_elem);
    //
    let data = gen_rand_data_f32(nb_elem, dim);
    let data_with_id = data.iter().zip(0..data.len()).collect();

    let ef_c = 50;
    let max_nb_connection = 50;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let mut hns = Hnsw::<f32, DistL1>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL1{});
    // to enforce the asked number of neighbour
    hns.set_keeping_pruned(true);
    hns.parallel_insert(&data_with_id);
    hns.dump_layer_info();
    //
    log::info!("calling kgraph.init_from_hnsw_all");
    let kgraph : KGraph<f32> = kgraph_from_hnsw_all(&hns, knbn).unwrap();
    log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
    let _kgraph_stats = kgraph.get_kraph_stats();
    // make a test for dimension estimation
    let id = 10;
    let dimension = kgraph.intrinsic_dim_at_data_id(&id).unwrap();
    println!("dimension around point : {}, dim = {:.3e}", id, dimension);
    log::info!("\n dimension around point : {}, dim = {:.3e}", id, dimension);
    //
    let dimension =  kgraph.estimate_intrinsic_dim(10000);
    assert!(dimension.is_ok());
    let dimension = dimension.unwrap();
    log::info!("\n estimation of dimension : {:.3e}, sigma : {:.3e} ", dimension.0 , dimension.1);
    println!("\n estimation of dimension : {:.3e}, sigma : {:.3e} ", dimension.0 , dimension.1);
    // test hubness estimation
    let hubness = self::hubness::Hubness::new(&kgraph);
    let s3 = hubness.get_standard3m();
    log::info!(" estimation of hubness : {:.3e}", s3);

}  // end of test_full_hnsw


#[test]
fn test_layer_hnsw() {
    //
    log_init_test();
    //
    let nb_elem = 80000;
    let dim = 30;
    let knbn = 20;
    //
    println!("\n\n test_serial nb_elem {:?}", nb_elem);
    //
    let data = gen_rand_data_f32(nb_elem, dim);
    let data_with_id : Vec<(&Vec<f32>, usize)> = data.iter().zip(0..data.len()).collect();

    let ef_c = 50;
    let layer = 1;
    let max_nb_connection = 64;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let mut hns = Hnsw::<f32, DistL1>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL1{});
    // to enforce the asked number of neighbour
    hns.set_keeping_pruned(true);
//    hns.set_extend_candidates(true);
    hns.parallel_insert(&data_with_id);
/*     for d in data_with_id {
        hns.insert(d);
    } */
    hns.dump_layer_info();
    //
    log::info!("calling kgraph.init_from_hnsw_layer");
    let kgraph : KGraph<f32> = kgraph_from_hnsw_layer(&hns, knbn, layer).unwrap();
    log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
    let _kgraph_stats = kgraph.get_kraph_stats();
    // testing output for ripser
    let fname = "test_ripser_output";
    log::info!("testing ripser output in file : {}", fname);
    let path = Path::new(fname);
    log::debug!("in to_ripser_sparse_dist : fname : {}", path.display());
    let fileres = OpenOptions::new().write(true).create(true).open(path);
    let file;
    if fileres.is_ok() {
        file = fileres.unwrap();
        let mut bufwriter = BufWriter::new(file);
        let res = kgraph.to_ripser_sparse_dist(&mut bufwriter);
        if res.is_err() {
            log::error!("kgraph.to_ripser_sparse_dist in {} failed", fname);
        }
    }
    else {
        log::error!("cannot open {}", path.display());
        assert_eq!(1,0);
    }
}  // end of test_layer_hnsw



#[test]
fn test_small_indexset() {
    let _ = env_logger::builder().is_test(true).try_init();
    let size = 30;
    let mut idxset = IndexSet::<usize>::with_capacity(size);
    let from = 10000;
    let between = Uniform::from(from..from+size);
    let mut rng = rand::thread_rng();

    for _i in 0..100000 {
        let xsi = between.sample(&mut rng);
        let (idx, _) = idxset.insert_full(xsi);
        assert!(idx < size);
    }
}  // end of test_small_indexset



} // end of tests