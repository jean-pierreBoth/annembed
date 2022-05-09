//! Get a very simple graph from hnsw to be used in kruksal algo and
//! neighborhood entropy computations
//! 
//! 

use anyhow::{anyhow};

use num_traits::{Float};
use num_traits::cast::FromPrimitive;

use indexmap::set::*;

use std::cmp::Ordering;

use std::sync::Arc;
use rand::thread_rng;

use quantiles::{ckms::CKMS};     // we could use also greenwald_khanna


use hnsw_rs::prelude::*;
use hnsw_rs::hnsw::{DataId};

use crate::tools::{dimension::*,nodeparam::*};
use rand::distributions::{Distribution};

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
    max_nbng : usize,
    /// number of nodes.
    /// If GraphK is initialized from the descendant of a point in Hnsw we do not know in advance the number of nodes!!
    nbnodes: usize,
    /// neighbours\[i\] contains the indexes of neighbours node i sorted by increasing weight edge!
    /// all node indexing is done after indexation in node_set
    neighbours : Vec<Vec<OutEdge<F>>>,
    /// to keep track of current node indexes.
    node_set : IndexSet<DataId>,
}   // end of struct KGraph





impl <F> KGraph<F> 
    where F : FromPrimitive + Float 
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
    /// When we get embedded data as an Array2<F>, row i of data corresponds to
    /// the original data with label get_data_id_from_idx(i)
    pub fn get_data_id_from_idx(&self, index:usize) -> Option<&DataId> {
        return self.node_set.get_index(index)
    }

    /// get the index corresponding to a given DataId
    pub fn get_idx_from_dataid(&self, data_id: &DataId) -> Option<usize> {
        return self.node_set.get_index_of(data_id)
    }

    /// usefule after embedding to get back to original indexes.
#[allow(unused)]
    pub(crate) fn get_indexset(&self) -> &IndexSet<DataId> {
        &self.node_set
    }

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
        println!("\n ==========================");
        println!("\n minimal graph statistics \n");
        println!("max in degree : {:.2e}", max_in_degree);
        println!("mean in degree : {:.2e}", mean_in_degree);
        println!("max max range : {:.2e} ", max_max_r.to_f32().unwrap());
        println!("min min range : {:.2e} ", min_min_r.to_f32().unwrap());
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
    log::info!("entering kgraph_from_hnsw_all");
    //
    let max_nbng = nbng;
    let mut nb_point_below_nbng = 0;
    let mut minimum_nbng = nbng;
    let mut mean_nbng = 0u64;
    // We must extract the whole structure , for each point the list of its nearest neighbours and weight<F> of corresponding edge
    let max_nb_conn = hnsw.get_max_nb_connection() as usize;    // morally this the k of knn bu we have that for each layer
    // check consistency between max_nb_conn and nbng
    if max_nb_conn <= nbng {
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
            log::trace!("neighbours must have {} neighbours, point {} got only {}", max_nbng, point_id, vec_tmp.len());
            if vec_tmp.len() == 0 {
                let p_id = point.get_point_id();
                log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer {} ", p_id.0, p_id.1);
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
    log::info!("mean number of neighbours obtained = {:.3e}", mean_nbng as f64 / nb_point as f64);
    log::info!("minimal number of neighbours {}", minimum_nbng);
    log::info!("number of points with less than : {} neighbours = {} ", nbng, nb_point_below_nbng);
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
        let mut nb_point_below_nbng = 0;
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
                    log::warn!("neighbours must have {} neighbours, got only {}", max_nbng, vec_tmp.len());
                    log::warn!(" layer {}  , pos in layer {} ", p_id.0, p_id.1);
                    if vec_tmp.len() == 0 {
                        let p_id = point.get_point_id();
                        log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer {} ", p_id.0, p_id.1);
                        node_set.remove(&index);
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
        log::info!("mean number of neighbours obtained = {:.3e}", mean_nbng);
        log::info!("minimal number of neighbours {}", minimum_nbng);
        log::info!("number of points with less than : {} neighbours = {} ", nbng, nb_point_below_nbng);
        if mean_nbng < nbng as f64 {
            println!(" mean number of neighbours obtained : {:.3e}", mean_nbng);
            println!(" possibly use hnsw.reset_keeping_pruned(true)");
        }
        //
        Ok(KGraph{max_nbng, nbnodes, neighbours, node_set})
    } // end of init_from_hnsw_layer



//==========================================================================================

use std::collections::HashMap;

#[allow(unused)]
use ndarray::{Array2};



/// Construct a projection from Hnsw data on layers above a given layers
/// Maintain for each point in the Hnsw structure nearest point in projected structure.
/// Possibly stores matrix of distances between filtered points
/// 
pub struct KGraphProjection<F> {
    /// we consider projection on points on layers above and including this layer 
    layer : usize,
    /// graph on which we project
    small_graph: KGraph<F>,
    /// for each data out of the filtered data, we keep an edge to nearest data in the filtered set
    proj_data : HashMap<NodeIdx, OutEdge<F>>,
    /// larger graph that is projected
    large_graph : KGraph<F>,
} // end of struct Projection<F>



impl <F> KGraphProjection<F>
    where F : Float + FromPrimitive {

    //  - first we construct graph consisting in upper (less populated) layers
    //  - Then we project : for points of others layers store the shorter edge from point to graph just constructed
    //  - at last we construct graph for the lower (more populated layers)
    //
    /// construct graph from layers above layer, projects data of aother layers on point in layers above layer
    pub fn new<T, D>(hnsw : &Hnsw<T,D>, nbng : usize, layer : usize) -> Self 
                    where T : Clone + Send + Sync,
                          D: Distance<T> + Send + Sync {
        log::debug!("KGraphProjection new  layer : {}", layer);
        let mut nb_point_to_collect = 0;
        let mut nb_point_below_nbng = 0;
        let max_nb_conn = hnsw.get_max_nb_connection() as usize;
        let max_level_observed = hnsw.get_max_level_observed() as usize;
        log::debug!("max level observed : {}", max_level_observed);
        // check number of points kept in
        if layer >= max_level_observed {
            log::error!("KGraphProjection::new, layer argument greater than nb_layer!!, layer : {}", layer);
            println!("KGraphProjection::new, layer argument greater than nb_layer!!, layer : {}", layer);

        }
        for l in (layer..=max_level_observed).rev() {
            nb_point_to_collect += hnsw.get_point_indexation().get_layer_nb_point(l);
            log::trace!(" layer : {}, nb points to collect : {}", l, nb_point_to_collect);
        }
        if nb_point_to_collect <= 0 {
            log::error!("!!!!!!!!!!!! KGraphProjection cannot collect points !!!!!!!!!!!!!, check layer argument");
            println!("!!!!!!!!!!!! KGraphProjection cannot collect points !!!!!!!!!!!!!, check layer argument");
        }
        // let _points = Vec::<std::sync::Arc<Point<F>>>::with_capacity(nb_point_to_collect);
        // let _distances = Array2::<F>::zeros((nb_point_to_collect,nb_point_to_collect));
        //
        let layer_u8 = layer as u8;
        log::debug!("Projection : number of point to collect : {}", nb_point_to_collect);
        let mut upper_graph_neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point_to_collect);
        for _i in 0..nb_point_to_collect {
            upper_graph_neighbours.push(Vec::<OutEdge<F>>::new());
        }
        let mut index_set = IndexSet::<DataId>::with_capacity(nb_point_to_collect);
        let mut _nb_point_upper_graph = 0;
        // We want to construct an indexation of nodes valid for the whole graph and for the upper layers.
        // So we begin indexation from upper layers
        for l in (layer..=max_level_observed).rev() {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                let (_, _) = index_set.insert_full(point.get_origin_id());
            }
        }
        // now we have an indexation of upper levels (less populatated levels), we clone present state of index_set
        let mut upper_index_set = index_set.clone();
        // we now continue to have the whole indexation 
        for l in (0..layer).rev() {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                let (_, _) = index_set.insert_full(point.get_origin_id());
            }
        }        
        // now we have the whole indexation, we construct upper(less populated) graph
        for l in (layer..=max_level_observed).rev() {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            //
            while let Some(point) = layer_iter.next() {
                // now point is an Arc<Point<F>>
                // point_id , we must reindex point_id for ProjectedDistances structure
                let origin_id = point.get_origin_id();
                let p_id = point.get_point_id();
                // remap _point_id
                let (index, newindex) = upper_index_set.insert_full(origin_id);
                assert_eq!(newindex, false);
                let neighbours_hnsw = point.get_neighborhood_id();
                // get neighbours of point in the same layer
                let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn);
                for m in layer..=max_level_observed {
                    for j in 0..neighbours_hnsw[m].len() {
                        let n_origin_id = neighbours_hnsw[m][j].get_origin_id();
                        let n_p_id = neighbours_hnsw[m][j].p_id;
                        if n_p_id.0 as usize >= layer {
                            // get index of !!! (and not get) IndexMap interface is error prone
                            let neighbour_idx = upper_index_set.get_index_of(&n_origin_id).unwrap();
                            vec_tmp.push(OutEdge::<F>{ node : neighbour_idx, weight : F::from_f32(neighbours_hnsw[m][j].distance).unwrap()});
                        }
                    } // end of for j
                }   // end of for m 
                vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
                assert!(vec_tmp.len() <= 1 || vec_tmp[0].weight <= vec_tmp[1].weight);    // temporary , check we did not invert order
                // keep only the asked size. Could we keep more ?
                if vec_tmp.len() < nbng {
                    nb_point_below_nbng += 1;
                    log::warn!("neighbours must have {} neighbours, got only {}", nbng, vec_tmp.len());
                    log::warn!(" layer {}  , pos in layer {} ", p_id.0, p_id.1);
                    if vec_tmp.len() == 0 {
                        let p_id = point.get_point_id();
                        log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer {} ", p_id.0, p_id.1);
                        upper_index_set.remove(&index);
                        index_set.remove(&index);
                        continue;
                    }
                } 
                vec_tmp.truncate(nbng);
                _nb_point_upper_graph += 1;
                // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
                upper_graph_neighbours[index] = vec_tmp;
                // compute distance with preceding points. Fills distances ????????????
            } // end of while
        } // end loop on upper layers
        // at this step we have upper Graph (and could store distances between points if not too many points) 
        // yet we do not have global projection mapping to NodeId of upper graph
        // so we need to loop on lower layers 
        log::trace!("number of points in upper graph : {} ", _nb_point_upper_graph);
        // now we construct projection
        let mut nb_point_without_projection = 0;
        let mut proj_data : HashMap<NodeIdx, OutEdge<F>> = HashMap::new();
        let mut points_with_no_projection = Vec::<Arc<Point<T>>>::new();
        for l in 0..layer {
            log::trace!("scanning projections of layer {}", l);
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                // we do as in KGraph.init_from_hnsw_layer
                let neighbours_hnsw = point.get_neighborhood_id();
                let best_distance = F::infinity();
                // get nearest point in upper layers.
                // possibly use a BinaryHeap?
                let mut best_edge = OutEdge::<F>{node : usize::MAX, weight : best_distance};
                for m in layer..=max_level_observed {
                    for j in 0..neighbours_hnsw[m].len() {
                        let n_origin_id = neighbours_hnsw[m][j].get_origin_id();
                        let n_p_id = neighbours_hnsw[m][j].p_id;
                        if n_p_id.0 >= layer_u8 && F::from(neighbours_hnsw[m][j].distance).unwrap() < best_distance {
                            // store edge with remapping dataid to nodeidx
                            let neighbour_index = upper_index_set.get_index_of(&n_origin_id).unwrap();
                            best_edge = OutEdge::<F>{ node : neighbour_index, weight : F::from_f32(neighbours_hnsw[m][j].distance).unwrap() };
                        }
                    } // end of for j
                }
                // we have best edge, insert it in 
                if best_edge.weight < F::infinity() && best_edge.node < usize::MAX {
                    let index = index_set.get_index_of(&point.get_origin_id()).unwrap();
                    proj_data.insert(index, best_edge);
                }
                else {
                    let p_id = point.get_point_id();
                    points_with_no_projection.push(Arc::clone(&point));
                    log::trace!(" no projection for point pid {}  {}", p_id.0, p_id.1);
                    nb_point_without_projection += 1; 
                }
                // now we must 
            } // end while
        } // end of search in densely populated layers
        log::info!("number of points without edge defining a projection= {} ", nb_point_without_projection);
        let nb_projection_fixed = fix_points_with_no_projection(&mut proj_data, &index_set, &points_with_no_projection, layer);
        log::info!("number of points without edge defining a projection= {} ", nb_point_without_projection - nb_projection_fixed);
        //
        // define projection by identity on upper layers
        for l in layer..=max_level_observed {
            log::trace!("scanning projections of layer {}", l);
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                let index = upper_index_set.get_index_of(&point.get_origin_id()).unwrap();
                let best_edge = OutEdge::<F>{ node : index, weight : F::from_f32(0.).unwrap() };
                proj_data.insert(index, best_edge);
            }
        }
        //
        log::info!("number of points with less than : {} neighbours = {} ", nbng, nb_point_below_nbng);
        let upper_graph = KGraph::<F>{ max_nbng : nbng, nbnodes : upper_graph_neighbours.len() , neighbours : upper_graph_neighbours, node_set : upper_index_set};
        log::info!("getting stats from reduced graph");
        let _graph_stats = upper_graph.get_kraph_stats();
        //
        // construct the dense whole  graph, lower layers or more populated layers
        //
        log::debug!(" constructing the whole graph");
        nb_point_to_collect = hnsw.get_nb_point();
        let mut graph_neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point_to_collect);
        for _i in 0..nb_point_to_collect {
            graph_neighbours.push(Vec::<OutEdge<F>>::new());
        }
        nb_point_below_nbng = 0;
        for l in 0..=max_level_observed {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            //
            while let Some(point) = layer_iter.next() {
                let origin_id = point.get_origin_id();
                let index = index_set.get_index_of(&origin_id).unwrap();
                let neighbours_hnsw = point.get_neighborhood_id();
                // get neighbours of point in the same layer  possibly use a BinaryHeap?
                let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn);
                for m in 0..layer {
                    for j in 0..neighbours_hnsw[m].len() {
                        let n_origin_id = neighbours_hnsw[m][j].get_origin_id();
                            // points are already indexed , or panic!
                            let neighbour_idx = index_set.get_index_of(&n_origin_id).unwrap();
                            vec_tmp.push(OutEdge::<F>{ node : neighbour_idx, weight : F::from_f32(neighbours_hnsw[m][j].distance).unwrap()});
                    } // end of for j
                }   // end of for m 
                vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
                if vec_tmp.len() < nbng {
                    nb_point_below_nbng += 1;
                    let p_id = point.get_point_id();
                    log::warn!("neighbours must have {} neighbours, got only {}", nbng, vec_tmp.len());
                    log::warn!(" layer {}  , pos in layer {} ", p_id.0, p_id.1);
                    if vec_tmp.len() == 0 {
                        log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer {} ", p_id.0, p_id.1);
                        index_set.remove(&index);
                        continue;
                    }
                } 
                vec_tmp.truncate(nbng);
                _nb_point_upper_graph += 1;
                // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
                graph_neighbours[index] = vec_tmp;
            }
        }  // end for on layers
        // we have both graph and projection
        let whole_graph = KGraph::<F>{ max_nbng : nbng, nbnodes : graph_neighbours.len() , neighbours : graph_neighbours, node_set : index_set};
        log::info!("getting stats from whole graph");
        let _graph_stats = whole_graph.get_kraph_stats();
        // 
        log::info!("number of points with less than : {} neighbours = {} ", nbng, nb_point_below_nbng);
        log::trace!("Projection exiting from new");
        //
        KGraphProjection{ layer, small_graph : upper_graph, proj_data : proj_data, large_graph : whole_graph}
    } // end of new


    /// get layer corresponding above which the projection is done
    pub fn get_layer(&self) -> usize {
        self.layer
    } // end of get_layer


    /// returns the edge defingin distance to nearest element in projected (small) graph
    /// The argument is a NodeIdx
    pub fn get_projection_by_nodeidx(&self, nodeidx : &NodeIdx) -> OutEdge<F> {
        *(self.proj_data.get(&nodeidx).unwrap())
    } // end of get_distance_to_projection


    /// returns the distance to nearest element in projected (small) graph
    /// The argument is a NodeIdx
    pub fn get_distance_to_projection_by_dataid(&self, data_id : &DataId) -> F {
        let edge = self.proj_data.get(&data_id).unwrap();
        self.proj_data.get(&edge.node).unwrap().weight
    } // end of get_distance_to_projection

    // return a reference to the small graph
    pub fn get_small_graph(&self) -> &KGraph<F> {
        &self.small_graph
    }

    // returns a reference to the large graph
    pub fn get_large_graph(&self) -> &KGraph<F> {
        &self.large_graph
    }

    /// returns quantile stats on distances to projection point
    pub fn get_projection_distance_quant(&self) -> CKMS<f32> {
        let mut quant = CKMS::<f32>::new(0.001);
        for (_, edge) in self.proj_data.iter() {
            quant.insert(F::to_f32(&edge.weight).unwrap());
        }
        //
        quant
    }
}  // end of impl block




// search a projection among projection of neighbours
fn get_point_approximate_projection<T, F>(point : &Arc<Point<T>>, proj_data : &mut HashMap<NodeIdx, OutEdge<F>> , 
        index_set : &IndexSet::<DataId>, layer : usize) -> Option<OutEdge<F>> 
        where   T : Clone + Send + Sync,
                F : Float {
    //
    let neighbours_hnsw = point.get_neighborhood_id();
    // search a neighbour with a projection
    for l in 0..=layer {
        let nbng_l = neighbours_hnsw[l].len();
        for j in 0..nbng_l {
            let n_origin_id = neighbours_hnsw[l][j].get_origin_id();
            // has this point a projection ?
            let neighbour_idx = index_set.get_index_of(&n_origin_id).unwrap();
            if let Some(proj) = proj_data.get(&neighbour_idx) {
                return Some(proj.clone());
            }
        }
    }
    return None;
} // end of get_approximate_projection


// hack to approximate projections for problem points. We take the projection of neighbour 
fn fix_points_with_no_projection<T,F>(proj_data : &mut HashMap<NodeIdx, OutEdge<F>>, index_set : &IndexSet::<DataId>, points : &Vec::<Arc<Point<T>>>, layer : usize) -> usize
        where   T: Clone + Send + Sync,
                F : Float {
    //
    let nb_points_noproj = points.len();
    let mut nb_fixed = 0;
    log::debug!("fix_points_with_no_projection, nb points : {} ", nb_points_noproj);
    //
    for point in points.iter() {
    if let Some(edge) = get_point_approximate_projection(point, proj_data, index_set, layer) {
        let origin_id = point.get_origin_id();
        let idx = index_set.get_index_of(&origin_id).unwrap();
        proj_data.insert(idx, edge);
        nb_fixed += 1;
    }
    } // end of for on points
    //
    log::trace!("fix_points_with_no_projection nb fixed {}", nb_fixed);
    //
    nb_fixed
} // end of fix_points_with_no_projection


// =============================================================================






mod tests {

//    cargo test fromhnsw  -- --nocapture
//    cargo test  fromhnsw::tests::test_graph_projection -- --nocapture
//    RUST_LOG=annembed::fromhnsw=TRACE cargo test fromhnsw -- --nocapture

#[allow(unused)]
use super::*;

use rand::distributions::{Uniform};
use rand::prelude::*;

#[allow(dead_code)]
fn log_init_test() {
    let res = env_logger::builder().is_test(true).try_init();
    if res.is_err() {
        println!("could not init log");
    }
}  // end of log_init_test


#[allow(unused)]
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
}


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


#[test]
fn test_graph_projection() {
    log_init_test();
    //
    let nb_elem = 80000;
    let dim = 30;
    let knbn = 10;
    //
    println!("\n\n test_graph_projection nb_elem {:?}", nb_elem);
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
    hns.parallel_insert(&data_with_id);
    hns.dump_layer_info();
    //
    let _graph_projection = KGraphProjection::<f32>::new(&hns, knbn , layer);


} // end of test_graph_projection


} // end of tests