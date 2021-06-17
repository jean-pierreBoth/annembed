//! Get a very simple graph from hnsw to be used in kruksal algo and
//! neighborhood entropy computations
//! 
//! 

use num_traits::{Float};
use num_traits::cast::FromPrimitive;

use indexmap::set::*;

use core::ops::*;  // for  AddAssign + SubAssign + MulAssign + DivAssign + RemAssign 
use std::fmt::*;   // for Display + Debug + LowerExp + UpperExp 
use std::cmp::Ordering;


use quantiles::{ckms::CKMS};     // we could use also greenwald_khanna


use hnsw_rs::prelude::*;
use hnsw_rs::hnsw::{Neighbour, DataId};



// morally F should be f32 and f64.  
// The solution from ndArray is F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Display + Debug + LowerExp + UpperExp + (ScalarOperand + LinalgScalar) + Send + Sync.   
// For edge weight we just need  F : FromPrimitive + Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Display + Debug + LowerExp + UpperExp + Send + Sync 

/// keep a node index compatible with NdArray
pub type NodeIdx = usize;

/// an outEdge gives the destination node and weight of edge.
#[derive(Clone,Copy,Debug)]
pub struct OutEdge<F> {
    pub node : NodeIdx,
    pub weight: F
}  // end of struct OutEdge<F>


impl <F>  OutEdge<F> {
    pub fn new(node:NodeIdx, weight: F) -> Self {
        OutEdge{node, weight}
    }
}

impl <F> PartialEq for OutEdge<F> 
    where F : Float {
    fn eq(&self, other: &OutEdge<F>) -> bool {
        return self.weight == other.weight;
    } // end eq
}


// CAVEAT coud use the PointWithOrder<T> implementation for Ord which panic on Nan.
/// order points by distance to self.
impl <F:Float> PartialOrd for OutEdge<F> {
    fn partial_cmp(&self, other: &OutEdge<F>) -> Option<Ordering> {
        self.weight.partial_cmp(& other.weight)
    } // end cmp
} // end impl PartialOrd


/// convert a neigbour from Hnsw to an edge in GraphK
impl <F> From<Neighbour> for OutEdge<F> 
            where F  : Float + FromPrimitive {
    //
    fn from(neighbour : Neighbour) -> OutEdge<F> {
        OutEdge{
            node : neighbour.d_id,
            weight : F::from_f32(neighbour.distance).unwrap()
        }
    } // end of from

}


//====================================================================================================


/// A structure to keep track of min and max distance to neighbour.
/// We keep assume that Nan are excluded once we have reached the point we need this.
struct RangeNgbh<F:Float>(F, F);


/// We may need  some statistics on the graph:
///  - range: distance to nearest and farthest nodes of each node
///  - how many edges arrives in a node (in_degree)
///  - quantiles for the distance to nearest neighbour of nodes
pub struct KGraphStat<F:Float> {
    /// for each node, distances to nearest and farthest neighbours
    ranges : Vec<RangeNgbh<F>>,
    /// incoming degrees
    in_degrees : Vec<u32>,
    /// mean incoming degree
    mean_in_degree : usize,
    /// max incoming degree. Useful to choose between Compressed Storage Mat or dense Array
    max_in_degree : usize,
    /// range statistics. We assume at this step we can use f32 
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
pub struct KGraph<F> {
    /// max number of neighbours of each node. Note it can a littel less.
    max_nbng : usize,
    /// number of nodes.
    /// If GraphK is initialized from the descendant of a point in Hnsw we do not know in advance the number of nodes!!
    nbnodes: usize,
    /// neighbours\[i\] contains the indexes of neighbours node i sorted by increasing weight edge!
    /// all node indexing is done after indexation in node_set
    pub neighbours : Vec<Vec<OutEdge<F>>>,
    /// to keep track of current node indexes.
    node_set : IndexSet<NodeIdx>,
}   // end of struct KGraph





impl <F> KGraph<F> 
    where F : FromPrimitive + Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
        Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync 
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

    /// get out edges from node
    pub fn get_out_edges(&self, node : NodeIdx) -> &Vec<OutEdge<F>> {
        &self.neighbours[node]
    }

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

    /// necessary after embedding to get back to original indexes.
    pub(crate) fn get_indexset(&self) -> &IndexSet<DataId> {
        &self.node_set
    }

    /// Fills in KGraphStat from KGraph
    pub fn get_kraph_stats(&self) -> KGraphStat<F> {
        let mut in_degrees : Vec<u32> = (0..self.nbnodes).into_iter().map(|_| 0).collect();
        let mut ranges = Vec::<RangeNgbh<F>>::with_capacity(self.nbnodes);
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
                ranges.push(RangeNgbh(min_r, max_r));
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
        mean_in_degree /= in_degrees.len() as f32;
        //
        println!("\n ==========================");
        println!("\n minimal graph statistics \n");
        println!("max in degree : {:.2e}", max_in_degree);
        println!("mean in degree : {:.2e}", mean_in_degree);
        println!("max max range : {:.2e} ", max_max_r);
        println!("min min range : {:.2e} ", min_min_r);
        if quant.count() > 0 {
            println!("min radius quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
                        quant.query(0.05).unwrap().1, quant.query(0.5).unwrap().1, 
                        quant.query(0.95).unwrap().1, quant.query(0.99).unwrap().1);
        }
        //
        KGraphStat{ranges, in_degrees, mean_in_degree : mean_in_degree.round() as usize, max_in_degree : max_in_degree as usize, 
                    min_radius_q : quant}
    }  // end of get_kraph_stats




    /// initialization of a graph with expected number of neighbours nbng.  
    /// 
    /// This initialization corresponds to the case where use all points of the hnsw structure
    /// see also *initialize_from_layer* and *initialize_from_descendants*.   
    /// nbng is the maximal number of neighbours kept. The effective mean number can be less,
    /// in this case use the Hnsw.set_keeping_pruned(true) to restrict pruning in the search.
    ///
    pub fn init_from_hnsw_all<D>(&mut self, hnsw : &Hnsw<F,D>, nbng : usize) -> std::result::Result<usize, usize> 
        where   F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
                    Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync,
                D : Distance<F> + Send + Sync {
        //
        log::info!("entering init_from_hnsw_all");
        //
        self.max_nbng = nbng;
        let mut nb_point_below_nbng = 0;
        let mut minimum_nbng = nbng;
        let mut mean_nbng = 0u64;
        // We must extract the whole structure , for each point the list of its nearest neighbours and weight<F> of corresponding edge
        let max_nb_conn = hnsw.get_max_nb_connection() as usize;    // morally this the k of knn bu we have that for each layer
        // check consistency between max_nb_conn and nbng
        if max_nb_conn <= nbng {
            log::error!("init_from_hnsw_all: number of neighbours must be greater than hnsw max_nb_connection : {} ", max_nb_conn);
            println!("init_from_hnsw_all: number of neighbours must be greater than hnsw max_nb_connection : {} ", max_nb_conn);
            return Err(1);
        }
        let point_indexation = hnsw.get_point_indexation();
        let nb_point = point_indexation.get_nb_point();
        self.node_set = IndexSet::with_capacity(nb_point);
        // now we have nb_point we can allocate neighbour field, and we push vectors inside as we will fill in ordre we do not know!
        self.neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point);
        for _i in 0..nb_point {
            self.neighbours.push(Vec::<OutEdge<F>>::new());
        }        
        //
        let point_indexation = hnsw.get_point_indexation();
        let mut point_iter = point_indexation.into_iter();
        while let Some(point) = point_iter.next() {
            // now point is an Arc<Point<F>>
            // point_id must be in 0..nb_point. CAVEAT This is not enforced as in petgraph. We should check that
            let point_id = point.get_origin_id();
            // remap _point_id
            let (index, _) = self.node_set.insert_full(point_id);
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
                    let (neighbour_idx, _already) = self.node_set.insert_full(neighbours_hnsw[i][j].get_origin_id());
                    assert!(index != neighbour_idx);
                    vec_tmp.push(OutEdge::<F>{ node : neighbour_idx, weight : F::from_f32(neighbours_hnsw[i][j].distance).unwrap()});
                }
            }
            vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
            assert!(vec_tmp.len() <= 1 || vec_tmp[0].weight <= vec_tmp[1].weight);    // temporary , check we did not invert order
            // keep only the asked size. Could we keep more ?
            if vec_tmp.len() < nbng {
                nb_point_below_nbng += 1;
                log::warn!("neighbours must have {} neighbours, point {} got only {}", point_id, self.max_nbng, vec_tmp.len());
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
            self.neighbours[index] = vec_tmp;
        }
        self.nbnodes = self.neighbours.len();
        assert_eq!(self.nbnodes, nb_point);
        log::trace!("KGraph::exiting init_from_hnsw_all");
        // now we can fill some statistics on density and incoming degrees for nodes!
        log::info!("mean number of neighbours obtained = {:.2e}", mean_nbng as f64 / nb_point as f64);
        log::info!("minimal number of neighbours {}", minimum_nbng);
        log::info!("number of points with less than : {} neighbours = {} ", nbng, nb_point_below_nbng);
        if (mean_nbng as f64 / nb_point as f64) < nbng as f64 {
            println!(" mean number of neighbours obtained : {:2.e}", mean_nbng);
            println!(" possibly use hnsw.reset_keeping_pruned(true)");
        }
        //
        Ok(minimum_nbng)
    }   // end init_from_hnsw_all


    /// extract points from a given layer (this provides sub sampling where each point has nbng neighbours.  
    /// 
    /// The number of neighbours asked for must be smaller than for init_from_hnsw_all as we do inspect only 
    /// a fraction of the points and a fraction of the neighbourhood of each point. (all the focus is inside a layer)
    pub fn init_from_hnsw_layer<D>(&mut self, hnsw : &Hnsw<F,D>, nbng : usize ,layer : usize) -> std::result::Result<usize, usize> 
        where   F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
                    Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync,
                D : Distance<F> + Send + Sync {
        //
        log::trace!("init_from_hnsw_layer");
        //
        self.max_nbng = nbng;
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
        self.neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point);
        for _i in 0..nb_point {
            self.neighbours.push(Vec::<OutEdge<F>>::new());
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
                if p_id.0 as usize != l {
                    log::trace!("p_id layer : {} ",p_id.0 as usize);
                }
                assert_eq!(p_id.0 as usize, l);
                // remap _point_id
                let (index, _) = self.node_set.insert_full(origin_id);
                if index >= nb_point {
                    log::trace!("init_from_hnsw_layer point_id {} index {}", origin_id, index);
                    assert!(index < nb_point);
                }
                //
                let neighbours_hnsw = point.get_neighborhood_id();
                // get neighbours of point in the same layer
                // possibly use a BinaryHeap?
                let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn);
                for j in 0..neighbours_hnsw[l].len() {
                    let n_origin_id = neighbours_hnsw[l][j].get_origin_id();
                    let n_p_id = neighbours_hnsw[l][j].p_id;
                    if n_p_id.0 as usize >= l {
                    // remap id. nodeset enforce reindexation from 0 to nbpoint
                        let (neighbour_idx, _already) = self.node_set.insert_full(n_origin_id);
                        vec_tmp.push(OutEdge::<F>{ node : neighbour_idx, weight : F::from_f32(neighbours_hnsw[l][j].distance).unwrap()});
                    }
                } // end of for j
                vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
                assert!(vec_tmp.len() <= 1 || vec_tmp[0].weight <= vec_tmp[1].weight);    // temporary , check we did not invert order
                // keep only the asked size. Could we keep more ?
                if vec_tmp.len() < nbng {
                    nb_point_below_nbng += 1;
                    log::warn!("neighbours must have {} neighbours, got only {}", self.max_nbng, vec_tmp.len());
                    log::warn!(" layer {}  , pos in layer {} ", p_id.0, p_id.1);
                    if vec_tmp.len() == 0 {
                        let p_id = point.get_point_id();
                        log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer {} ", p_id.0, p_id.1);
                        self.node_set.remove(&index);
                        continue;
                    }
                } 
                vec_tmp.truncate(nbng);
                nb_point_collected += 1;
                mean_nbng += vec_tmp.len() as u64;
                minimum_nbng = minimum_nbng.min(vec_tmp.len());
                // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
                self.neighbours[index] = vec_tmp;
            } // end of while
        }
        self.nbnodes = self.neighbours.len();
        assert_eq!(self.nbnodes, nb_point);
        log::trace!("KGraph::exiting init_from_hnsw_layer");
        log::trace!("collected {} points", nb_point_collected);
        // now we can fill some statistics on density and incoming degrees for nodes!
        log::info!("mean number of neighbours obtained = {:.2e}", mean_nbng as f64 / nb_point_collected as f64);
        log::info!("minimal number of neighbours {}", minimum_nbng);
        log::info!("number of points with less than : {} neighbours = {} ", nbng, nb_point_below_nbng);
        if (mean_nbng as f64 / nb_point as f64) < nbng as f64 {
            println!(" mean number of neighbours obtained : {:2.e}", mean_nbng);
            println!(" possibly use hnsw.reset_keeping_pruned(true)");
        }
        //
        Ok(minimum_nbng)
    } // end of init_from_hnsw_layer
}  // end of impl KGraph<F>




// =============================================================================



mod tests {

//    cargo test fromhnsw  -- --nocapture
//    RUST_LOG=annembed::fromhnsw=TRACE cargo test fromhnsw -- --nocapture

#[allow(unused)]
use super::*;

use rand::distributions::{Uniform};
use rand::prelude::*;

#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  


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


/// test conversion of full hnsw to KGraph
#[test]
fn test_full_hnsw() {
    //
    log_init_test();
    //
    let nb_elem = 20000;
    let dim = 30;
    let knbn = 10;
    //
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
    let mut kgraph = KGraph::<f32>::new();
    log::info!("calling kgraph.init_from_hnsw_all");
    let res = kgraph.init_from_hnsw_all(&hns, knbn);
    if res.is_err() {
        panic!("init_from_hnsw_all  failed");
    }
    log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
    let _kgraph_stats = kgraph.get_kraph_stats();
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
    let mut kgraph = KGraph::<f32>::new();
    log::info!("calling kgraph.init_from_hnsw_layer");
    let res = kgraph.init_from_hnsw_layer(&hns, knbn, layer);
    if res.is_err() {
        panic!("init_from_hnsw_all  failed");
    }
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
        let (idx, _already) = idxset.insert_full(xsi);
        assert!(idx < size);
    }
}


} // end of tests