//! Get a very simple graph from hnsw to be used in kruksal algo and
//! neighborhood entropy computations
//! 
//! 

use num_traits::{Float};

use indexmap::set::*;

use core::ops::*;  // for  AddAssign + SubAssign + MulAssign + DivAssign + RemAssign 
use std::fmt::*;   // for Display + Debug + LowerExp + UpperExp 
use std::cmp::Ordering;
use num_traits::cast::FromPrimitive;

use hnsw_rs::prelude::*;
use hnsw_rs::hnsw::Neighbour;

/// morally F should be f32 and f64
/// The solution from ndArray is F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Display + Debug + LowerExp + UpperExp + (ScalarOperand + LinalgScalar) + Send + Sync 
/// For edge weight we just need  F : FromPrimitive + Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Display + Debug + LowerExp + UpperExp + Send + Sync 



/// keep a node index compatible with NdArray
pub type NodeIdx = usize;

/// an outEdge gives the destination node and weight of edge.
#[derive(Clone,Copy,Debug)]
pub struct OutEdge<F> {
    node : NodeIdx,
    weight: F
}  // end of struct OutEdge<F>


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

use num_traits::real::Real;

/// A structure to keep track of min and max distance to neighbour.
/// We keep assume that Nan are excluded once we have reached the point we need this.
struct RangeNgbh<F:Float>(F, F);


/// We may need  some statistics on the graph
///  - range of neighbourhood
///  - how many edges arrives in a node (in_degree)
pub struct KGraphStat<F:Float> {
    /// min and max negihbours for each node
    ranges : Vec<RangeNgbh<F>>,
    /// incoming degrees
    in_degrees : Vec<u32>,
}  // end of KGraphStat


impl <F:Float> KGraphStat<F> {
    // extract a density index on point based on max distance of k-th Neighbour
    pub fn get_density_index(&self) -> Vec<F> {
        let density = self.ranges.iter().map(|x| x.1.recip()).collect();
        //
        return density;
    } // get_density_index

}  // end of impl block for KGraphStat



/// 
/// A very minimal graph for this crate (otherwise use petgraph)
/// The graph comes from an k-nn search so we know the number of neighbours we have
/// W is a weight on edges and must satisfy Ord, hence the structure OutEdge<F>
/// 
/// The nodes must be indexed from 0 to nbnodes-1 (same as hnsw_rs crate)
/// 
/// The first initialization from hnsw is a full hnsw representation,
/// but it should be possible to selecat a layer to get a subsampling of data
/// or the whole children of a given node at any layer to get a specific region of the data. 
/// 
pub(crate) struct KGraph<F> {
    /// The number of neighbours of each node.
    nbng : usize,
    /// numboer of nodes. The nodes must be numbered from 0 to nb_nodes.
    /// If GraphK is initialized from the descendant of a point in Hnsw we do not know in advance the number of nodes!!
    nbnodes: usize,
    /// an edge is given by 2 nodes and a weight
    edges : Vec<(NodeIdx, OutEdge<F>)>,
    /// neighbours[i] contains the indexes of neighbours node i sorted by increasing weight edge!
    neighbours : Vec<Vec<OutEdge<F>>>,
    /// to keep track of current node indexes.
    node_set : IndexSet<NodeIdx>,

}   // end of struct KGraph





impl <F> KGraph<F> 
    where F : FromPrimitive + Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
        Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync 
{

    /// allocates a graph with expected size nbnodes and nbng neighbours 
    pub fn new(nbng : usize) -> Self {
        let neighbours_init = Vec::<Vec<OutEdge<F>>>::new();
        KGraph {
            nbng : nbng,
            nbnodes : 0,
            edges : Vec::< (NodeIdx, OutEdge<F>) >::new(),
            neighbours :  neighbours_init,
            node_set : IndexSet::new(),
        }
    }  // end of new



    // edges are supposed directed 
    fn insert_edge_list(edges : &[(NodeIdx, NodeIdx, OutEdge<F>)]) {

        let nb_edge = edges.len();
        for _i in 0..nb_edge {
                // check source and target node


        }


    }  // end of insert_edge_list


    pub fn get_kraph_stats(&self) -> KGraphStat<F> {
        let mut in_degrees = Vec::<u32>::with_capacity(self.nbnodes);
        let mut ranges = Vec::<RangeNgbh<F>>::with_capacity(self.nbnodes);
        //
        let mut max_max_r = F::zero();
        let mut min_min_r = F::max_value();
        //
        for i in 0..self.neighbours.len() {
            let min_r = self.neighbours[i][0].weight;
            let max_r = self.neighbours[i][self.neighbours[i].len()-1].weight;
            //
            max_max_r = max_max_r.max(max_r);
            min_min_r = min_min_r.min(min_r);
            //
            ranges.push(RangeNgbh(min_r, max_r));
            for j in 0..self.neighbours[i].len() {
                    in_degrees[self.neighbours[i][j].node] += 1;
            }
        }
        // dump some info
        let mut max_nb = 0;
        for i in 0..in_degrees.len() {
            max_nb = max_nb.max(in_degrees[i]);
        }
        //
        println!("\n ==========================");
        println!("\n minimal graph statistics \n");
        println!("max in degree : {}", max_nb);
        println!("max max range : {} ", max_max_r);
        println!("min min range : {} ", min_min_r);
        //
        KGraphStat{ranges, in_degrees }
    }  // end of get_kraph_stats





    /// initialization for the case we use all points of the hnsw structure
    /// see also *initialize_from_layer* and *initialize_from_descendants*
    /// 
    pub fn init_from_hnsw_all<D>(&mut self, hnsw : &Hnsw<F,D>) -> std::result::Result<usize, usize> 
        where   F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
                    Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync,
                D : Distance<F> + Send + Sync {
        //
        // We must extract the whole structure , for each point the list of its nearest neighbours and weight<F> of corresponding edge
        let max_nb_conn = hnsw.get_max_nb_connection() as usize;    // morally this the k of knn bu we have that for each layer
        // check consistency between max_nb_conn and nbng
        if max_nb_conn <= self.nbng {
            log::error!("init_from_hnsw_all: number of neighbours must be greater than hnsw max_nb_connection : {} ", max_nb_conn);
            println!("init_from_hnsw_all: number of neighbours must be greater than hnsw max_nb_connection : {} ", max_nb_conn);
            return Err(1);
        }
        let point_indexation = hnsw.get_point_indexation();
        let nb_point = point_indexation.get_nb_point();
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
                    vec_tmp.push(OutEdge::<F>{ node : neighbour_idx, weight : F::from_f32(neighbours_hnsw[i][j].distance).unwrap()});
                }
            }
            vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
            assert!(vec_tmp[0].weight < vec_tmp[1].weight);    // temporary , check we did not invert order
            // keep only the good size. Could we keep more ?
            if vec_tmp.len() < max_nb_conn {
                log::error!("neighbours must have {} neighbours", self.nbng);
                return Err(1);                
            }
            vec_tmp.truncate(max_nb_conn);
            //
            // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
            // =====================================================================================================================================
            //  
            self.neighbours[index] = vec_tmp;
        }
        self.nbnodes = self.neighbours.len();
        assert_eq!(self.nbnodes, nb_point);
        // now we can fill some statistics on density and incoming degrees for nodes!
        //
        Ok(1)
    }   // end init_from_hnsw_all


    // convert into a matrix representation for spectral embedding
    // either a CsMat or a dense Array2 depending on maximal incoming degree 
    fn into_matrepr(&self) {

    }

}  // end of impl KGraph<F>

use ndarray::{Array2};
use sprs::{CsMat};

/// We can represent the graph as dense Array2 or a CompressedRow Storage matrice 
enum GraphMatRepr<'a, F> {
    FULL(&'a Array2<F>),
    CSR( &'a CsMat<F>),
}




// =============================================================================



mod tests {




} // end of tests