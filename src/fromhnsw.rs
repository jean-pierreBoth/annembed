//! Get a very simple graph from hnsw to be used in kruksal algo and
//! neighborhood entropy computations
//! 
//! 

use num_traits::{Float};

use indexmap::set::*;

use core::ops::*;  // for  AddAssign + SubAssign + MulAssign + DivAssign + RemAssign 
use std::fmt::*;   // for Display + Debug + LowerExp + UpperExp 
use std::result;
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
    nbnodes: usize,
    /// an edge is given by 2 nodes and a weight
    edges : Vec<(NodeIdx, OutEdge<F>)>,
    /// neighbours[i] contains the indexes of neighbours node i sorted by increasing weight edge!
    neighbours : Vec<Vec<OutEdge<F>>>,
    /// to keep track of current node indexes.
    node_set : IndexSet<NodeIdx>,
    /// to keep track of incoming degree of each node.
    // If in_degrees are smalls, this enable all spectral initialization to run in csr mode!
    in_degree : Vec<u32>,
}   // end of struct KGraph





impl <F> KGraph<F> 
    where F : FromPrimitive + Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
        Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync 
{

    /// allocates a graph with nbnodes and nbng neighbours 
    pub fn new(nbng : usize, nbnodes: usize) -> Self {
        let mut neighbours_init = Vec::<Vec<OutEdge<F>>>::with_capacity(nbnodes);
        for _i in 0..nbnodes {
            neighbours_init.push(Vec::<OutEdge<F>>::with_capacity(nbng));
        }
        KGraph {
            nbng : nbng,
            nbnodes : nbnodes,
            edges : Vec::< (NodeIdx, OutEdge<F>) >::with_capacity(nbnodes*nbng),
            neighbours :  neighbours_init,
            node_set : IndexSet::new(),
            in_degree : Vec::<u32>::with_capacity(nbnodes)
        }
    }  // end of new

    // insert a list of neighbours for a point and edges update accordingly
    fn insert_node_neighbourhood(&mut self, node : NodeIdx , neighbours : &Vec::<OutEdge<F>>) -> result::Result<usize, usize> {

        // check it the first insertion for this node
        let (index, already) = self.node_set.insert_full(node);
        if already {
            return Err(index);
        }
        if index >= self.nbnodes {  // check index
            self.neighbours.resize(100, Vec::<OutEdge<F>>::with_capacity(self.nbng));
        }
        if neighbours.len() != self.nbng { // check number of neighbours
            log::error!("neighbours must have {} neighbours", self.nbng);
            return Err(neighbours.len());
        }
        // we can insert neighbours at index and count incoming degrees

        // count incoming degrees
        // 
        Ok(1)
    } // end of insert_neighbourhood


    // edges are supposed directed 
    fn insert_edge_list(edges : &[(NodeIdx, NodeIdx, OutEdge<F>)]) {

        let nb_edge = edges.len();
        for _i in 0..nb_edge {
                // check source and target node


        }


    }  // end of insert_edge_list


    // extract a density index on point based on max distance of k-th Neighbour
    fn get_density_index(&self) -> Vec<F> {
        let mut density = Vec::<F>::with_capacity(self.nbnodes);
        //
        for i in 0..self.neighbours.len() {
            let mut dmax = F::zero();self.neighbours[i][0];
            for j in 0..self.neighbours[i].len() {
                dmax = F::max(dmax, self.neighbours[i][j].weight);
            }
            density[i] = dmax.recip();
        }
        //
        return density;
    } // get_density_index


    // compute incoming_degrees
    fn get_incoming_degrees(&mut self) -> &Vec<u32> {
        for i in 0..self.neighbours.len() {
            for j in 0..self.neighbours[i].len() {
                self.in_degree[self.neighbours[i][j].node] += 1;
            }
        }
        &self.in_degree
    } // end of get_incoming_degrees


    /// initialization for the case we use all points of the hnsw structure
    /// see also *initialize_from_layer* and *initialize_from_descendants*
    pub fn init_from_hnsw_all<D>(hnsw : &Hnsw<F,D>) ->  KGraph<F>
        where   F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
                    Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync,
                D : Distance<F> + Send + Sync {
        //
        // We must extract the whole structure , for each point the list of its nearest neighbours and weight<F> of corresponding edge
        let max_nb_conn = hnsw.get_max_nb_connection() as usize;    // morally this the k of knn bu we have that for each layer
        let point_indexation = hnsw.get_point_indexation();
        let nb_point = point_indexation.get_nb_point();
        let mut kgraph = KGraph::<F>::new(nb_point, max_nb_conn as usize);
        //
        let point_indexation = hnsw.get_point_indexation();
        let mut point_iter = point_indexation.into_iter();
        let mut index = 0;
        while let Some(point) = point_iter.next() {
            // now point is an Arc<Point<F>>
            // point_id must be in 0..nb_point. CAVEAT This is not enforced as in petgraph. We should check that
            let _point_id = point.get_origin_id();
            let neighbours_hnsw = point.get_neighborhood_id();
            // neighbours_hnsw contains neighbours in each layer
            // we flatten the layers and transfer neighbours to KGraph::_neighbours
            // possibly use a BinaryHeap?
            let nb_layer = neighbours_hnsw.len();
            let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn*nb_layer);
            for i in 0..nb_layer {
                neighbours_hnsw[i].iter().map(|n| vec_tmp.push(OutEdge::<F>::from(*n)));
            }
            vec_tmp.sort_unstable_by(| a, b | a.partial_cmp(b).unwrap_or(Ordering::Less));
            assert!(vec_tmp[0].weight < vec_tmp[1].weight);
            // keep only the good size. Could we keep more ?
            vec_tmp.truncate(max_nb_conn);
            let res_insert = kgraph.insert_node_neighbourhood(index, &vec_tmp);
            if !res_insert.is_ok() {
                panic!("insert failed");
            }
            index += 1;
        }

        kgraph
    }   // end init_from_hnsw_all



}  // end of impl KGraph<F>


