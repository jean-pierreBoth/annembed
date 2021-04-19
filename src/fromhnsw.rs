//! Get a very simple graph from hnsw to be used in kruksal algo and
//! neighborhood entropy computations
//! 
//! 

use num_traits::{Float};

use core::ops::*;  // for  AddAssign + SubAssign + MulAssign + DivAssign + RemAssign 
use std::fmt::*;   // for Display + Debug + LowerExp + UpperExp 

use hnsw_rs::prelude::*;


/// morally F should be f32 and f64
/// The solution from ndArray is F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Display + Debug + LowerExp + UpperExp + (ScalarOperand + LinalgScalar) + Send + Sync 
/// For edge weight we just need  F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Display + Debug + LowerExp + UpperExp + Send + Sync 
pub struct EdgeWeight<F> {
    weight: F
}


/// keep a node index compatible with NdArray
pub type NodeIdx = usize;



/// 
/// A very minimal grpah for this crate (otherwise use petgraph)
/// The graph comes from an k-nn search so we know the number of neighbours we have
/// W is a weight on edges and must satisfy Ord, hence the structure EdgeWeight<F>
/// 
/// The nodes must be indexed from 0 to nbnodes-1 (same as hnsw_rs crate)
/// 
pub(crate) struct KGraph<F> {
    nbnodes: usize,
    ///
    nbng : usize,
    /// an edge is given by 2 nodes and a weight
    edges : Vec<(NodeIdx, NodeIdx, EdgeWeight<F>)>,
    /// neighbours[i] contains the indexes of node i sorted by increasing weight edge!
    neighbours : Vec<Vec<NodeIdx>>,
}

impl <F> KGraph<F> 
    where F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
        Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync 
{

    /// allocates a graph with nbnodes and nbng neighbours 
    pub fn new(nbnodes: usize, nbng : usize) -> Self {
        let mut neighbours_init = Vec::<Vec<NodeIdx>>::with_capacity(nbnodes);
        for i in 0..nbnodes {
            neighbours_init.push(Vec::<NodeIdx>::with_capacity(nbng));
        }
        KGraph {
            nbnodes : nbnodes,
            nbng : nbng,
            edges : Vec::< (NodeIdx, NodeIdx, EdgeWeight<F>) >::with_capacity(nbnodes*nbng),
            neighbours :  neighbours_init,
        }
    }  // end of new

    // insert a list of neighbours for a point and edges update accordingly
    fn insert_neighbourhood(&mut self, node : usize , neighbours : &Vec::<NodeIdx>) {
        // check it the first insertion for this node

        // modify neighbours

        // modify edges

    } // end of insert_neighbourhood

}  // end of impl KGraph<F>



impl <F,D> From<&Hnsw<F,D> > for KGraph<F> 
    where   F : Float + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
                Display + Debug + LowerExp + UpperExp + std::iter::Sum + Send + Sync,
            D : Distance<F> + Send + Sync
    {
    
        fn from(hnsw : &Hnsw<F,D>) -> Self {
            // We must extract from the structure , for each point the list of its nearest neighbours and weight<F> of corresponding edge
            let max_nb_conn = hnsw.get_max_nb_connection();    // morally this the k of knn
            let point_indexation = hnsw.get_point_indexation();
            let nb_point = point_indexation.get_nb_point();
            let mut kgraph = KGraph::<F>::new(nb_point, max_nb_conn as usize);
            let max_nb_conn = hnsw.get_max_nb_connection();    // morally this the k of knn
            let point_indexation = hnsw.get_point_indexation();
            let mut point_iter = point_indexation.into_iter();
            while let Some(point) = point_iter.next() {
                // now point is an Arc<Point<F>>
                // point_id must be in 0..nb_point This is not enforced as in petgraph. We should check that
                let point_id = point.get_origin_id();
                let neighbours = point.get_neighborhood_id();

            }

            kgraph
        }

}