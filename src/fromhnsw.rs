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
        Display + Debug + LowerExp + UpperExp + Send + Sync 
{

    /// allocates a graph with nbnodes and nbng neighbours 
    pub fn new(nbnodes: usize, nbng : usize) -> Self {
        KGraph {
            nbnodes : nbnodes,
            nbng : nbng,
            edges : Vec::< (NodeIdx, NodeIdx, EdgeWeight<F>) >::with_capacity(nbnodes*nbng),
            neighbours :  Vec::<Vec<NodeIdx>>::new(),
        }
    }  // end of new
}  // end of impl KGraph<F>