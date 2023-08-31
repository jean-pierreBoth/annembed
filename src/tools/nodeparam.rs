//! Describes an outgoing edge from a node, 
//! the neighbourhood of a node as list of edges from a node (struct NodeParam)
//! and structure NodeParams which gather the description of all nodes
//! 
//! 
//! 

use serde::{Serialize, Deserialize};

use num_traits::Float;
use num_traits::cast::FromPrimitive;

use std::cmp::Ordering;

use hnsw_rs::hnsw::Neighbour;

/// keep a node index compatible with NdArray
pub type NodeIdx = usize;

/// an outEdge gives the destination node and weight of edge.
#[derive(Clone,Copy,Debug, Serialize, Deserialize)]
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
}   // end impl From<Neighbour>


// We need this structure to compute entropy od neighbour distribution
/// This structure stores gathers parameters of a node:
///  - its local scale
///  - list of edges. The f32 field constains distance (increasing order) or directed (decreasing) proba of edge going out of each node
///    (distance and proba) to its nearest neighbours as referenced in field neighbours of KGraph.
///
/// Identity of neighbour node must be fetched in KGraph structure to spare memory
#[derive(Clone)]
pub struct NodeParam {
    pub(crate) scale: f32,
    pub(crate) edges: Vec<OutEdge<f32>>,
}

impl NodeParam {
    pub fn new(scale: f32, edges: Vec<OutEdge<f32>>) -> Self {
        NodeParam { scale, edges }
    }

    /// for a given node index return corresponding edge if it is in neighbours, None else 
    pub fn get_edge(&self, i : NodeIdx) -> Option<&OutEdge<f32>> {
        self.edges.iter().find( |&&edge| edge.node == i)
    }  // end of is_around

    /// perplexity. Hill number cf Leinster
    pub fn get_perplexity(&self) -> f32 {
        let h : f32 = self.edges.iter().map(|&x| -x.weight * x.weight.ln()).sum();
        h.exp()
    }

    /// get number of out edges
#[allow(unused)]
    pub fn  get_nb_edges(&self) -> usize {
        self.edges.len()
    }
} // end of NodeParam


impl Default for NodeParam {
    fn default() -> Self { 
        return NodeParam {scale : 0f32 , edges : Vec::<OutEdge<f32>>::new() };
    }
}
//=================================================================================================================


/// We maintain NodeParam for each node as it enables scaling in the embedded space and cross entropy minimization.
pub struct NodeParams {
    pub params: Vec<NodeParam>,
    pub max_nbng : usize,
}

impl NodeParams {
    pub fn new(params :Vec<NodeParam>, max_nbng : usize) -> Self {
        NodeParams{params, max_nbng}
    }
    //
    pub fn get_node_param(&self, node: NodeIdx) -> &NodeParam {
        return &self.params[node];
    }

    pub fn get_nb_nodes(&self) -> usize {
        self.params.len()
    }

    pub fn get_max_nbng(&self) -> usize {
        self.max_nbng
    }
} // end of NodeParams

