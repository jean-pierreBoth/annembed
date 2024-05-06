//! single linkage hdbscan
//!
//!
//!
//! implements single linkage clustering on top of Kruskal algorithm.
//!
//!

#![allow(unused)]

use num_traits::cast::FromPrimitive;
use num_traits::int::PrimInt;

use num_traits::float::*; // tp get FRAC_1_PI from FloatConst

use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::collections::BinaryHeap;

use hnsw_rs::prelude::*;

use super::kruskal::*;
use crate::fromhnsw::kgraph::KGraph;
use crate::fromhnsw::kgraph_from_hnsw_all;

// 1.  We get from the hnsw a list of edges for kruskal algorithm
// 2.  Run kruskal algorithm ,  we get nodes of edge, weigth of edge and parent of nodes
//        - so we get at each step the id of cluster representative that unionized.
//
// 3. Fill a Dendrogram structure

/// This structure represent a merge step
///
pub struct UnionStep<NodeIdx: PrimInt, F: Float> {
    /// node a of edge removed
    nodea: NodeIdx,
    /// node b of edge removed
    nodeb: NodeIdx,
    /// weight of edge
    weight: F,
    /// step at which the merge occurs
    step: usize,
    /// representative of nodea in the union-find
    clusta: NodeIdx,
    /// representative of nodeb in the union-find
    clustb: NodeIdx,
} // end of struct UnionStep

/// Some basic statistics on Clusters
pub struct ClusterStat {
    /// mean of density around each point
    /// For hnsw , we compute mean dist to k-neighbours for each point and compute mean on cluster.
    mean_density: f32,
    /// nuber of terms in Cluster
    size: u32,
}

pub struct Dendrogram<NodeIdx: PrimInt, F: Float> {
    steps: Vec<UnionStep<NodeIdx, F>>,
}

impl<NodeIdx: PrimInt, F: Float> Dendrogram<NodeIdx, F> {
    pub fn new(nbstep: usize) -> Self {
        Dendrogram {
            steps: Vec::<UnionStep<NodeIdx, F>>::with_capacity(nbstep),
        }
    }
} // end of impl Dendrogram

/// edge to be stored in a binary heap for Dendogram formation
struct Edge<F: Float + PartialOrd> {
    nodea: u32,
    nodeb: u32,
    weight: F,
}

// We can do that beccause we cannot have NaN coming from Hnsw
fn compare_edge<F: Float + PartialOrd>(edgea: &Edge<F>, edgeb: &Edge<F>) -> Ordering {
    match (edgea.weight, edgeb.weight) {
        (x, y) if x.is_nan() && y.is_nan() => Ordering::Equal,
        (x, _) if x.is_nan() => Ordering::Greater,
        (_, y) if y.is_nan() => Ordering::Less,
        (_, _) => edgea.weight.partial_cmp(&edgeb.weight).unwrap(),
    }
}
// We need to implement an Ord for edge based on a float representation of Edge weight

impl<F: Float + PartialOrd> PartialEq for Edge<F> {
    fn eq(&self, other: &Self) -> bool {
        self.nodea == other.nodea && self.nodeb == other.nodeb
    }
}

impl<F: Float + PartialOrd> Eq for Edge<F> {}

impl<F: Float + PartialOrd> PartialOrd for Edge<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.weight.partial_cmp(&other.weight)
    }
}

impl<F: Float + PartialOrd> Ord for Edge<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_edge(&self, &other)
    }
}

/// The structure driving Single Linkage Clustering
/// It is constructed from a Hnsw
pub struct SLclustering<NodeIdx: PrimInt, F: Float> {
    // the kgraph summary provided by hnsw
    kgraph: KGraph<F>,
    //
    dendrogram: Dendrogram<NodeIdx, F>,
    // ask for at most nbcluster. We can stop if we get in nbcluster union steps
    nbcluster: usize,
} // end of  SLclustering

impl<'a, NodeIdx: PrimInt, F> SLclustering<NodeIdx, F>
where
    F: PartialOrd
        + FromPrimitive
        + Float
        + Send
        + Sync
        + Clone
        + std::fmt::UpperExp
        + std::iter::Sum,
{
    //
    pub fn new<D>(hnsw: &Hnsw<F, D>, nbcluster: usize) -> Self
    where
        D: Distance<F> + Send + Sync,
    {
        //
        // get kgraph summary
        //
        let nbng = hnsw.get_max_nb_connection() as usize;
        let kgraph = kgraph_from_hnsw_all(hnsw, nbng).unwrap();
        //
        let nbstep = kgraph.get_nb_nodes() - nbcluster;
        SLclustering {
            kgraph,
            dendrogram: Dendrogram::<NodeIdx, F>::new(nbstep),
            nbcluster,
        }
    } // end of new

    /// computes clustering
    pub fn cluster(&mut self) {
        let _kgraph_stats = self.kgraph.get_kraph_stats();
        //
        // get a list of (node, node, weight of edge), compute mst
        let neighboourhood_info = self.kgraph.get_neighbours();
        let nbnodes = neighboourhood_info.len();
        let max_nbng = self.kgraph.get_max_nbng();
        let mut edge_list = Vec::<(u32, u32, F)>::with_capacity(max_nbng * nbnodes);
        for i in 0..nbnodes {
            for edge in &neighboourhood_info[i] {
                edge_list.push((i as u32, edge.node as u32, edge.weight));
            }
        }
        let mst_edge_iter = kruskal(&edge_list);
        // now we transfer edges in a binary_heap
        let mut edge_heap = BinaryHeap::<Edge<F>>::with_capacity(edge_list.len());
        for edge in mst_edge_iter {
            edge_heap.push(Edge {
                nodea: edge.0,
                nodeb: edge.1,
                weight: edge.2,
            });
        }
        // have an iterator of edge traversing tree , in increasing order

        // we initialize clusters with singletons

        // we run unification (possibly with density filter)
    } // end of cluster
} // end of impl for Hclust
