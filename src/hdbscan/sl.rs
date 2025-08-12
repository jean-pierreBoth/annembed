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
use std::collections::{BinaryHeap, HashSet};

use hnsw_rs::prelude::*;

use super::core_distance::{CoreDistance, mutual_reachability_distance};
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
        Some(self.cmp(other))
    }
}

impl<F: Float + PartialOrd> Ord for Edge<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_edge(self, other)
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

impl<NodeIdx: PrimInt, F> SLclustering<NodeIdx, F>
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

    /// Compute core distances for all points using the k-nearest neighbor graph
    pub fn compute_core_distances(&self, min_samples: usize) -> Vec<CoreDistance<F>> {
        let neighbours = self.kgraph.get_neighbours();
        let mut core_distances = Vec::with_capacity(neighbours.len());
        
        for (point_id, edges) in neighbours.iter().enumerate() {
            // Convert edges to neighbor format
            let neighbors: Vec<(usize, F)> = edges
                .iter()
                .map(|e| (e.node, e.weight))
                .collect();
            
            core_distances.push(CoreDistance::new(point_id, neighbors, min_samples));
        }
        
        core_distances
    }
    
    /// Build minimum spanning tree using mutual reachability distances
    /// Returns a vector of edges (a, b, mrd_weight) sorted by weight
    pub fn build_mrd_mst(&self, min_samples: usize) -> Vec<(usize, usize, F)> {
        let core_distances = self.compute_core_distances(min_samples);
        let neighbours = self.kgraph.get_neighbours();
        
        // Build edge list with mutual reachability distances
        let mut mrd_edges = Vec::new();
        let mut seen_edges = std::collections::HashSet::new();
        
        for (i, edges) in neighbours.iter().enumerate() {
            for edge in edges {
                let j = edge.node;
                // Only add each edge once (use canonical ordering)
                let edge_key = if i < j { (i, j) } else { (j, i) };
                
                if seen_edges.insert(edge_key) {
                    let mrd = mutual_reachability_distance(
                        i,
                        j,
                        edge.weight,
                        &core_distances,
                    );
                    mrd_edges.push((i as u32, j as u32, mrd));
                }
            }
        }
        
        // Run Kruskal's algorithm on the MRD edges
        let mst: Vec<(usize, usize, F)> = kruskal(&mrd_edges)
            .map(|(a, b, w)| (a as usize, b as usize, w))
            .collect();
        
        log::info!("Built MST with {} edges using MRD", mst.len());
        
        mst
    }
    
    /// Build minimum spanning tree using standard distances (original method)
    pub fn build_standard_mst(&self) -> Vec<(usize, usize, F)> {
        let neighbours = self.kgraph.get_neighbours();
        let nbnodes = neighbours.len();
        let max_nbng = self.kgraph.get_max_nbng();
        
        let mut edge_list = Vec::<(u32, u32, F)>::with_capacity(max_nbng * nbnodes);
        let mut seen_edges = std::collections::HashSet::new();
        
        for (n, edge_vec) in neighbours.iter().enumerate() {
            for edge in edge_vec.iter() {
                // Only add each edge once
                let edge_key = if n < edge.node { (n, edge.node) } else { (edge.node, n) };
                if seen_edges.insert(edge_key) {
                    edge_list.push((n as u32, edge.node as u32, edge.weight));
                }
            }
        }
        
        kruskal(&edge_list)
            .map(|(a, b, w)| (a as usize, b as usize, w))
            .collect()
    }
    
    /// computes clustering
    pub fn cluster(&mut self) {
        let _kgraph_stats = self.kgraph.get_kraph_stats();
        //
        // get a list of (node, node, weight of edge), compute mst
        let neighboourhood_info = self.kgraph.get_neighbours();
        let nbnodes = neighboourhood_info.len();
        let max_nbng = self.kgraph.get_max_nbng();
        let mut edge_list = Vec::<(u32, u32, F)>::with_capacity(max_nbng * nbnodes);
        for (n, edge_vec) in neighboourhood_info.iter().enumerate() {
            for edge in edge_vec.iter() {
                edge_list.push((n as u32, edge.node as u32, edge.weight));
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

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Helper function to create a simple HNSW index from data
    fn build_hnsw(data: &[Vec<f32>]) -> Hnsw<f32, DistL2> {
        let nb_data = data.len();
        let data_with_id: Vec<(&Vec<f32>, usize)> = 
            data.iter().zip(0..nb_data).collect();
        
        let max_nb_conn = 8.max(nb_data / 2);
        let ef_c = 200;
        let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize).max(1);
        
        let mut hnsw = Hnsw::<f32, DistL2>::new(
            max_nb_conn, 
            nb_data, 
            nb_layer, 
            ef_c, 
            DistL2 {}
        );
        // Use serial insert for deterministic tests
        for (data, id) in data_with_id {
            hnsw.insert((data, id));
        }
        hnsw
    }
    
    #[test]
    fn test_core_distance_computation() {
        // Create a simple dataset
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let core_distances = clustering.compute_core_distances(3);
        
        // All points should have finite core distances
        for cd in &core_distances {
            assert!(cd.get_core_distance().is_finite(), 
                    "Point {} should have finite core distance", cd.point_id);
            assert!(cd.is_core_point());
        }
    }
    
    #[test]
    fn test_mrd_mst_construction() {
        // Two well-separated clusters
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            // Cluster 2  
            vec![5.0, 0.0],
            vec![5.1, 0.0],
            vec![5.0, 0.1],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        
        // Build both MSTs
        let standard_mst = clustering.build_standard_mst();
        let mrd_mst = clustering.build_mrd_mst(3);
        
        // Both should have n-1 edges
        assert_eq!(standard_mst.len(), data.len() - 1);
        assert_eq!(mrd_mst.len(), data.len() - 1);
        
        // Find the edge with maximum weight (should be between clusters)
        let max_edge = mrd_mst.iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();
        
        // The heaviest edge should connect the two clusters
        let (a, b, weight) = max_edge;
        let cluster_a = if *a < 3 { 0 } else { 1 };
        let cluster_b = if *b < 3 { 0 } else { 1 };
        
        assert_ne!(cluster_a, cluster_b,
                   "Heaviest edge ({},{}) with weight {} should be between clusters",
                   a, b, weight);
    }
    
    #[test]
    fn test_mrd_properties() {
        // Test that MRD satisfies its mathematical properties
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let core_distances = clustering.compute_core_distances(2);
        
        // MRD should be symmetric
        let dist_01 = 1.0_f32;
        let mrd_01 = mutual_reachability_distance(0, 1, dist_01, &core_distances);
        let mrd_10 = mutual_reachability_distance(1, 0, dist_01, &core_distances);
        assert!((mrd_01 - mrd_10).abs() < 1e-6, "MRD should be symmetric");
        
        // MRD >= actual distance
        assert!(mrd_01 >= dist_01 - 1e-6, "MRD should be >= actual distance");
        
        // MRD >= core distances
        assert!(mrd_01 >= core_distances[0].get_core_distance() - 1e-6);
        assert!(mrd_01 >= core_distances[1].get_core_distance() - 1e-6);
    }
    
    #[test]
    fn test_outlier_detection() {
        // Cluster with one outlier
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![10.0, 10.0], // Outlier
        ];
        
        let hnsw = build_hnsw(&data);
        let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
        let core_distances = clustering.compute_core_distances(3);
        
        // Outlier should have much larger core distance
        let outlier_core = core_distances[4].get_core_distance();
        let cluster_core_max = core_distances[..4].iter()
            .map(|cd| cd.get_core_distance())
            .fold(0.0_f32, |a, b| a.max(b));
        
        assert!(outlier_core > cluster_core_max * 10.0,
                "Outlier core distance {} should be much larger than cluster max {}",
                outlier_core, cluster_core_max);
    }
}
