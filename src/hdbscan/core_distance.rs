//! Core distance computation for HDBSCAN
//! 
//! Core distance is the distance to the k-th nearest neighbor (where k = min_samples).
//! This is used to compute the mutual reachability distance which forms the basis
//! of the HDBSCAN algorithm.

use num_traits::float::Float;
use std::fmt::Debug;

/// Stores core distance information for a point
#[derive(Debug, Clone)]
pub struct CoreDistance<F: Float> {
    /// The point's identifier
    pub point_id: usize,
    /// Distance to min_samples-th nearest neighbor
    pub core_dist: F,
    /// K nearest neighbors with distances (sorted by distance)
    pub neighbors: Vec<(usize, F)>,
}

impl<F: Float> CoreDistance<F> {
    /// Create a new CoreDistance from a point's neighbors
    /// 
    /// # Arguments
    /// * `point_id` - The identifier of the point
    /// * `neighbors` - Vector of (neighbor_id, distance) pairs, assumed to be sorted by distance
    /// * `min_samples` - The minimum number of samples to form a core point
    /// 
    /// # Returns
    /// A CoreDistance struct with the computed core distance
    pub fn new(point_id: usize, neighbors: Vec<(usize, F)>, min_samples: usize) -> Self {
        // Core distance is the distance to the (min_samples-1)th neighbor
        // We use min_samples-1 because the point itself is counted as one sample
        let core_dist = if min_samples <= 1 {
            // Degenerate case: min_samples = 1 means only the point itself
            F::zero()
        } else if neighbors.len() >= min_samples - 1 {
            // Standard case: we have enough neighbors
            neighbors[min_samples - 2].1
        } else if !neighbors.is_empty() {
            // Edge case: fewer neighbors than min_samples, use the furthest neighbor
            neighbors.last().unwrap().1
        } else {
            // No neighbors at all - point is isolated
            F::infinity()
        };
        
        CoreDistance {
            point_id,
            core_dist,
            neighbors,
        }
    }
    
    /// Get the core distance for this point
    pub fn get_core_distance(&self) -> F {
        self.core_dist
    }
    
    /// Check if this point is a core point (has finite core distance)
    pub fn is_core_point(&self) -> bool {
        self.core_dist.is_finite()
    }
    
    /// Get the number of neighbors within core distance
    pub fn num_neighbors_in_core(&self) -> usize {
        self.neighbors
            .iter()
            .take_while(|(_, dist)| *dist <= self.core_dist)
            .count()
    }
}

/// Compute mutual reachability distance between two points
/// 
/// The mutual reachability distance is defined as:
/// MRD(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
/// 
/// This transformation ensures that dense points remain close to each other
/// while sparse points are pushed away, which helps HDBSCAN identify
/// clusters of varying densities.
/// 
/// # Arguments
/// * `point_a` - Index of the first point
/// * `point_b` - Index of the second point  
/// * `distance` - The actual distance between points a and b
/// * `core_distances` - Vector of CoreDistance structs for all points
/// 
/// # Returns
/// The mutual reachability distance between the two points
pub fn mutual_reachability_distance<F: Float>(
    point_a: usize,
    point_b: usize,
    distance: F,
    core_distances: &[CoreDistance<F>],
) -> F {
    let core_a = core_distances[point_a].core_dist;
    let core_b = core_distances[point_b].core_dist;
    
    // MRD is the maximum of the two core distances and the actual distance
    core_a.max(core_b).max(distance)
}

/// Compute mutual reachability distance using only core distance values
/// 
/// This is a convenience function when you have core distances but not
/// the full CoreDistance structs.
#[allow(dead_code)]
pub fn mutual_reachability_distance_from_core<F: Float>(
    core_dist_a: F,
    core_dist_b: F,
    distance: F,
) -> F {
    core_dist_a.max(core_dist_b).max(distance)
}

/// Batch compute all core distances for a dataset
/// 
/// This function computes core distances for all points in parallel when possible.
/// It assumes the neighbor information is already available (e.g., from a KGraph).
#[allow(dead_code)]
pub fn compute_all_core_distances<F: Float>(
    neighbors: &[Vec<(usize, F)>],
    min_samples: usize,
) -> Vec<CoreDistance<F>> {
    neighbors
        .iter()
        .enumerate()
        .map(|(point_id, point_neighbors)| {
            CoreDistance::new(point_id, point_neighbors.clone(), min_samples)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_core_distance_computation() {
        // Test with enough neighbors
        let neighbors = vec![
            (1, 1.0_f32),
            (2, 2.0_f32),
            (3, 3.0_f32),
            (4, 4.0_f32),
            (5, 5.0_f32),
        ];
        
        let core_dist = CoreDistance::new(0, neighbors.clone(), 3);
        assert_eq!(core_dist.get_core_distance(), 2.0_f32);
        assert!(core_dist.is_core_point());
        
        // Test with fewer neighbors than min_samples
        let few_neighbors = vec![(1, 1.5_f32), (2, 2.5_f32)];
        let core_dist_few = CoreDistance::new(0, few_neighbors, 5);
        assert_eq!(core_dist_few.get_core_distance(), 2.5_f32);
        
        // Test with no neighbors
        let no_neighbors: Vec<(usize, f32)> = vec![];
        let core_dist_none = CoreDistance::new(0, no_neighbors, 3);
        assert!(core_dist_none.get_core_distance().is_infinite());
        assert!(!core_dist_none.is_core_point());
    }
    
    #[test]
    fn test_mutual_reachability_distance() {
        let core_distances = vec![
            CoreDistance::new(0, vec![(1, 1.0_f32), (2, 2.0_f32)], 3),
            CoreDistance::new(1, vec![(0, 1.0_f32), (2, 1.5_f32)], 3),
        ];
        
        // Test MRD computation
        // Point 0 has core distance 2.0, Point 1 has core distance 1.5
        // Distance between them is 1.0
        // MRD should be max(2.0, 1.5, 1.0) = 2.0
        let mrd = mutual_reachability_distance(0, 1, 1.0_f32, &core_distances);
        assert_eq!(mrd, 2.0_f32);
        
        // Test when actual distance is largest
        let mrd_large = mutual_reachability_distance(0, 1, 3.0_f32, &core_distances);
        assert_eq!(mrd_large, 3.0_f32);
    }
    
    #[test]
    fn test_mutual_reachability_from_core() {
        let mrd = mutual_reachability_distance_from_core(2.0_f32, 1.5_f32, 1.0_f32);
        assert_eq!(mrd, 2.0_f32);
        
        let mrd_actual = mutual_reachability_distance_from_core(1.0_f32, 0.5_f32, 3.0_f32);
        assert_eq!(mrd_actual, 3.0_f32);
    }
    
    #[test]
    fn test_batch_core_distance() {
        let all_neighbors = vec![
            vec![(1, 1.0_f32), (2, 2.0_f32), (3, 3.0_f32)],
            vec![(0, 1.0_f32), (2, 1.5_f32), (3, 2.5_f32)],
            vec![(0, 2.0_f32), (1, 1.5_f32), (3, 1.0_f32)],
        ];
        
        let core_distances = compute_all_core_distances(&all_neighbors, 3);
        
        assert_eq!(core_distances.len(), 3);
        assert_eq!(core_distances[0].get_core_distance(), 2.0_f32);
        assert_eq!(core_distances[1].get_core_distance(), 1.5_f32);
        assert_eq!(core_distances[2].get_core_distance(), 1.5_f32);
    }
}