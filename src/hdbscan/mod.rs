//! module hdbscan
//! 
//! The implementation here is a Single Linkage algorithm. 
//! It is a fast algorithm based on an a k-nn algorithm and minimum spanning tree.
//! The choice of the Single Linkage Algorithm is justified by the Carlsson, Memoli paper
//! cited below.
//! 
//! We build upon the crates hnsw-rs for the k-nn implementation and petgraph 
//! for the the minimum spanning tree (Kruskal) algorithm. 
//! 
//! 
//! The following papers give some light on the theoretical problems underlying the hierarchical clustering algorithms:
//! 
//! - Density-Based Clustering Based on Hierarchical Density Estimates.
//!   Campello Moulavi Sander (2013)
//! 
//! - Characterization, Stability and Convergence of Hierarchical Clustering Methods
//!  Carlsson, Memoli. 2010
//!
//! - Consistent procedures for cluster tree estimation 
//!     Chaudhuri K., DasGupta S., Kpotufe S., Von Luxburg 2014
//! 
//! - Hierarchical Clustering: Objective Functions and Algorithms
//!     Cohen-Addad V., Kanade V., Mathieu C. (2017) 
//!  
//! 


mod kruskal;
mod sl;

// 1.  We get from the hnsw a list of edges for kruskal algorithm
// 2.  Run kruskal algorithm ,  we get a MinSpanningTree<G>
//        - iteration through the MinSpanningTree<G> gives give edges in increasing weight.
//        - construct den
// 3. Get an ordered list of edges.



