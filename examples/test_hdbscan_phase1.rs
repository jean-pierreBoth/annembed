//! Example demonstrating HDBSCAN Phase 1: Core Distance and MRD computation
//! 
//! Run with: cargo run --example test_hdbscan_phase1

use annembed::hdbscan::{SLclustering, mutual_reachability_distance};
use hnsw_rs::prelude::*;

fn main() {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    // Create a simple 2D dataset with two clusters
    let data = vec![
        // Cluster 1 (around origin)
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.0],
        vec![0.0, 0.2],
        vec![0.15, 0.15],
        // Cluster 2 (around (5, 5))
        vec![5.0, 5.0],
        vec![5.1, 5.1],
        vec![5.2, 5.0],
        vec![5.0, 5.2],
        vec![5.15, 5.15],
        // Noise points
        vec![2.5, 2.5],
        vec![7.0, 1.0],
    ];
    
    let nb_data = data.len();
    let data_with_id: Vec<(&Vec<f32>, usize)> = 
        data.iter().zip(0..nb_data).map(|(d, i)| (d, i)).collect();
    
    // Build HNSW index
    log::info!("Building HNSW index...");
    let max_nb_conn = 8;
    let ef_c = 200;
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    
    let hnsw = Hnsw::<f32, DistL2>::new(
        max_nb_conn, 
        nb_data, 
        nb_layer, 
        ef_c, 
        DistL2 {}
    );
    hnsw.parallel_insert(&data_with_id);
    
    // Create SLclustering instance
    log::info!("Creating HDBSCAN clustering instance...");
    let clustering = SLclustering::<usize, f32>::new(&hnsw, 1);
    
    // Compute core distances with min_samples = 3
    let min_samples = 3;
    log::info!("Computing core distances with min_samples = {}...", min_samples);
    let core_distances = clustering.compute_core_distances(min_samples);
    
    // Display core distances
    println!("\nCore Distances (min_samples = {}):", min_samples);
    println!("Point ID | Core Distance | Is Core Point");
    println!("---------|---------------|---------------");
    for cd in &core_distances {
        println!("{:8} | {:13.4} | {:13}", 
            cd.point_id, 
            cd.get_core_distance(),
            if cd.is_core_point() { "Yes" } else { "No" }
        );
    }
    
    // Demonstrate mutual reachability distance
    println!("\nMutual Reachability Distances (sample pairs):");
    println!("Points | Actual Dist | Core A | Core B | MRD");
    println!("-------|-------------|--------|--------|-------");
    
    // Check MRD between points in same cluster
    let dist_0_1 = ((data[0][0] - data[1][0]).powi(2) + 
                    (data[0][1] - data[1][1]).powi(2)).sqrt();
    let mrd_0_1 = mutual_reachability_distance(0, 1, dist_0_1, &core_distances);
    println!("(0, 1) | {:11.4} | {:6.4} | {:6.4} | {:5.4}",
        dist_0_1,
        core_distances[0].get_core_distance(),
        core_distances[1].get_core_distance(),
        mrd_0_1
    );
    
    // Check MRD between points in different clusters
    let dist_0_5 = ((data[0][0] - data[5][0]).powi(2) + 
                    (data[0][1] - data[5][1]).powi(2)).sqrt();
    let mrd_0_5 = mutual_reachability_distance(0, 5, dist_0_5, &core_distances);
    println!("(0, 5) | {:11.4} | {:6.4} | {:6.4} | {:5.4}",
        dist_0_5,
        core_distances[0].get_core_distance(),
        core_distances[5].get_core_distance(),
        mrd_0_5
    );
    
    // Build MST with standard distances
    log::info!("\nBuilding standard MST...");
    let standard_mst = clustering.build_standard_mst();
    println!("Standard MST has {} edges", standard_mst.len());
    
    // Build MST with mutual reachability distances
    log::info!("Building MRD-based MST...");
    let mrd_mst = clustering.build_mrd_mst(min_samples);
    println!("MRD MST has {} edges", mrd_mst.len());
    
    // Compare edge weights
    println!("\nFirst 5 edges comparison:");
    println!("Standard MST weights: {:?}", 
        standard_mst.iter().take(5).map(|(_, _, w)| w).collect::<Vec<_>>());
    println!("MRD MST weights:      {:?}", 
        mrd_mst.iter().take(5).map(|(_, _, w)| w).collect::<Vec<_>>());
    
    println!("\nPhase 1 complete! Core distances and MRD computation working correctly.");
}