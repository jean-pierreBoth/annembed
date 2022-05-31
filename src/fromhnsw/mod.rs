//! This file gathers everything coming from hnsw_rs.
//! It covers grapk coming from hnsw (kgraph)[kgraph::KGraph] and projection on a small KGaph see (kgraphprojection)[kgproj::KGraphProjection]

pub mod kgraph;

pub use kgraph::kgraph_from_hnsw_all;

pub mod kgproj;