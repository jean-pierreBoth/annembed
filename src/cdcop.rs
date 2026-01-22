//! This module implements Carré Du champ operator computations.
//! It builds upon DiffusionMaps tools.
//!
//! Bibliography
//! - *Diffusion Maps*. Coifman Lafon Appl. Comput. Harmon. Anal. 21 (2006) 5–30
//! - *Diffusion Geometry*. Ioo Jones 2024 https://arxiv.org/abs/2405.10858
//! - *Bamberger.J Jones.I Carre du Champ 2025. https://arxiv.org/abs/2510.05930
//!

#![allow(unused)]

use ndarray::Array2;
use num_traits::Float;
use num_traits::cast::FromPrimitive;

use hnsw_rs::prelude::*;

use crate::diffmaps::*;
use crate::fromhnsw::{kgraph::KGraph, kgraph_from_hnsw_all};
use crate::graphlaplace::*;

/// The structure computes the transition kernel to neighbours using DiffusionMaps with adhoc parameters.
/// Then it computes the Covariance of the transition kernel at each asked for point.
/// This provides the best local normal approximation of the data and gives information on the geometry of the data
///   
pub struct CarreDuChamp {
    dparams: DiffusionParams,
    //
    glaplacian: Option<GraphLaplacian>,
}

fn graph_laplacian_from_hnsw<T, D, F>(hnsw: &Hnsw<T, D>) -> GraphLaplacian
where
    T: Send + Sync + Clone,
    D: Distance<T> + Send + Sync,
    F: Float
        + FromPrimitive
        + std::marker::Sync
        + Send
        + std::fmt::UpperExp
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Into<f64>,
{
    let mut dparams = DiffusionParams::build_with_variable_bandwidth();
    dparams.set_alfa(0.);
    dparams.set_beta(0.);
    //
    let mut dmap = DiffusionMaps::new(dparams);
    dmap.laplacian_from_hnsw::<T, D, F>(hnsw, &dparams)
}

impl CarreDuChamp {
    pub fn new(dparams: &DiffusionParams) -> Self {
        CarreDuChamp {
            dparams: dparams.clone(),
            glaplacian: None,
        }
    }

    pub fn from_hnsw<T, D, F>(hnsw: &Hnsw<T, D>) -> CarreDuChamp
    where
        T: Send + Sync + Clone,
        D: Distance<T> + Send + Sync,
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::ops::DivAssign
            + Into<f64>,
    {
        let mut dparams = DiffusionParams::build_with_variable_bandwidth();

        let mut dparams = DiffusionParams::build_with_variable_bandwidth();
        dparams.set_alfa(0.);
        dparams.set_beta(0.);
        //
        let mut dmap = DiffusionMaps::new(dparams);
        let laplacian = dmap.laplacian_from_hnsw::<T, D, F>(hnsw, &dparams);
        let cdc = CarreDuChamp {
            dparams: dparams.clone(),
            glaplacian: Some(laplacian),
        };
        //
        cdc
    }
    //

    /// compute carre du champ at point given its rank
    pub fn get_cdc_at_point<F>(point_idx: usize) -> Array2<F>
    where
        F: Float
            + FromPrimitive
            + std::marker::Sync
            + Send
            + std::fmt::UpperExp
            + std::iter::Sum
            + std::ops::AddAssign
            + std::ops::DivAssign
            + Into<f64>,
    {
        panic!("not yet implemented")
    }
}
