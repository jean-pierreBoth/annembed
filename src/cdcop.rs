//! This module implements Carré Du champ operator computations.
//! It builds upon DiffusionMaps tools.
//!
//! Bibliography
//! - *Diffusion Maps*. Coifman Lafon Appl. Comput. Harmon. Anal. 21 (2006) 5–30
//! - *Diffusion Geometry*. Ioo Jones 2024 https://arxiv.org/abs/2405.10858
//! - *Bamberger.J Jones.I Carre du Champ 2025. https://arxiv.org/abs/2510.05930
//!

#![allow(unused)]

use indexmap::IndexSet;
use ndarray::{Array1, Array2, ArrayView};
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
    // to keep track of rank DataId conversion
    index: Option<IndexSet<DataId>>,
    // We need coordinates to compute cdc
    data: Array2<f32>,
}

fn graph_laplacian_from_hnsw<T, D>(hnsw: &Hnsw<T, D>) -> GraphLaplacian
where
    T: Send + Sync + Clone + Float + FromPrimitive + Into<f32>,
    D: Distance<T> + Send + Sync,
{
    let mut dparams = DiffusionParams::build_with_variable_bandwidth();
    dparams.set_alfa(0.);
    dparams.set_beta(0.);
    //
    let mut dmap = DiffusionMaps::new(dparams);
    dmap.laplacian_from_hnsw::<T, D, f32>(hnsw, &dparams)
}

impl CarreDuChamp {
    /// Construct the CarreDuChamp, consuming the Hnsw structure, just keep the data point
    pub fn from_hnsw<T, D>(hnsw: Hnsw<T, D>) -> CarreDuChamp
    where
        T: Copy + Clone + Send + Sync + Into<f32>,
        D: Distance<T> + Send + Sync,
    {
        Self::from_hnsw_ref(&hnsw)
    }

    /// Construct the CarreDuChamp, consuming the Hnsw structure
    pub fn from_hnsw_ref<T, D>(hnsw: &Hnsw<T, D>) -> CarreDuChamp
    where
        T: Copy + Clone + Send + Sync + Into<f32>,
        D: Distance<T> + Send + Sync,
    {
        let mut dparams = DiffusionParams::build_with_variable_bandwidth();
        dparams.set_alfa(0.);
        dparams.set_beta(0.);
        //
        // We need to collect point coordintates. (Cf Kgraph construction)
        // TODO: do we drop hnsw after that
        //

        let point_indexation = hnsw.get_point_indexation();
        let nb_point = point_indexation.get_nb_point();
        let mut index_set = IndexSet::<DataId>::with_capacity(nb_point);
        let dimension = point_indexation.get_data_dimension();
        let mut data = Array2::<f32>::zeros((nb_point, dimension));
        //
        let point_iter = point_indexation.into_iter();
        for point in point_iter {
            let point_id = point.get_origin_id();
            // remap _point_id
            let (index, _) = index_set.insert_full(point_id);
            let mut coord = ndarray::ArrayView1::from(point.get_v());
            for i in 0..dimension {
                data.index_axis_mut(ndarray::Axis(0), index)[i] = coord[i].into();
            }
        }
        //
        let mut dmap = DiffusionMaps::new(dparams);
        let laplacian = dmap.laplacian_from_hnsw::<T, D, f32>(&hnsw, &dparams);
        let cdc = CarreDuChamp {
            dparams: dparams.clone(),
            glaplacian: Some(laplacian),
            index: Some(index_set),
            data,
        };
        //
        cdc
    }
    //

    /// compute carre du champ at point given its rank
    pub fn get_cdc_at_point(&self, point_idx: usize) -> Array2<f32> {
        //
        let glaplacian = self.glaplacian.as_ref().unwrap();
        // retrieve point
        let point_in = self.data.row(point_idx);
        // compute mean
        let mean = glaplacian.apply_kernel(&point_in);
        // check distance between point_in and point_out

        // compute covariance along data dimension
        let dim = self.data.shape()[1];
        let mut cov = Array2::<f32>::zeros((dim, dim));
        // get list of index conscerned by row point_idx
        let neighbours = glaplacian.get_kernel_row_csvec(point_idx);
        for (n, proba) in neighbours.iter() {
            for i in 0..dim {
                for j in 0..=i {
                    cov[[i, j]] +=
                        proba * (self.data[[n, i]] - mean[i]) * (self.data[[n, j]] - mean[j]);
                }
            }
        }
        // compute trace, eigenvalue and possible renormalization
        panic!("not yet implemented")
    }
}
