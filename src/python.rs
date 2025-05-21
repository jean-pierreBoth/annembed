//! PyO3 bindings for the `annembed` crate.
//!
//! • `embed`       – HNSW-initialised gradient-descent embedding  
//! • `dmap_embed` – Diffusion-Maps embedding
//!
//! Both functions return a NumPy `ndarray` (shape =`(n_samples, dim)`).

use std::path::Path;

use crate::diffmaps::{DiffusionMaps, DiffusionParams};
use crate::fromhnsw::kgraph::{kgraph_from_hnsw_all, KGraph};
use crate::fromhnsw::kgproj::KGraphProjection;
use crate::prelude::{
    get_toembed_from_csv, write_csv_array2, Embedder, EmbedderParams,
};
use cpu_time::ProcessTime;
use hnsw_rs::prelude::*;
use ndarray::Array2;
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{Bound, types::PyModule};
// ──────────────────────────────────────────────────────────────────────────
// Helper types
// ──────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct HnswArgs {
    distance: String,
    max_conn: usize,
    ef: usize,
    knbn: usize,
    scale_mod: f64,
}

impl HnswArgs {
    fn validate(&self) -> Result<(), &'static str> {
        match self.distance.as_str() {
            "DistL1" | "DistL2" | "DistCosine" | "DistJeffreys" | "DistJensenShannon" => Ok(()),
            _ => Err("distance must be DistL1 | DistL2 | DistCosine | DistJeffreys | DistJensenShannon"),
        }
    }
}

macro_rules! with_dist {
    ($name:expr, $alias:ident, $body:block) => {{
        match $name.as_str() {
            "DistL1" => { type $alias = DistL1; $body }
            "DistCosine" => { type $alias = DistCosine; $body }
            "DistJeffreys" => { type $alias = DistJeffreys; $body }
            "DistJensenShannon" => { type $alias = DistJensenShannon; $body }
            _ => { type $alias = DistL2; $body }
        }
    }};
}

fn build_kgraph(
    pairs: &[(&Vec<f64>, usize)],
    hp: &HnswArgs,
    nb_layer: usize,
) -> KGraph<f64> {
    with_dist!(hp.distance, D, {
        let n = pairs.len();
        let mut h = Hnsw::<f64, D>::new(hp.max_conn, n, nb_layer, hp.ef, D::default());
        h.modify_level_scale(hp.scale_mod);
        h.parallel_insert(pairs);
        kgraph_from_hnsw_all(&h, hp.knbn).expect("k-graph construction failed")
    })
}

fn build_kgraph_proj(
    pairs: &[(&Vec<f64>, usize)],
    hp: &HnswArgs,
    nb_layer: usize,
    proj_layer: usize,
) -> KGraphProjection<f64> {
    with_dist!(hp.distance, D, {
        let n = pairs.len();
        let mut h = Hnsw::<f64, D>::new(hp.max_conn, n, nb_layer, hp.ef, D::default());
        h.modify_level_scale(hp.scale_mod);
        h.parallel_insert(pairs);
        KGraphProjection::<f64>::new(&h, hp.knbn, proj_layer)
    })
}

// ──────────────────────────────────────────────────────────────────────────
// embed  – gradient-descent
// ──────────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    csvfile,
    *,
    outfile = None,
    delim = ",",
    dim = 2,
    batch = 20,
    nbsample = 10,
    hierarchy = 0,
    scale = 1.0,
    quality_sampling = None,
    distance = "DistL2",
    nbconn = 64,
    ef = 512,
    knbn = 10,
    scale_modification = 1.0
))]
fn embed(
    py: Python,
    csvfile: &str,
    outfile: Option<&str>,
    delim: &str,
    dim: usize,
    batch: usize,
    nbsample: usize,
    hierarchy: usize,
    scale: f64,
    quality_sampling: Option<f64>,
    distance: &str,
    nbconn: usize,
    ef: usize,
    knbn: usize,
    scale_modification: f64,
) -> PyResult<Py<PyAny>> {
    if delim.chars().count() != 1 {
        return Err(PyValueError::new_err("`delim` must be a single character"));
    }

    let hp = HnswArgs { distance: distance.into(),
                        max_conn: nbconn, ef, knbn,
                        scale_mod: scale_modification };
    hp.validate().map_err(PyValueError::new_err)?;

    // 1. Load data
    let data = get_toembed_from_csv::<f64>(
        Path::new(csvfile),
        delim.as_bytes()[0],
        quality_sampling.unwrap_or(1.0),
    ).map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
    let pairs: Vec<_> = data.iter().zip(0..data.len()).collect();
    let nb_layer = 16.min((data.len() as f32).ln().trunc() as usize);

    // 2. Embedding parameters
    let mut ep = EmbedderParams::default();
    ep.asked_dim = dim;
    ep.nb_grad_batch = batch;
    ep.nb_sampling_by_edge = nbsample;
    ep.hierarchy_layer = hierarchy;
    ep.scale_rho = scale;

    // 3. Run the embedder
    let cpu = ProcessTime::now();
    let embedded: Array2<f64> = if hierarchy == 0 {
        let kg = build_kgraph(&pairs, &hp, nb_layer);
        let mut e = Embedder::new(&kg, ep);
        e.embed().map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        e.get_embedded_reindexed()
    } else {
        let pr = build_kgraph_proj(&pairs, &hp, nb_layer, hierarchy);
        let mut e = Embedder::from_hkgraph(&pr, ep);
        e.embed().map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        e.get_embedded_reindexed()
    };
    eprintln!("embed cpu time {:?}", cpu.elapsed());

    // 4. Optional CSV dump
    if let Some(p) = outfile {
        let mut w = csv::Writer::from_path(p)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        write_csv_array2(&mut w, &embedded).unwrap();
        w.flush().unwrap();
    }

    // 5. Return to Python
    Ok(embedded.view().to_pyarray(py).unbind().into())    // → Py<PyAny>
}

// ──────────────────────────────────────────────────────────────────────────
// dmap_embed  – Diffusion-Maps
// ──────────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    csvfile,
    *,
    outfile = None,
    delim = ",",
    dim = 2,
    alfa = 1.0,
    beta = 0.0,
    time = 5.0,
    hierarchy = 0,
    quality_sampling = None,
    distance = "DistL2",
    nbconn = 64,
    ef = 512,
    knbn = 10,
    scale_modification = 1.0
))]
fn dmap_embed(
    py: Python,
    csvfile: &str,
    outfile: Option<&str>,
    delim: &str,
    dim: usize,
    alfa: f32,
    beta: f32,
    time: f32,
    hierarchy: usize,
    quality_sampling: Option<f64>,
    distance: &str,
    nbconn: usize,
    ef: usize,
    knbn: usize,
    scale_modification: f64,
) -> PyResult<Py<PyAny>> {
    if delim.chars().count() != 1 {
        return Err(PyValueError::new_err("`delim` must be a single character"));
    }

    let hp = HnswArgs { distance: distance.into(),
                        max_conn: nbconn, ef, knbn,
                        scale_mod: scale_modification };
    hp.validate().map_err(PyValueError::new_err)?;

    // 1. Load data
    let data = get_toembed_from_csv::<f64>(
        Path::new(csvfile),
        delim.as_bytes()[0],
        quality_sampling.unwrap_or(1.0),
    ).map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
    let pairs: Vec<_> = data.iter().zip(0..data.len()).collect();
    let nb_layer = 16.min((data.len() as f32).ln().trunc() as usize);

    // 2. Diffusion-maps parameters
    let mut dp = DiffusionParams::new(dim, Some(time), None);
    dp.set_alfa(alfa);
    dp.set_beta(beta);
    dp.set_hlayer(hierarchy);

    // 3. Compute embedding
    let cpu = ProcessTime::now();
    let embedded: Array2<f64> = if hierarchy == 0 {
        let kg = build_kgraph(&pairs, &hp, nb_layer);
        DiffusionMaps::new(dp)
            .embed_from_kgraph(&kg, &dp)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?
    } else {
        let pr = build_kgraph_proj(&pairs, &hp, nb_layer, hierarchy);
        DiffusionMaps::new(dp)
            .embed_from_kgraph(pr.get_small_graph(), &dp)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?
    };
    eprintln!("dmap_embed cpu time {:?}", cpu.elapsed());

    // 4. Optional CSV dump
    if let Some(p) = outfile {
        let mut w = csv::Writer::from_path(p)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        write_csv_array2(&mut w, &embedded).unwrap();
        w.flush().unwrap();
    }

    // 5. Return to Python
    Ok(embedded.view().to_pyarray(py).unbind().into())
}

// ──────────────────────────────────────────────────────────────────────────
// Python module definition
// ──────────────────────────────────────────────────────────────────────────

#[pymodule]
fn annembed(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
   m.add_function(wrap_pyfunction!(embed, m)?)?;
   m.add_function(wrap_pyfunction!(dmap_embed, m)?)?;
   m.add("__all__", vec!["embed", "dmap_embed"])?;
   // keep reference to numpy so the module stays alive
   let _ = numpy::get_array_module(py)?;
   Ok(())
}