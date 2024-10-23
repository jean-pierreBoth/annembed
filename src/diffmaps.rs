//!  (Kind of) Diffusion maps embedding.
//!
//! This module (presently) computes a diffusion embedding for the kernel constructed from nearest neighbours
//! stored in a Hnsw structure, see in module [embedder](crate::embedder).  
//! In particular the kernel sets the diagonal to 0 and nearest neighbour weight to 1.
//!
//!

use num_traits::cast::FromPrimitive;
use num_traits::Float;

use hnsw_rs::prelude::*;
use ndarray::Array2;
// use ndarray_linalg::Scalar;

use crate::embedder::*;
use crate::fromhnsw::*;
use crate::graphlaplace::*;
use crate::tools::nodeparam::*;

#[derive(Copy, Clone)]
pub struct DiffusionParams {
    /// dimension of embedding
    asked_dim: usize,
    /// embedding time
    t: Option<f32>,
} // end of DiffusionParams

impl DiffusionParams {
    pub fn new(asked_dim: usize, t_opt: Option<f32>) -> Self {
        DiffusionParams {
            asked_dim,
            t: t_opt,
        }
    }
    /// get embedding time
    pub fn get_t(&self) -> Option<f32> {
        self.t
    }
    //

    pub fn get_embedding_dimension(&self) -> usize {
        self.asked_dim
    }
} // end of DiffusionParams

pub struct DiffusionMaps {
    /// parameters to use
    params: DiffusionParams,
    /// node parameters coming from graph transformation
    _node_params: Option<NodeParams>,
} // end of DiffusionMaps

impl DiffusionMaps {
    /// iitialization from NodeParams
    pub fn new(params: DiffusionParams) -> Self {
        DiffusionMaps {
            params,
            _node_params: None,
        }
    }

    /// do the whole work chain : hnsw construction, graph conversion, NodeParams transformation
    /// T is the type on which distances in Hnsw are computed,  
    /// F is f32 or f64 depending on how diffusions Maps is to be computed.
    pub fn embed_hnsw<T, D, F>(&mut self, hnsw: &Hnsw<T, D>) -> Array2<F>
    where
        D: Distance<T> + Send + Sync,
        T: Clone + Send + Sync,
        F: Float + FromPrimitive + std::marker::Sync + Send + std::fmt::UpperExp + std::iter::Sum,
    {
        //
        let knbn = hnsw.get_max_nb_connection();
        let kgraph = kgraph_from_hnsw_all::<T, D, F>(hnsw, knbn as usize).unwrap();
        // get NodeParams. CAVEAT to_proba_edges apply initial shift!!
        let nodeparams = to_proba_edges::<F>(&kgraph, 1., 2.);
        get_dmap_embedding::<F>(&nodeparams, self.params.asked_dim, self.params.get_t())
    }
} // end of impl DiffusionsMaps

// this function initialize and returns embedding by a svd (or else?)
// We are intersested in first eigenvalues (excpeting 1.) of transition probability matrix
// i.e last non null eigenvalues of laplacian matrix!!
// The time used is the one in argument in t_opt if not None.
// If t_opt is none the time is compute so that $ (\lambda_{2}/\lambda_{1})^t \less 0.9 $
pub(crate) fn get_dmap_embedding<F>(
    initial_space: &NodeParams,
    asked_dim: usize,
    t_opt: Option<f32>,
) -> Array2<F>
where
    F: Float + FromPrimitive,
{
    //
    assert!(asked_dim >= 2);
    // get eigen values of normalized symetric lapalcian
    let mut laplacian = get_laplacian(initial_space);
    //
    log::debug!("got laplacian, going to svd ... asked_dim :  {}", asked_dim);
    let svd_res = laplacian.do_svd(asked_dim + 25).unwrap();
    // As we used a laplacian and probability transitions we eigenvectors corresponding to lower eigenvalues
    let lambdas = svd_res.get_sigma().as_ref().unwrap();
    // singular vectors are stored in decrasing order according to lapack for both gesdd and gesvd.
    if lambdas.len() > 2 && lambdas[1] > lambdas[0] {
        panic!("svd spectrum not decreasing");
    }
    // we examine spectrum
    // our laplacian is without the term I of I-G , we use directly G symetrized so we consider upper eigenvalues
    log::info!(
        " first 3 eigen values {:.2e} {:.2e} {:2e}",
        lambdas[0],
        lambdas[1],
        lambdas[2]
    );
    // get info on spectral gap
    log::info!(
        " last eigenvalue computed rank {} value {:.2e}",
        lambdas.len() - 1,
        lambdas[lambdas.len() - 1]
    );
    //
    log::debug!("keeping columns from 1 to : {}", asked_dim);
    // We get U at index in range first_non_zero-max_dim..first_non_zero
    let u = svd_res.get_u().as_ref().unwrap();
    log::debug!("u shape : nrows: {} ,  ncols : {} ", u.nrows(), u.ncols());
    if u.ncols() < asked_dim {
        log::warn!(
            "asked dimension  : {} svd obtained less than asked for : {}",
            asked_dim,
            u.ncols()
        );
    }
    let real_dim = asked_dim.min(u.ncols());
    // we can get svd from approx range so that nrows and ncols can be number of nodes!
    let mut embedded = Array2::<F>::zeros((u.nrows(), real_dim));
    // according to theory (See Luxburg or Lafon-Keller diffusion maps) we must go back to eigen vectors of rw laplacian.
    // Appendix A of Coifman-Lafon Diffusion Maps. Applied Comput Harmonical Analysis 2006.
    // moreover we must get back to type F
    let normalized_lambdas = lambdas / (*lambdas)[0];
    let time = match t_opt {
        Some(t) => t,
        _ => 5.0f32.min(0.9f32.ln() / (normalized_lambdas[2] / normalized_lambdas[1]).ln()),
    };
    log::info!("get_dmap_initial_embedding applying dmap time {:.2e}", time);
    let sum_diag = laplacian.degrees.iter().sum::<f32>();
    for i in 0..u.nrows() {
        let row_i = u.row(i);
        let weight_i = (laplacian.degrees[i] / sum_diag).sqrt();
        for j in 0..real_dim {
            // divide j value by diagonal and convert to F. take l_{i}^{t} as in dmap
            embedded[[i, j]] =
                F::from_f32(normalized_lambdas[j + 1].powf(time) * row_i[j + 1] / weight_i)
                    .unwrap();
        }
    }
    log::debug!("ended get_dmap_initial_embedding");
    embedded
} // end of get_dmap_initial_embedding

//======================================================================================================================

/// This function runs a parallel insertion of rows of an `Array2<T>` into a  Hnsw<T,D>.  
/// The hnsw structure must have chosen main parameters as the number of connection and layers, but
/// be empty.   
/// Returns number of point inserted if success.
pub fn array2_insert_hnsw<T, D>(data: &Array2<T>, hnsw: &mut Hnsw<T, D>) -> Result<usize, usize>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    //
    if hnsw.get_nb_point() > 0 {
        log::error!(
            "array2_insert_hnsw , insertion on non empty hnsw structure, nb point : {}",
            hnsw.get_nb_point()
        );
        return Err(1);
    }
    // we do parallel insertion by blocks of size blocksize
    let blocksize = 10000;
    let (nb_row, _) = data.dim();

    let nb_block = nb_row / blocksize;
    for i in 0..nb_block {
        let start = i * blocksize;
        let end = i * blocksize + blocksize - 1;
        let to_insert = (start..=end)
            .map(|n| (data.row(n).to_slice().unwrap(), n))
            .collect();
        hnsw.parallel_insert_slice(&to_insert);
    }
    let start = nb_block * blocksize;
    let to_insert = (start..nb_row)
        .map(|n| (data.row(n).to_slice().unwrap(), n))
        .collect();
    hnsw.parallel_insert_slice(&to_insert);
    //
    Ok(hnsw.get_nb_point())
} // end of array2_insert_hnsw

//=======================================================================

mod tests {} // end of mod tests
