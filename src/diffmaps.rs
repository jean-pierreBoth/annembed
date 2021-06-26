//!  diffusion maps embedding



use num_traits::{Float};
use num_traits::cast::FromPrimitive;

use ndarray::{Array2};
use ndarray_linalg::{Scalar};


use crate::nodeparam::*;
use crate::graphlaplace::*;


#[derive(Copy,Clone)]
pub struct DiffusionParams {
    asked_dim : usize,
    /// embedding time 
    t : Option<f32>,
} // end of DiffusionParams


impl DiffusionParams {
    pub fn new(asked_dim : usize, t_opt : Option<f32>) -> Self {
        DiffusionParams{asked_dim, t : t_opt}
    }
    //
    pub fn get_t(&self) ->  Option<f32> {
        self.t
    }

} // end of DiffusionParams


pub struct DiffusionMaps {
    params : DiffusionParams,
    /// node parameters coming from graph transformation
    _node_params: NodeParams,

} // end of DiffusionMaps


impl DiffusionMaps {
    /// do the whole worrk chain : hnsw construction, graph conversion, NodePArams transformation
    pub fn embed<F>(&mut self, data : &Array2<F>) -> Array2<F>  where
        F : Float + FromPrimitive {
    //
        let (nb_data, _) = data.dim();
        let embedded = Array2::<F>::zeros((nb_data, self.params.asked_dim));
        embedded        
    }

}  // end of impl DiffusionsMaps



// this function initialize and returns embedding by a svd (or else?)
// We are intersested in first eigenvalues (excpeting 1.) of transition probability matrix
// i.e last non null eigenvalues of laplacian matrix!!
// The time used is the one in argument in t_opt if not None.
// If t_opt is none the time is compute so that $ (\lambda_{2}/\lambda_{1})^t \less 0.9 $
pub(crate) fn get_dmap_embedding<F>(initial_space : &NodeParams, asked_dim: usize, t_opt : Option<f32>) -> Array2<F> 
    where F :  Float + FromPrimitive {
    //
    assert!(asked_dim >= 2);
    // get eigen values of normalized symetric lapalcian
    let mut laplacian = get_laplacian(initial_space);
    //
    log::debug!("got laplacian, going to svd ... asked_dim :  {}", asked_dim);
    let svd_res = laplacian.do_svd(asked_dim+25).unwrap();
    // As we used a laplacian and probability transitions we eigenvectors corresponding to lower eigenvalues
    let lambdas = svd_res.get_sigma().as_ref().unwrap();
    // singular vectors are stored in decrasing order according to lapack for both gesdd and gesvd. 
    if lambdas.len() > 2 && lambdas[1] > lambdas[0] {
        panic!("svd spectrum not decreasing");
    }
    // we examine spectrum
    // our laplacian is without the term I-G , we use directly G symetrized so we consider upper eigenvalues
    log::info!(" first 3 eigen values {:.2e} {:.2e} {:2e}",lambdas[0], lambdas[1] , lambdas[2]);
    // get info on spectral gap
    log::info!(" last eigenvalue computed rank {} value {:.2e}", lambdas.len()-1, lambdas[lambdas.len()-1]);
    //
    log::debug!("keeping columns from 1 to : {}", asked_dim);
    // We get U at index in range first_non_zero-max_dim..first_non_zero
    let u = svd_res.get_u().as_ref().unwrap();
    log::debug!("u shape : nrows: {} ,  ncols : {} ", u.nrows(), u.ncols());
    // we can get svd from approx range so that nrows and ncols can be number of nodes!
    let mut embedded = Array2::<F>::zeros((u.nrows(), asked_dim));
    // according to theory (See Luxburg or Lafon-Keller diffusion maps) we must go back to eigen vectors of rw laplacian.
    // Appendix A of Coifman-Lafon Diffusion Maps. Applied Comput Harmonical Analysis 2006.
    // moreover we must get back to type F
    let normalized_lambdas = lambdas/(*lambdas)[0];
    let time = match t_opt {
        Some(t) => t,
            _   => 5.0f32.min(0.9f32.ln()/ (normalized_lambdas[2]/normalized_lambdas[1]).ln()),
    };
    log::info!("get_dmap_initial_embedding applying dmap time {:.2e}", time);
    let sum_diag = laplacian.degrees.into_iter().sum::<f32>(); 
    for i in 0..u.nrows() {
        let row_i = u.row(i);
        let weight_i = (laplacian.degrees[i]/sum_diag).sqrt();
        for j in 0..asked_dim {
            // divide j value by diagonal and convert to F. TODO could take l_{i}^{t} as in dmap?
            embedded[[i, j]] = F::from_f32(normalized_lambdas[j+1].pow(time) * row_i[j+1] / weight_i).unwrap();
        }
    }
    log::trace!("ended get_dmap_initial_embedding");
    return embedded;
} // end of get_dmap_initial_embedding


