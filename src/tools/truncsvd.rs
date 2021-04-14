//! This module implements a randomized truncated svd
//! par multiplication by random orthogonal matrix and Q.R decomposition
//! Halko-Tropp 2011 P 242-245
//! Mahoney Lectures notes on randomized linearAlgebra 2016. P 149-150
//! We use gaussian matrix (instead SRTF as we have a small rank)


use rand::prelude::*;

use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use ndarray::prelude::*;


// Halko Tropp Algo 4.2 P 243
// We are given a matrix A of dimension (m,n) and we want to get approximate its image at rank r
// We compute a gaussian matrix such that || (I - Q * Qt) * A || < epsil
// - compute P = A * Q multiply right a gaussian (r, n) the matrix data M (m,n)
// - QR decomposition of P


struct RandomGaussianMatrix {
    gauss_mat : Array2::<f64>
}



impl RandomGaussianMatrix {

    /// given dimensions allocate and initialize with random gaussian values matrix
    pub fn new(dims : Ix2) -> Self {
        let mut  gauss_mat = Array2::zeros(dims);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4664397);
        let stdnormal = StandardNormal{};
        for v in gauss_mat.iter_mut() {
            *v = stdnormal.sample(&mut rng);
        }
        RandomGaussianMatrix{gauss_mat}
    }

}  // end of impl block for RandomGaussianMatrix


struct RandomGaussianGenerator {
    rng:Xoshiro256PlusPlus
}



impl RandomGaussianGenerator {
    pub fn new() -> Self {
       let rng = Xoshiro256PlusPlus::seed_from_u64(4664397);
       RandomGaussianGenerator{rng}
    }

    pub fn generate_matrix(&mut self, dims: Ix2) -> RandomGaussianMatrix {
        let mut  gauss_mat = Array2::zeros(dims);
        let stdnormal = StandardNormal{};
        for v in gauss_mat.iter_mut() {
            *v = stdnormal.sample(&mut self.rng);
        }
        RandomGaussianMatrix{gauss_mat}
    }


    // generate a standard N(0,1) vector of N(0,1) of dimension dim
    fn generate_stdn_vect(&mut self, dim: Ix1) -> Array1<f64> {
        let mut  gauss_v = Array1::zeros(dim);
        let stdnormal = StandardNormal{};
        for v in gauss_v.iter_mut() {
            *v = stdnormal.sample(&mut self.rng);
        }
        gauss_v
    }
}  // end of impl RandomGaussianGenerator


/// compute an approximate truncated svf
pub struct TruncSvd<'a> {
    /// matrix we want to approximate range of
    data : &'a Array2<f64>,
    /// asked rank
    rank : u32,
    /// matrix of left eigen vectors
    left_vectors : Option<Array2<f64>>,
    /// transpose matrix of right vectors
    right_vec_t: Option<Array2<f64>>,
    lambdas : Option<Array1<f64>>
} // end of struct TruncSvd 



impl <'a> TruncSvd<'a> {

    pub fn new(data : &'a Array2<f64>, rank:u32) -> Self {
        TruncSvd{data, rank, left_vectors : None , right_vec_t : None , lambdas : None} 


    }
    // algo 42. from Halko-Tropp
    fn adaptative_normal_sampling(&mut self) {

    }
}  // end of impl TruncSvd
