//! This module implements a randomized truncated svd
//! by multiplication by random orthogonal matrix and Q.R decomposition
//! Halko-Tropp Probabilistic Algorithms For Matrix Decomposition 2011 P 242-245
//! Mahoney Lectures notes on randomized linearAlgebra 2016. P 149-150
//! We use gaussian matrix (instead SRTF as we have a small rank)
//! 

#![allow(dead_code)]


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


// Recall that ndArray is C-order row order.
/// compute an approximate truncated svf
/// The data matrix is supposed given as a (m,n) matrix. n is the number of data and m their dimension.
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


#[inline]
pub fn norm_l2(v : &Array1<f64>) -> f64 {
    v.into_iter().map(|x| x*x).sum::<f64>().sqrt()
}

impl <'a> TruncSvd<'a> {

    pub fn new(data : &'a Array2<f64>, rank:u32) -> Self {
        TruncSvd{data, rank, left_vectors : None , right_vec_t : None , lambdas : None} 


    }

    /// return  y - projection of y on space spanned by y.
    fn orthogonalize_with_Q(Q: &Vec<Array1<f64>>, y: &mut Array1<f64>) {
        let nb_q = Q.len();
        if nb_q == 0 {
            return;
        }
        let size_d = y.len();
        // check dimension coherence between Q and y
        assert_eq!(Q[nb_q - 1].len(),size_d);
        //
        let mut proj_qy = Array1::<f64>::zeros(size_d);
        for i in 0..nb_q {
            proj_qy  += &(Q[i].dot(y) * &Q[i]);
        }
        *y -= &proj_qy;
    }  // end of orthogonalize_with_Q



    /// Adaptive Randomized Range Finder algo 4.2. from Halko-Tropp
    fn adaptative_normal_sampling(&mut self, epsil:f64, rank : usize) {
        let mut rng = RandomGaussianGenerator::new();
        let Q = Vec::<Array1<f64>>::with_capacity(rank);
        let mut Y = Vec::<Array1<f64>>::new();
        // 
        let data_shape = self.data.shape();
        // we store omaga_i vector as row vector as Rust has C order it is easier to extract rows !!
        let omega = rng.generate_matrix(Dim([rank, data_shape[1]]));
        for i in 0..rank {
            Y.push(self.data.dot(&omega.gauss_mat.row(i)));
        }
        // This vectors stores L2-norm of each Y vector
        let mut norms_Y : Array1<f64> = Y.into_iter().map( |y| norm_l2(&y)).collect();        
        //  to get Q as an Array2 : Array2::from_shape_vec((nrows, ncols), data)?;
    //    let y_Qy = orthogonalize_with_Q(&Q, &mut y);

    }
}  // end of impl TruncSvd
