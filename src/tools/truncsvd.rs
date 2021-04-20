//! This module implements a randomized truncated svd
//! by multiplication by random orthogonal matrix and Q.R decomposition
//! 
//! Halko-Tropp Probabilistic Algorithms For Matrix Decomposition 2011 P 242-245
//! See also Mahoney Lectures notes on randomized linearAlgebra 2016. P 149-150
//! We use gaussian matrix (instead SRTF as we have a small rank)
//! 

#![allow(dead_code)]



use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use ndarray::prelude::*;

use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst

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
/// The data matrix is supposed given as a (m,n) matrix. m is the number of data and n their dimension.
pub struct RangeApprox<'a> {
    /// matrix we want to approximate range of. We s
    data : &'a Array2<f64>,
    /// asked rank
    rank : u32,
    /// matrix of left eigen vectors
    left_vectors : Option<Array2<f64>>,
    /// transpose matrix of right vectors
    right_vec_t: Option<Array2<f64>>,
    lambdas : Option<Array1<f64>>
} // end of struct RangeApprox 


#[inline]
pub fn norm_l2(v : &ArrayView1<f64>) -> f64 {
    v.into_iter().map(|x| x*x).sum::<f64>().sqrt()
}


/// return  y - projection of y on space spanned by y.
fn orthogonalize_with_q(q: &[Array1<f64>], y: &mut ArrayViewMut1<f64>) {
    let nb_q = q.len();
    if nb_q == 0 {
        return;
    }
    let size_d = y.len();
    // check dimension coherence between Q and y
    assert_eq!(q[nb_q - 1].len(),size_d);
    //
    let mut proj_qy = Array1::<f64>::zeros(size_d);
    for i in 0..nb_q {
        proj_qy  += &(q[i].dot(y) * &q[i]);
    }
    *y -= &proj_qy;
}  // end of orthogonalize_with_Q



impl <'a> RangeApprox<'a> {

    pub fn new(data : &'a Array2<f64>, rank:u32) -> Self {
        RangeApprox{data, rank, left_vectors : None , right_vec_t : None , lambdas : None} 
    }



    // 1. we sample y vectors by batches of size r, 
    // 2. we othogonalize them with vectors in q_mat
    // 3. We normalize the y and add them in q_mat.
    /// Adaptive Randomized Range Finder algo 4.2. from Halko-Tropp
    fn adaptative_normal_sampling(&mut self, epsil:f64, r : usize) -> Array2<f64> {
        let mut rng = RandomGaussianGenerator::new();
        let data_shape = self.data.shape();
        let m = data_shape[0];
        // q_mat and y_mat store vector of interest as rows to take care of Rust order.
        let mut q_mat = Vec::<Array1<f64>>::new();         // q_mat stores vectors of size m
        let stop_val : f64 = epsil/(10. * (2. * f64::FRAC_1_PI()).sqrt());
        // 
        // we store omaga_i vector as row vector as Rust has C order it is easier to extract rows !!
        let omega = rng.generate_matrix(Dim([r, data_shape[1]]));    //  omega is (r, n)
        // so Y is a (data_shape[0], rank) or (m,r) with Tropp notations
        // It will contains the last r vector sampled
        let mut y_mat = self.data.dot(&omega.gauss_mat.t());
        // This vectors stores L2-norm of each Y column vector of which there are r
        let mut norms_y : Array1<f64> = (0..r).into_iter().map( |i| norm_l2(&y_mat.column(i))).collect();
        assert_eq!(norms_y.len() , r); 
        let mut norm_sup_y;
        norm_sup_y = norms_y.iter().max_by(|x,y| x.partial_cmp(y).unwrap()).unwrap();
        let mut j = 0;
        while norm_sup_y > &stop_val {
            // numerical stabilization
            if q_mat.len() > 1 {
                orthogonalize_with_q(&q_mat[0..q_mat.len()-1], &mut y_mat.row_mut(j));
            }
            let y_j = y_mat.row(j);
            let n_j =  norm_l2(&y_j);
            let q_j = &y_j / n_j;
            // we add q_j to q_mat so we consumed on y vector
            q_mat.push(q_j.clone());
            // we make another y, first we sample a new omega_j vector of size n
            let omega_j_p1 = rng.generate_stdn_vect(Ix1(data_shape[1]));
            let mut y_j_p1 = self.data.dot(&omega_j_p1);    // y_j_p1 is of size m 
            // we orthogonalize new y with all q's i.e q_mat
            orthogonalize_with_q(&q_mat, &mut y_j_p1.view_mut());
            // the new y will takes the place of old y at rank j%r so we always have the last r y that have been sampled
            for k in 0..y_j_p1.len() {
                y_mat.column_mut(j)[k] = y_j_p1[k];
            }
            norms_y[j] = norm_l2(&y_mat.column(j));
            // we orthogonalize old y's with new q_j
            for j in 0..r {
                let y_j = &mut y_mat.row_mut(j);
                let prodq_y = q_j.view().dot(y_j) * &q_j;
                *y_j -= &prodq_y;
            }
            // we update j
            j = (j+1)%r;
            // we update norm_sup_y
            norm_sup_y = norms_y.iter().max_by(|x,y| x.partial_cmp(y).unwrap()).unwrap();
        }
        //  to get Q as an Array2 : Array2::from_shape_vec((nrows, ncols), data)?;
        // https://docs.rs/ndarray/0.15.1/ndarray/struct.ArrayBase.html#method.view_mut
        //

        let mut q_as_array2  = Array2::zeros((q_mat.len(), m));
        for i in 0..q_mat.len() {
            for j in 0..m {
                q_as_array2[[j,i]] = q_mat[i][j];
            }
        }
        // we return an array2 where each row is a data of reduced dimension
       q_as_array2
    } // end of adaptative_normal_sampling
}  // end of impl RangeApprox



mod tests {

use super::*;

    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }  

#[test]
    fn test_arrayview_mut() {
        let mut array = array![[1, 2], [3, 4]];
        let to_add =  array![1, 1];
        let mut row_mut = array.row_mut(0);
        row_mut += &to_add;
        assert_eq!(array[[0,0]], 2);
        assert_eq!(array[[0,1]], 3);
    }

}