//! This module implements a randomized truncated svd
//! by multiplication by random orthogonal matrix and Q.R decomposition
//! 
//! We are given a matrix A of dimension (m,n) and we want a matrix Q of size(m, rank)
//! We compute a gaussian matrix such that || (I - Q * Qt) * A || < epsil

//! Halko-Tropp Probabilistic Algorithms For Approximate Matrix Decomposition 2011 P 242-245
//! See also Mahoney Lectures notes on randomized linearAlgebra 2016. P 149-150
//! We use gaussian matrix (instead SRTF as in the Ann context we have a small rank)
//! 

#![allow(dead_code)]



use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use ndarray::prelude::*;

use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst

// Halko Tropp Algo Probabilistic Algorithms for Approximate Matrix decomposition Algo 4.2 P 243
// We are given a matrix A of dimension (m,n) and we want to get approximate its image at rank r
// We compute a gaussian matrix such that || (I - Q * Qt) * A || < epsil
// - compute P = A * Q multiply right a gaussian (r, n) the matrix data M (m,n)
// - QR decomposition of P


struct RandomGaussianMatrix {
    mat : Array2::<f64>
}



impl RandomGaussianMatrix {

    /// given dimensions allocate and initialize with random gaussian values matrix
    pub fn new(dims : Ix2) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4664397);
        let stdnormal = StandardNormal{};
        let mat : Array2::<f64> = ArrayBase::from_shape_fn(dims, |_| {
            stdnormal.sample(&mut rng)
        });
        //
        RandomGaussianMatrix{mat}
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
        RandomGaussianMatrix::new(dims)
    }


    // generate a standard N(0,1) vector of N(0,1) of dimension dim
    fn generate_stdn_vect(&mut self, dim: Ix1) -> Array1<f64> {
        let stdnormal = StandardNormal{};
        let gauss_v : Array1<f64> =  ArrayBase::from_shape_fn(dim, |_| {
            stdnormal.sample(&mut self.rng)
        });
        gauss_v
    }
}  // end of impl RandomGaussianGenerator


// Recall that ndArray is C-order row order.
/// compute an approximate truncated svf
/// The data matrix is supposed given as a (m,n) matrix. m is the number of data and n their dimension.
pub struct RangeApprox<'a> {
    /// matrix we want to approximate range of. We s
    data : &'a Array2<f64>,
} // end of struct RangeApprox 


/// compute L2-norm of an array
#[inline]
pub fn norm_l2<D:Dimension>(v : &ArrayView<f64, D>) -> f64 {
    v.into_iter().map(|x| x*x).sum::<f64>().sqrt()
}


/// return  y - projection of y on space spanned by q's vectors.
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

    pub fn new(data : &'a Array2<f64>) -> Self {
        RangeApprox{data} 
    }



    // 1. we sample y vectors by batches of size r, 
    // 2. we othogonalize them with vectors in q_mat
    // 3. We normalize the y and add them in q_mat.
    // 4. The loop (on j)
    //      - take one y (at rank j), normalize it , push it in q_vec
    //      - generate a new y to replace the one pushed into q
    //      - orthogonalize new y with vectors in q
    //      - orthogonalize new q with all y vectors except the new one (the one at j+r)
    //             (so that when y will pushed into q it is already orthogonal to preceedind q)
    //      - check for norm sup of y
    //
    /// Returns a matrix Q such that || data - Q*t(Q)*data || < epsil
    /// Adaptive Randomized Range Finder algo 4.2. from Halko-Tropp
    fn adaptative_randomized_range_finder(&self, epsil:f64, r : usize) -> Array2<f64> {
        let mut rng = RandomGaussianGenerator::new();
        let data_shape = self.data.shape();
        let m = data_shape[0];
        // q_mat and y_mat store vector of interest as rows to take care of Rust order.
        let mut q_mat = Vec::<Array1<f64>>::new();         // q_mat stores vectors of size m
        let stop_val : f64 = epsil/(10. * (2. * f64::FRAC_1_PI()).sqrt());
        // 
        // we store omaga_i vector as row vector as Rust has C order it is easier to extract rows !!
        let omega = rng.generate_matrix(Dim([data_shape[1], r]));    //  omega is (n, r)
        // We could store Y = data * omega as matrix (m,r), but as we use Y column,
        // we store Y (as Q) as a Vec of Array1<f64>
        let mut y_vec =  Vec::<Array1<f64>>::with_capacity(r);
        for j in 0..r {
            y_vec.push(self.data.dot(&omega.mat.column(j)));
        }
        // This vectors stores L2-norm of each Y  vector of which there are r
        let mut norms_y : Array1<f64> = (0..r).into_iter().map( |i| norm_l2(&y_vec[i].view())).collect();
        assert_eq!(norms_y.len() , r); 
        //
        let mut norm_sup_y;
        norm_sup_y = norms_y.iter().max_by(|x,y| x.partial_cmp(y).unwrap()).unwrap();
        log::debug!(" norm_sup {} ",norm_sup_y);
        let mut j = 0;
        let mut nb_iter = 0;
        let max_iter = data_shape[0].min(data_shape[1]);
        //
        while norm_sup_y > &stop_val && nb_iter <= max_iter {
            // numerical stabilization
            if q_mat.len() > 0 {
                orthogonalize_with_q(&q_mat[0..q_mat.len()], &mut y_vec[j].view_mut());
            }
            // get norm of current y vector
            let n_j =  norm_l2(&y_vec[j].view());
            if n_j < f64::EPSILON {
                log::debug!("exiting at nb_iter {} with n_j {:.3e} ", nb_iter, n_j);
                break;
            }
            println!("j {} n_j {:.3e} ", j, n_j);
            log::debug!("j {} n_j {:.3e} ", j, n_j);
            let q_j = &y_vec[j] / n_j;
            // we add q_j to q_mat so we consumed on y vector
            q_mat.push(q_j.clone());
            // we make another y, first we sample a new omega_j vector of size n
            let omega_j_p1 = rng.generate_stdn_vect(Ix1(data_shape[1]));
            let mut y_j_p1 = self.data.dot(&omega_j_p1);    // y_j_p1 is of size m 
            // we orthogonalize new y with all q's i.e q_mat
            orthogonalize_with_q(&q_mat, &mut y_j_p1.view_mut());
            // the new y will takes the place of old y at rank j%r so we always have the last r y that have been sampled
            y_j_p1.assign_to(y_vec[j].view_mut());
            // we orthogonalize old y's with new q_j. Can be made //
            for k in 0..r {
                if k != j {
                    // avoid k = j as the j vector is the new one
                    let prodq_y = q_j.view().dot(&y_vec[k]) * &q_j;
                    y_vec[k] -= &prodq_y;
                }
            }
            // we update norm_sup_y
            norms_y[j] = norm_l2(&y_vec[j].view());
            norm_sup_y = norms_y.iter().max_by(|x,y| x.partial_cmp(y).unwrap()).unwrap();
            log::debug!(" j {} norm_sup {:.3e} ", j, norm_sup_y);
            // we update j and nb_iter
            j = (j+1)%r;
            nb_iter += 1;
        }
        //
        // to avoid the cost to zeros
        log::debug!("range finder returning a a matrix ({}, {})", m, q_mat.len());
        let mut q_as_array2  = Array2::uninit((m, q_mat.len()));
        for i in 0..q_mat.len() {
            for j in 0..m {
                q_as_array2[[j,i]] = std::mem::MaybeUninit::new(q_mat[i][j]);
            }
        }
        // we return an array2 where each row is a data of reduced dimension
        unsafe{ q_as_array2.assume_init()}
    } // end of adaptative_normal_sampling
}  // end of impl RangeApprox


fn check_range_finder(a_mat : &ArrayView2<f64>, q_mat: &ArrayView2<f64>) -> f64{
    let residue = a_mat - q_mat.dot(&q_mat.t()).dot(a_mat);
    let norm_residue = norm_l2(&residue.view());
    norm_residue
}

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
    } // end of test_arrayview_mut

    #[test]
    fn test_range_approx_1() {
        log_init_test();
        //
        let data = RandomGaussianGenerator::new().generate_matrix(Dim([6,50]));
        let range_approx = RangeApprox::new(&data.mat);
        let q = range_approx.adaptative_randomized_range_finder(0.05, 5);
        let residue = check_range_finder(&data.mat.view(), &q.view());
        log::debug!(" residue {:3.e} ", residue);
    } // end of test_range_approx_1

    #[test]
    fn test_range_approx_2() {
        log_init_test();
        //
        let data = RandomGaussianGenerator::new().generate_matrix(Dim([50,500]));
        let range_approx = RangeApprox::new(&data.mat);
        let q = range_approx.adaptative_randomized_range_finder(0.05, 5);
        let residue = check_range_finder(&data.mat.view(), &q.view());
        log::debug!(" residue {:3.e} ", residue);
    } // end of test_range_approx_1


}  // end of module test