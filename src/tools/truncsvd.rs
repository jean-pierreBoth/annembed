//! This module implements a randomized truncated svd
//! by multiplication by random orthogonal matrix and Q.R decomposition
//! 
//! We are given a matrix A of dimension (m,n) and we want a matrix Q of size(m, rank)
//! We compute a gaussian matrix such that || (I - Q * Qt) * A || < epsil

//! Halko-Tropp Probabilistic Algorithms For Approximate Matrix Decomposition 2011 P 242-245
//! See also Mahoney Lectures notes on randomized linearAlgebra 2016. P 149-150
//! We use gaussian matrix (instead SRTF as in the Ann context we have a small rank)
//! 
//! the type F must verify F : Float + FromPrimitive + Scalar + ndarray::ScalarOperand
//! so it is f32 or f64

// Float provides PartialOrd which is not in Scalar.
// ndarray::ScalarOperand provides array * F
// ndarray_linalg::Scalar provides Exp notation + Display + Debug + Serialize and sum on iterators


#![allow(dead_code)]



use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

//use ndarray::prelude::*;

use ndarray::{Dim, Array1, Array2, ArrayBase, Dimension, ArrayView, ArrayViewMut1, ArrayView2 , Ix1, Ix2};

use ndarray_linalg::{Scalar, Lapack};
// use ndarray_linalg::{UVTFlag,svddc};

use lax::{layout::MatrixLayout, UVTFlag};

use std::marker::PhantomData;

use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst
use num_traits::cast::FromPrimitive;



struct RandomGaussianMatrix<F:Float> {
    mat : Array2::<F>
}



impl <F> RandomGaussianMatrix<F> where F:Float+FromPrimitive {

    /// given dimensions allocate and initialize with random gaussian values matrix
    pub fn new(dims : Ix2) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4664397);
        let stdnormal = StandardNormal{};
        let mat : Array2::<F> = ArrayBase::from_shape_fn(dims, |_| {
            F::from_f64(stdnormal.sample(&mut rng)).unwrap()
        });
        //
        RandomGaussianMatrix{mat}
    }

}  // end of impl block for RandomGaussianMatrix


struct RandomGaussianGenerator<F> {
    rng:Xoshiro256PlusPlus,
    _ty : std::marker::PhantomData<F>
}



impl <F:Float+FromPrimitive> RandomGaussianGenerator<F> {
    pub fn new() -> Self {
       let rng = Xoshiro256PlusPlus::seed_from_u64(4664397);
       RandomGaussianGenerator::<F>{rng, _ty: PhantomData}
    }

    pub fn generate_matrix(&mut self, dims: Ix2) -> RandomGaussianMatrix<F> {
        RandomGaussianMatrix::<F>::new(dims)
    }


    // generate a standard N(0,1) vector of N(0,1) of dimension dim
    fn generate_stdn_vect(&mut self, dim: Ix1) -> Array1<F> {
        let stdnormal = StandardNormal{};
        let gauss_v : Array1<F> =  ArrayBase::from_shape_fn(dim, |_| {
            F::from_f64(stdnormal.sample(&mut self.rng)).unwrap()
        });
        gauss_v
    }
}  // end of impl RandomGaussianGenerator


// Recall that ndArray is C-order row order.
/// compute an approximate truncated svf
/// The data matrix is supposed given as a (m,n) matrix. m is the number of data and n their dimension.
pub struct RangeApprox<'a, F: Scalar> {
    /// matrix we want to approximate range of. We s
    data : &'a Array2<F>,
} // end of struct RangeApprox 


/// compute L2-norm of an array
#[inline]
pub fn norm_l2<D:Dimension, F:Scalar>(v : &ArrayView<F, D>) -> F {
    let s : F = v.into_iter().map(|x| (*x)*(*x)).sum::<F>();
    s.sqrt()
}


/// return  y - projection of y on space spanned by q's vectors.
fn orthogonalize_with_q<F:Scalar + ndarray::ScalarOperand >(q: &[Array1<F>], y: &mut ArrayViewMut1<F>) {
    let nb_q = q.len();
    if nb_q == 0 {
        return;
    }
    let size_d = y.len();
    // check dimension coherence between Q and y
    assert_eq!(q[nb_q - 1].len(),size_d);
    //
    let mut proj_qy = Array1::<F>::zeros(size_d);
    for i in 0..nb_q {
        proj_qy  += &(&q[i] * q[i].dot(y));
    }
    *y -= &proj_qy;
}  // end of orthogonalize_with_Q



impl <'a, F > RangeApprox<'a, F> 
     where  F : Float + Scalar  + ndarray::ScalarOperand {

    pub fn new(data : &'a Array2<F>) -> Self {
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
    fn adaptative_randomized_range_finder(&self, epsil:f64, r : usize) -> Array2<F> {
        let mut rng = RandomGaussianGenerator::new();
        let data_shape = self.data.shape();
        let m = data_shape[0];
        // q_mat and y_mat store vector of interest as rows to take care of Rust order.
        let mut q_mat = Vec::<Array1<F>>::new();         // q_mat stores vectors of size m
        let stop_val  = epsil/(10. * (2. * f64::FRAC_1_PI()).sqrt());
        // 
        // we store omaga_i vector as row vector as Rust has C order it is easier to extract rows !!
        let omega = rng.generate_matrix(Dim([data_shape[1], r]));    //  omega is (n, r)
        // We could store Y = data * omega as matrix (m,r), but as we use Y column,
        // we store Y (as Q) as a Vec of Array1<f64>
        let mut y_vec =  Vec::<Array1<F>>::with_capacity(r);
        for j in 0..r {
            y_vec.push(self.data.dot(&omega.mat.column(j)));
        }
        // This vectors stores L2-norm of each Y  vector of which there are r
        let mut norms_y : Array1<F> = (0..r).into_iter().map( |i| norm_l2(&y_vec[i].view())).collect();
        assert_eq!(norms_y.len() , r); 
        //
        let mut norm_sup_y;
        norm_sup_y = norms_y.iter().max_by(|x,y| x.partial_cmp(y).unwrap()).unwrap();
        log::debug!(" norm_sup {} ",norm_sup_y);
        let mut j = 0;
        let mut nb_iter = 0;
        let max_iter = data_shape[0].min(data_shape[1]);
        //
        while norm_sup_y > &F::from_f64(stop_val).unwrap() && nb_iter <= max_iter {
            // numerical stabilization
            if q_mat.len() > 0 {
                orthogonalize_with_q(&q_mat[0..q_mat.len()], &mut y_vec[j].view_mut());
            }
            // get norm of current y vector
            let n_j =  norm_l2(&y_vec[j].view());
            if n_j < Float::epsilon() {
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
                    let prodq_y = &q_j * q_j.view().dot(&y_vec[k]);
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


    /// Algorithm 4.4 from Tropp
    /// This algorith returns a (m,l) matrix approximation the range of input, q is a number of iterations
    fn randomized_subspace_iteration(&self, l:usize, q : usize)  {
        let mut rng = RandomGaussianGenerator::<F>::new();
        let data_shape = self.data.shape();
        let m = data_shape[0];
        let omega = rng.generate_matrix(Dim([data_shape[1], l]));
        let q = self.data.dot(&omega.mat);
        // do first QR decomposition of q and overwrite it
    }

}  // end of impl RangeApprox


fn check_range_finder<F:Float+ Scalar> (a_mat : &ArrayView2<F>, q_mat: &ArrayView2<F>) -> F {
    let residue = a_mat - q_mat.dot(&q_mat.t()).dot(a_mat);
    let norm_residue = norm_l2(&residue.view());
    norm_residue
}


//================================ SVD part ===============================


pub struct SvdApprox<'a, F: Scalar> {
    /// matrix we want to approximate range of. We s
    data : &'a Array2<F>,
    u : Option<Array2<F>>,
    s : Option<Array1<F>>,
    vt : Option<Array2<F>>
} // end of struct SvdApprox


impl <'a, F> SvdApprox<'a, F>  
     where  F : Float + Lapack + Scalar  + ndarray::ScalarOperand  {

    fn new(data : &'a Array2<F>) -> Self {
        SvdApprox{data, u : None, s : None, vt :None}
    }
 
    // direct svd from Algo 5.1 of Halko-Tropp
    fn direct_svd(&mut self, order : usize) -> Result<usize, String> {
        let ra = RangeApprox::new(self.data);
        // CAVEAT parameters to adjusted
        let q = ra.adaptative_randomized_range_finder(0.2, order + 10);
        let mut b = q.t().dot(self.data);
        //
        let layout = MatrixLayout::C { row: b.shape()[0] as i32, lda: b.shape()[1] as i32 };
        let slice_for_svd_opt = b.as_slice_mut();
        if slice_for_svd_opt.is_none() {
            println!("direct_svd Matrix cannot be transformed into a slice : not contiguous or not in standard order");
            return Err(String::from("not contiguous or not in standard order"));
        }
        let res_svd = F::svddc(layout, UVTFlag::Some, slice_for_svd_opt.unwrap());
        if res_svd.is_err()  {
            println!("direct_svd, svddc failed");
        };
        // we have to decode res and fill in SvdApprox fields.
        // lax does encapsulte dgesvd (double) and sgesvd (single)  which returns U and Vt as vectors.
        // We must reconstruct Array2 from slices.
        // now we must match results
        // u is (m,r) , vt must be (r, n) with m = self.data.shape()[0]  and n = self.data.shape()[1]

        //
        Ok(1)
    } // end of do_svd

} // end of block impl for SvdApprox

//=========================================================================

mod tests {

use super::*;




#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  

#[test]
    fn test_arrayview_mut() {
        let mut array = ndarray::array![[1, 2], [3, 4]];
        let to_add =  ndarray::array![1, 1];
        let mut row_mut = array.row_mut(0);
        row_mut += &to_add;
        assert_eq!(array[[0,0]], 2);
        assert_eq!(array[[0,1]], 3);
    } // end of test_arrayview_mut

    #[test]
    fn test_range_approx_1() {
        log_init_test();
        //
        let data = RandomGaussianGenerator::<f64>::new().generate_matrix(Dim([6,50]));
        let range_approx = RangeApprox::new(&data.mat);
        let q = range_approx.adaptative_randomized_range_finder(0.05, 5);
        let residue = check_range_finder(&data.mat.view(), &q.view());
        log::debug!(" residue {:3.e} ", residue);
    } // end of test_range_approx_1

    #[test]
    fn test_range_approx_2() {
        log_init_test();
        //
        let data = RandomGaussianGenerator::<f32>::new().generate_matrix(Dim([50,500]));
        let range_approx = RangeApprox::new(&data.mat);
        let q = range_approx.adaptative_randomized_range_finder(0.05, 5);
        let residue = check_range_finder(&data.mat.view(), &q.view());
        log::debug!(" residue {:3.e} ", residue);
    } // end of test_range_approx_1


}  // end of module test