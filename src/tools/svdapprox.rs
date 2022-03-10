//! This module implements a randomized truncated svd of a (m,n) matrix.
//! 
//! It builds upon the search an orthogonal matrix Q of reduced rank such that 
//!  || (I - Q * Qt) * A || < epsil.
//!
//! The reduced rank Q can be found using 2 algorithms described in 
//! Halko-Tropp Probabilistic Algorithms For Approximate Matrix Decomposition 2011
//! Cf [Halko-Tropp](https://epubs.siam.org/doi/abs/10.1137/090771806)
//!
//! - The Adaptive Randomized Range Finder (Algo 4.2 of Tropp-Halko, P 242-244)  
//!     This methods stops iterations when a precision criteria has been reached.
//! 
//! - The Randomized Subspace Iteration (Algo 4.4  of Tropp-Halko P244)  
//!     This methods asks for a specific output and is more adapted for slow decaying spectrum.
//!
//! See also Mahoney Lectures notes on randomized linearAlgebra 2016. (P 149-150).
//! We use gaussian matrix (instead SRTF as in the Ann context we have a small rank)
//! 
//! the type F must verify F : Float + FromPrimitive + Scalar + ndarray::ScalarOperand + Lapack
//! so it is f32 or f64

// num_traits::float::Float : Num + Copy + NumCast + PartialOrd + Neg<Output = Self>,  PartialOrd which is not in Scalar.
//     and nan() etc

// num_traits::Real : Num + Copy + NumCast + PartialOrd + Neg<Output = Self>
// as float but without nan() infinite() 

// ndarray::ScalarOperand provides array * F
// ndarray_linalg::Scalar provides Exp notation + Display + Debug + Serialize and sum on iterators





use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;


use ndarray::{Dim, Array, Array1, Array2, ArrayBase, Dimension, ArrayView, ArrayViewMut1, ArrayView2 , Ix1, Ix2};

// pub to avoid to re-import everywhere explicitly
pub use ndarray_linalg::{Scalar, Lapack};

use lax::{layout::MatrixLayout, UVTFlag, QR_};

use std::marker::PhantomData;

use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst
use num_traits::cast::FromPrimitive;

use sprs::{prod, CsMat, TriMat};

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


//==================================================================================================



/// an enum coding for the type of representation
pub enum MatType {
    FULL,
    CSR
}



// We can do range approximation on both dense Array2 and CsMat representation of matrices.
/// enum storing the matrix for our 2 types of matrix representation
#[derive(Clone)]
pub enum MatMode<F> {
    FULL(Array2<F>),
    CSR(CsMat<F>),
}

/// We need a minimal Matrix structure to factor the 2 linear algebra operations we need to do an approximated svd
#[derive(Clone)]
pub struct MatRepr<F> {
    data : MatMode<F>,
}  // end of struct MatRepr


impl <F> MatRepr<F> where
    F: Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default  {

    /// initialize a MatRepr from an Array2
    #[inline]
    pub fn from_array2(mat: Array2<F>) -> MatRepr<F> {
        MatRepr { data : MatMode::FULL(mat) }
    }

    pub fn from_trimat(trimat : TriMat<F>) -> MatRepr<F> {
        MatRepr { data : MatMode::CSR(trimat.to_csr()) }
    }

    /// initialize a MatRepr from a CsMat
    #[inline]
    pub fn from_csrmat(mat: CsMat<F>) -> MatRepr<F> {
        assert!(mat.is_csr());
        MatRepr { data : MatMode::CSR(mat) }
    }

    /// a common interface to get matrix dimension. returns [nbrow, nbcolumn]
    pub fn shape(&self) -> [usize; 2] {
       match &self.data {
            MatMode::FULL(mat) =>  { return [mat.shape()[0], mat.shape()[1]]; },
            MatMode::CSR(csmat) =>  { return [csmat.shape().0, csmat.shape().1]; },
        };
    } // end of shape 

    /// returns true if we have a row compressed representation
    pub fn is_csr(&self) -> bool {
        match &self.data {
            MatMode::FULL(_) =>  { return false },
            MatMode::CSR(_) =>  { return true },
        };
    } // end of is_csr


    /// returns a mutable reference to full matrice if data is given as full matrix, an Error otherwise
    pub fn get_full_mut(&mut self) -> Result<&mut Array2<F>, usize> {
        match &mut self.data {
            MatMode::FULL(mat) => { return Ok(mat) }, 
            _                  => {return Err(1) }, 
        };
    } // end of get_full_mut

    pub fn get_csr(&self) -> Result<&CsMat<F>, usize> {
        match &self.data {
            MatMode::CSR(mat) => { return Ok(mat) }, 
            _                  => {return Err(1) }, 
        };
    } // end of get_csr


    /// get a reference to matrix representation
    pub fn get_data(&self) -> &MatMode<F> {
        &self.data
    } // enf of get_data


    /// get a mutable reference to matrix representation
    pub fn get_data_mut(&mut self) -> &mut MatMode<F> {
        &mut self.data
    } // end of get_data_mut

    
    /// Matrix Vector multiplication. We use raw interface to get Blas.
    pub fn mat_dot_vector(&self, vec : &Array1<F>) -> Array1<F>  {
        match &self.data {
            MatMode::FULL(mat) => { return mat.dot(vec);},
            MatMode::CSR(csmat) =>  {
                // allocate result
                let mut vres = Array1::<F>::zeros(csmat.rows());
                let vec_slice = vec.as_slice().unwrap();
                prod::mul_acc_mat_vec_csr(csmat.view(), vec_slice, vres.as_slice_mut().unwrap());
                return vres;
            },
        };
    } // end of matDotVector

    /// just multiplication by beta in a unified way
    pub fn scale(&mut self, beta : F) {
        match &mut self.data {
            MatMode::FULL(mat) => { *mat *= beta;},
            MatMode::CSR(csmat) =>  { csmat.scale(beta) ;},
        };
    } // end of scale


    /// return a transposed copy
    pub fn transpose_owned(&self) -> Self {
        let transposed = match &self.data {
                MatMode::FULL(mat)        => { MatRepr::<F>::from_array2(mat.t().to_owned())},
                // in CSR mode we must reconvert to csr beccause the transposed view is csc
                MatMode::CSR(csmat) =>  { MatRepr::<F>::from_csrmat(csmat.transpose_view().to_csr())},
        };
        transposed
    }
} // end of impl block for MatRepr


// I need a function to compute (once and only once in svd) a product B  = tQ*CSR for Q = (m,r) with r small (<=5) and CSR(m,n)
// The matrix Q comes from range_approx so its rank (columns number) will really be small as recommended in csc_mulacc_dense_colmaj doc
// B = (r,n) with n original data dimension (we can expect n < 1000  and r <= 10
// We compute b = tQ*CSR with bt = transpose((transpose(CSR)*Q))
// We need to clone the result to enforce standard layout.

/// Returns t(qmat)*csrmat int a full matrix. Matrices must have appropriate dimensions for multiplication to avoid panic!
pub fn small_transpose_dense_mult_csr<F>(qmat : &Array2<F>, csrmat : &CsMat<F>) -> Array2<F> 
    where F: Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc {
    // transpose csrmat (it becomes a cscmat! )
    let cscmat = csrmat.transpose_view();
    let (qm_r, qm_c) = qmat.dim();             // we expect qm_c to be <= 10 as it corresponds to a rank approximated matrix
    let (csc_r , csc_c) = cscmat.shape();
    assert_eq!(csc_c, qm_r);
            //  let mut bt = Array2::<F>::zeros((csc_r, qm_c));
    let mut b =  Array2::<F>::zeros((qm_c, csc_r));
    // we transpose to get the right dimension in csc_mulacc_dense_colmaj (see the documentation for t() in ndarray)
    b.swap_axes(0,1);
    prod::csc_mulacc_dense_colmaj(cscmat, qmat.view(), b.view_mut());
    log::trace!("small_transpose_dense_mult_csr returning  ({},{}) matrix", csc_r, qm_c);
            // We want a Owned matrix in the STANDARD LAYOUT!! 
            // Array::from_shape_vec(bt.t().raw_dim(), bt.t().iter().cloned().collect()).unwrap()
    // we retranspose ! 
    b.swap_axes(0,1);
    b
} // end of small_dense_mult_csr



//==================================================================================================


/// We can ask for a range approximation of matrix on two modes:
/// - epsil     : asking for precision l2 norm residual under epsil
/// - step      : at each iteration, step new base vectors of the range matrix are searched.
///               between 5 and 10 seems adequate. To adapt to rank approximation searched.
///               Must be greater or equal to 2.
/// - max_rank  : maximum rank of approximation
#[derive(Clone, Copy, Debug)]
pub struct RangePrecision {
    /// precision asked for. Froebonius norm of the residual
    epsil :f64,
    /// increment step for the number of base vector of the range matrix  5 to 10  is a good range 
    step : usize,
    /// maximum rank asked. Iterations stop when epsil preicison is reached or maximum rank is reached.
    max_rank : usize,
}

impl RangePrecision {
    /// epsil : precision required, step : rank increment, max_rank : max rank asked
    pub fn new(epsil : f64, step_arg : usize, max_rank : usize) -> Self {
        let step;
        if step_arg <= 1 {
            log::info!("resetting step to 2, 1 is too small");
            step = 2;
        }
        else {
            step = step_arg;
        }
        RangePrecision{epsil, step, max_rank}
    }

}  // end of RangePrecision

/// We can ask for a range approximation of matrix with a fixed target range
/// - asking for a range
///    It is then necessary to fix the number of QR iterations to be done 
#[derive(Clone, Copy, Debug)]
pub struct RangeRank {
    /// asked rank
    rank : usize,
    /// number of QR decomposition
    nbiter : usize
}

impl RangeRank {
    /// initializes a RangeRank structure with asked rank and maximum QR decompositions
    pub fn new(rank: usize, nbiter : usize) -> Self {
        RangeRank{rank, nbiter}
    }
}  // end of RangeRank


/// The enum representing the 2 modes (and algorithms) of approximations
#[derive(Clone, Copy, Debug)]
pub enum RangeApproxMode {
    EPSIL(RangePrecision),
    RANK(RangeRank),
} // end of RangeApproxMode


// Recall that ndArray is C-order row order.
/// compute an approximate truncated svd.  
/// The data matrix is supposed given as a (m,n) matrix. m is the number of data and n their dimension.
pub struct RangeApprox<'a, F: Scalar> {
    /// matrix we want to approximate range of. We s
    mat : &'a MatRepr<F>,
    /// mode of approximation asked for.
    mode : RangeApproxMode
} // end of struct RangeApprox 



/// Lapack is necessary here beccause of QR_ traits coming from Lapack
impl <'a, F > RangeApprox<'a, F> 
     where  F : Float + Scalar + Lapack + ndarray::ScalarOperand  + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + num_traits::MulAdd + Default {

    /// describes the problem, matrix format and range approximation mode asked for.
    pub fn new(mat : &'a MatRepr<F>, mode : RangeApproxMode) -> Self {
        RangeApprox{mat, mode} 
    }

    /// This function returns an orthonormal matrix Q such that either  || (I - Q * Qt) * A || < epsil.
    /// or a fixed rank orthonormal Q such that || (I - Q * Qt) * A || small enough if asked rank is sufficiently large.
    /// Depending on mode, an adaptative algorithm or the fixed rang QR iterations will be called
    /// For CsMat matrice only the RangeApproxMode::EPSIL is possible (as we need QR decomposition for Sparse Mat from sprs...),
    /// in the other case the function will return None.. 
    pub fn get_approximator(&self) -> Option<Array2<F>> {
        match self.mode {
            RangeApproxMode::EPSIL(precision) => {
                return Some(adaptative_range_finder_matrep(self.mat, precision.epsil, precision.step, precision.max_rank));
            }, 
            RangeApproxMode::RANK(rank) => {
                match &self.mat.data {
                    MatMode::FULL(array) => { return Some(subspace_iteration_full(&array,  rank.rank, rank.nbiter));},

                    MatMode::CSR(csr_mat)  => { 
                                                return Some(subspace_iteration_csr(&csr_mat, rank.rank, rank.nbiter));
                                                }
                }; // end of match on representation
            },
        };
    }  // end of approximate

}  // end of impl RangeApprox



/// Given a (m,n) matrice A, this algorithm returns a (m,l) orthogonal matrix Q approximation the range of input. 
/// l is the asked rank and nb_iter is a number of iterations.
/// 
/// The matrix Q is such that || A - Q * t(Q) * A || should be small as l increases. (Froebonius norm)
/// 
/// It implements the QR iterations as descibed in Algorithm 4.4 from Halko-Tropp
/// 
pub fn subspace_iteration_full<F> (mat : &Array2<F>, rank : usize, nbiter : usize) -> Array2<F>
            where F : Float + Scalar  + Lapack + ndarray::ScalarOperand {
    //
    log::debug!("in svdapprox::subspace_iteration_full rank: {:?}, nbiter : {:?}", rank, nbiter);
    //
    let mut rng = RandomGaussianGenerator::<F>::new();
    let data_shape = mat.shape();
    let m = data_shape[0];
    let n = data_shape[1];
    let l = m.min(n).min(rank);
    if rank > l {
        log::info!("reducing asked rank in subspace_iteration to {}", l);
    }
    //
    let omega = rng.generate_matrix(Dim([data_shape[1], l]));
    let mut y_m_l = mat.dot(&omega.mat);   // y is a (m,l) matrix
    let mut y_n_l = Array2::<F>::zeros((n,l));
    let layout = MatrixLayout::C { row: m as i32, lda: l as i32 };
    // do first QR decomposition of y and overwrite it
    do_qr(layout, &mut y_m_l);
    for _j in 1..nbiter {
        // data.t() * y
        ndarray::linalg::general_mat_mul(F::one() , &mat.t(), &y_m_l, F::zero(), &mut y_n_l);
        // qr returns a (n,n)
        do_qr(MatrixLayout::C {row : y_n_l.shape()[0] as i32 ,  lda : y_n_l.shape()[1] as i32}, &mut y_n_l);
        // data * y_n_l  -> (m,l)    (m,n)*(n,l) = (m,l)    y_m_l = mat.dot(&mut y_n_l)
        ndarray::linalg::general_mat_mul(F::one() , &mat, &y_n_l, F::zero(), &mut y_m_l);
        // qr of y * data
        do_qr(MatrixLayout::C {row : y_m_l.shape()[0] as i32 ,  lda : y_m_l.shape()[1] as i32}, &mut y_m_l);
    }
    // 
    y_m_l
}  // end of subspace_iteration_full



/// Given a (m,n) matrice A, this algorithm returns a (m,l) orthogonal matrix Q approximation the range of input. 
/// l is the asked rank and nb_iter is a number of iterations.
/// 
/// The matrix Q is such that || A - Q * t(Q) * A || should be small as l increases. (Froebonius norm)
/// 
/// It implements the QR iterations as descibed in Algorithm 4.4 from Halko-Tropp
/// 
pub fn subspace_iteration_csr<F> (csrmat: &CsMat<F>, rank : usize, nbiter : usize) -> Array2<F>
            where F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc {
    //
    log::debug!("in svdapprox::subspace_iteration_csr rank: {:?}, nbiter : {:?}", rank, nbiter);
    //
    let mut rng = RandomGaussianGenerator::<F>::new();
    let data_shape = csrmat.shape();
    let m = data_shape.0;
    let n = data_shape.1;
    let l = m.min(n).min(rank);
    if rank > l {
        log::info!("reducing asked rank in subspace_iteration to {}", l);
    }
    //
    let omega = rng.generate_matrix(Dim([data_shape.1, l]));
    // y is a (m,l) matrix
    let mut y_m_l = Array2::<F>::zeros((m,l));
    prod::csr_mulacc_dense_rowmaj(csrmat.view(), omega.mat.view(), y_m_l.view_mut());
    // y_n_l is a (n,l) matrix
    let mut y_n_l = Array2::<F>::zeros((n,l));
    let layout = MatrixLayout::C { row: m as i32, lda: l as i32 };
    // do first QR decomposition of y and overwrite it
    do_qr(layout, &mut y_m_l);
    for _j in 1..nbiter {
        // data.t() * y
        y_n_l.fill(F::zero());
        prod::csc_mulacc_dense_rowmaj(csrmat.transpose_view(), y_m_l.view(), y_n_l.view_mut());
        // qr returns a (n,n)
        do_qr(MatrixLayout::C {row : y_n_l.shape()[0] as i32 ,  lda : y_n_l.shape()[1] as i32}, &mut y_n_l);
        // data * y_n_l  -> (m,l)
        y_m_l.fill(F::zero());
        prod::csr_mulacc_dense_rowmaj(csrmat.view(), y_n_l.view(), y_m_l.view_mut());
        // qr of y * data
        do_qr(MatrixLayout::C {row : y_m_l.shape()[0] as i32 ,  lda : y_m_l.shape()[1] as i32}, &mut y_m_l);
    }
    // 
    y_m_l
}  // end of subspace_iteration_matrepr





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

#[cfg_attr(doc, katexit::katexit)]
/// If mat is a (m,n) matrix this function returns an orthonormal matrix Q of dimensions (m,l) such that :
/// $$ || mat - Q*Q^{t}*mat || < ε$$  with probability at least $$ 1. - min(m,n) 10^{-r} $$
///  
///  - epsil is the l2 norm of the last block of r columns vectors added in Q.
///  - r is the number of random vectors sampled to initialize the orthogonalization process.
///    A rule of thumb is to use r between 5 ad 10. The higher the more cpu is required.
///  - max_rank is the maximum rank asked for.
/// 
/// 
///  Iterations stop when the l2 norm of the block of r vectors added is less than epsil or if max_rank has been reached. 
///  This last stop rule is somewhat easier to define.
/// 
/// The main use of this function is the following :     
/// we define  
///  $$A = Q^{t}*mat $$ so  A is a (l,n) matrix with l<<n  
///  - do the the svd of A : A = U*Σ*V  
///  - aproximate the svd of mat by (Q*U)*Σ*V
/// 
/// Algorithm : Adaptive Randomized Range Finder algo 4.2. from Halko-Martinsson-Tropp 2011
/// 
pub fn adaptative_range_finder_matrep<F>(mat : &MatRepr<F> , epsil:f64, r : usize, max_rank : usize) -> Array2<F> 
        where F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {
    //
    log::debug!("\n  in adaptative_range_finder_matrep, mat shape {:?}, epsil {:.3e}, r : {} , max_rank {}", mat.shape(), epsil, r, max_rank);
    //
    let mut rng = RandomGaussianGenerator::new();
    let data_shape = mat.shape();
    let m = data_shape[0];  // nb rows
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
        let c = omega.mat.column(j).to_owned();
        let y_tmp = mat.mat_dot_vector(&c);
        y_vec.push(y_tmp);
    }
    // This vectors stores L2-norm of each Y  vector of which there are r
    let mut norms_y : Array1<F> = (0..r).into_iter().map( |i| norm_l2(&y_vec[i].view())).collect();
    assert_eq!(norms_y.len() , r); 
    println!(" norms_y : {:.3e}",norms_y);
    //
    let mut norm_sup_y;
    let norm_iter_res = norms_y.iter().max_by(|x,y| x.partial_cmp(y).unwrap());
    if norm_iter_res.is_none() {
        log::error!("svdapprox::adaptative_range_finder_matrep cannot sort norms");
        println!("{:.3e}",norms_y);
        std::panic!("adaptative_range_finder_matrep sorting norms failed, most probably some Nan");
    }
    norm_sup_y = norm_iter_res.unwrap();
    log::debug!(" norm_sup {:.3e} ",norm_sup_y);
    let mut j = 0;
    let mut nb_iter = 0;
    let max_iter = data_shape[0].min(data_shape[1]);
    let stop_val = *norm_sup_y * F::from_f64(stop_val).unwrap();
    //
    while norm_sup_y > &stop_val && nb_iter <= max_iter && q_mat.len() < max_rank {
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
//        log::trace!("j {} n_j {:.3e} ", j, n_j);
        let q_j = &y_vec[j] / n_j;
        // we add q_j to q_mat so we consumed on y vector
        q_mat.push(q_j.clone());
        // we make another y, first we sample a new omega_j vector of size n
        let omega_j_p1 = rng.generate_stdn_vect(Ix1(data_shape[1]));
        let mut y_j_p1 = mat.mat_dot_vector(&omega_j_p1);
        // we orthogonalize new y with all q's i.e q_mat
        orthogonalize_with_q(&q_mat, &mut y_j_p1.view_mut());
        // the new y will takes the place of old y at rank j%r so we always have the last r y that have been sampled
        y_vec[j].assign(&y_j_p1);
        // we orthogonalize old y's with new q_j.  CAVEAT Can be made // if necessary 
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
        log::debug!("  nb_iter {} j {} norm_sup {:.3e}", nb_iter, j, norm_sup_y);
        // we update j and nb_iter
        j = (j+1)%r;
        nb_iter += 1;
    }
    log::debug!("adaptative_range_finder_matrep exit iteration {}, norm sup {:.3e} ", nb_iter, norm_sup_y);
    //
    // to avoid the cost to zeros
    log::debug!("range finder returning a a matrix ({}, {})", m, q_mat.len());
    //  method uninit from version 0.15.0 and later
    let mut q_as_array2  = Array2::uninit((m, q_mat.len()));     // as sprs wants ndarray 0.14.0
    for i in 0..q_mat.len() {
        for j in 0..m {
            q_as_array2[[j,i]] = std::mem::MaybeUninit::new(q_mat[i][j]);
        }
    }
    // we return an array2 where each row is a data of reduced dimension
    unsafe{ q_as_array2.assume_init()}
} // end of adaptative_range_finder_csmat


/// just to check a range approximation
pub fn check_range_approx<F:Float+ Scalar> (a_mat : &ArrayView2<F>, q_mat: &ArrayView2<F>) -> F {
    let residue = a_mat - & q_mat.dot(&q_mat.t()).dot(a_mat);
    let norm_residue = norm_l2(&residue.view());
    norm_residue
}


/// checks the quality of range  approximation.
/// The check for CSR mat is somewhat inefficient, as it involves reallocations but this functions is just for testing
pub fn check_range_approx_repr<F> (a_mat : &MatRepr<F>, q_mat: &Array2<F>) -> F 
            where F:Float+ Scalar + Lapack + ndarray::ScalarOperand + num_traits::MulAdd + sprs::MulAcc {
    let norm_residue = match &a_mat.data {
        MatMode::FULL(mat)      =>  {   let residue = mat - &(q_mat.dot(&q_mat.t()).dot(mat));
                                        let norm_residue = norm_l2(&residue.view());
                                        norm_residue
                                    },
        MatMode::CSR(csr_mat)   =>  {   let b = small_transpose_dense_mult_csr(q_mat, csr_mat);
                                        let residue = csr_mat.to_dense() - &(q_mat.dot(&b));
                                        let norm_residue = norm_l2(&residue.view());
                                        norm_residue
                                    },
   };
   norm_residue
}  // end of check_range_approx_repr



//================================ SVD part ===============================

/// result of svd.
/// For original data (m,n) with m the number of data vectors and n their dimension returns
/// U and Vt with eigenvectors stored in column, so projected data are accessed by rows
pub struct SvdResult<F> {
    /// eigenvalues
    pub   s : Option<Array1<F>>,
    /// left eigenvectors. (m,r) matrix where r is rank asked for and m the number of data.
    pub   u : Option<Array2<F>>,
    /// transpose of right eigen vectors. (r,n) matrix
    pub  vt : Option<Array2<F>>
} // end of struct SvdResult<F> 


impl <F> SvdResult<F> {
    #[inline]
    pub fn get_sigma(&self) -> &Option<Array1<F>> {
        &self.s
    }

    /// returns U
    #[inline]
    pub fn get_u(&self) -> &Option<Array2<F>> {
        &self.u
    }

    /// returns Vt
    #[inline]
    pub fn get_vt(&self) -> &Option<Array2<F>> {
        &self.vt
    }
}  // end of impl SvdResult



/// Approximated svd.
/// The first step is to find a range approximation of the matrix.
/// This step can be done by asking for a required precision or a minimum rank for dense matrices represented by Array2
/// or Csr matrices
pub struct SvdApprox<'a, F: Scalar> {
    /// matrix we want to approximate range of.
    data : &'a MatRepr<F>,
} // end of struct SvdApprox


impl <'a, F> SvdApprox<'a, F>  
     where  F : Float + Lapack + Scalar  + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + num_traits::MulAdd + Default {
    
    pub fn new(data : &'a MatRepr<F>) -> Self {
        SvdApprox{data}
    }

    /// direct svd from Algo 5.1 of Halko-Tropp
    /// Returns an error if either the preliminary range_approximation or the partial svd failed, else returns a SvdResult
    pub fn direct_svd(&mut self, parameters : RangeApproxMode) -> Result<SvdResult<F>, String> {
        let ra = RangeApprox::new(self.data, parameters);
        let q;
        let q_opt = ra.get_approximator();
        if q_opt.is_some() {
            q = q_opt.unwrap();
        }
        else {
            return Err(String::from("range approximation failed"));
        }
        //
        let mut b = match &self.data.data {
            MatMode::FULL(mat) => { q.t().dot(mat)},
            MatMode::CSR(mat)  => { 
                                    log::trace!("direct_svd got csr matrix");
                                    small_transpose_dense_mult_csr(&q, mat)
                                },
        };
        //
        let layout = MatrixLayout::C { row: b.shape()[0] as i32, lda: b.shape()[1] as i32 };
        let slice_for_svd_opt = b.as_slice_mut();
        if slice_for_svd_opt.is_none() {
            println!("direct_svd Matrix cannot be transformed into a slice : not contiguous or not in standard order");
            return Err(String::from("not contiguous or not in standard order"));
        }
        // use divide conquer (calls lapack gesdd), faster but could use svd (lapack gesvd)
        log::trace!("direct_svd calling svddc driver");
        let res_svd_b = F::svddc(layout,  UVTFlag::Some, slice_for_svd_opt.unwrap());
        if res_svd_b.is_err()  {
            println!("direct_svd, svddc failed");
        };
        // we have to decode res and fill in SvdApprox fields.
        // lax does encapsulte dgesvd (double) and sgesvd (single)  which returns U and Vt as vectors.
        // We must reconstruct Array2 from slices.
        // now we must match results
        // u is (m,r) , vt must be (r, n) with m = self.data.shape()[0]  and n = self.data.shape()[1]
        let res_svd_b = res_svd_b.unwrap();
        let r = res_svd_b.s.len();
        let m = b.shape()[0];
        let n = b.shape()[1];
        // must convert from Real to Float ...
        let s : Array1<F> = res_svd_b.s.iter().map(|x| F::from(*x).unwrap()).collect::<Array1<F>>();
        //
        let s_u : Option<Array2<F>>;
        if let Some(u_vec) = res_svd_b.u {
            let u_1 = Array::from_shape_vec((m, r), u_vec).unwrap();
            s_u = Some(q.dot(&u_1));
        }
        else {
            s_u = None;
        }
        let s_vt : Option<Array2<F>>;
        if let Some(vt_vec) = res_svd_b.vt {
            s_vt = Some(Array::from_shape_vec((r, n), vt_vec).unwrap());
        }
        else {
            s_vt = None;
        }
        //
        Ok(SvdResult{s : Some(s), u : s_u, vt : s_vt})
    } // end of do_svd

} // end of block impl for SvdApprox



//================ utilities ===========================//



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


// do qr decomposition (calling Lax q function) of mat (m, n) which must be in C order
// The purpose of this function is just to avoid the R allocation in Lax qr 
//
fn do_qr<F> (layout : MatrixLayout, mat : &mut Array2<F>)
    where F : Float + Lapack + Scalar + QR_ + ndarray::ScalarOperand 
{
    let (_, _) = match layout {
        MatrixLayout::C { row, lda } => (row as usize , lda as usize),
        _ => panic!()
    };
    let tau_res = F::householder(layout, mat.as_slice_mut().unwrap());
    if tau_res.is_err() {
        log::error!("svdapprox::do_qr : a lapack error occurred in F::householder");
        panic!();
    }
    let tau = tau_res.unwrap();
    F::q(layout, mat.as_slice_mut().unwrap(), &tau).unwrap();
} // end of do_qr



//=========================================================================

#[cfg(test)]
mod tests {

//    cargo test svdapprox  -- --nocapture
//    RUST_LOG=annembed::tools::svdapprox=TRACE cargo test svdapprox  -- --nocapture

#[allow(unused)]
use super::*;

use sprs::{CsMat, TriMatBase};

#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  

#[test]
    fn test_arrayview_mut() {
        log_init_test();
        let mut array = ndarray::array![[1, 2], [3, 4]];
        let to_add =  ndarray::array![1, 1];
        let mut row_mut = array.row_mut(0);
        row_mut += &to_add;
        assert_eq!(array[[0,0]], 2);
        assert_eq!(array[[0,1]], 3);
    } // end of test_arrayview_mut

    #[test]
    fn test_range_approx_randomized_1() {
        log_init_test();
        //
        let data = RandomGaussianGenerator::<f64>::new().generate_matrix(Dim([15,50]));
        let norm_data = norm_l2(&data.mat.view());
        let rp = RangePrecision { epsil : 0.05 , step : 5, max_rank : 10};
        let matrepr = MatRepr::from_array2(data.mat);
        let range_approx = RangeApprox::new(&matrepr,  RangeApproxMode::EPSIL(rp));
        let q = range_approx.get_approximator().unwrap();
        log::info!(" q(m,n) {} {} ", q.shape()[0], q.shape()[1]);
        let residue = check_range_approx_repr(&matrepr, &q);
        log::info!(" subspace_iteration nom_l2 {:.2e} residue {:.2e} \n", norm_data, residue);
    } // end of test_range_approx_1

    #[test]
    fn test_range_approx_randomized_2() {
        log_init_test();
        //
        let data = RandomGaussianGenerator::<f32>::new().generate_matrix(Dim([50,500]));
        let norm_data = norm_l2(&data.mat.view());
        let rp = RangePrecision { epsil : 0.05 , step : 5, max_rank : 25};
        let matrepr = MatRepr::from_array2(data.mat);
        let range_approx = RangeApprox::new(&matrepr,  RangeApproxMode::EPSIL(rp));
        let q = range_approx.get_approximator().unwrap();
        //
        log::info!(" q(m,n) {} {} ", q.shape()[0], q.shape()[1]);
        let residue = check_range_approx_repr(&matrepr, &q);
        log::info!(" subspace_iteration nom_l2 {:.2e} residue {:.2e} \n", norm_data, residue);
    } // end of test_range_approx_1


    #[test]
    fn test_range_approx_subspace_iteration_1() {
        log_init_test();
        //
        let data = RandomGaussianGenerator::<f64>::new().generate_matrix(Dim([12,50]));
        let norm_data = norm_l2(&data.mat.view());
        let rp = RangeRank { rank : 11 , nbiter : 7};            // check for too large rank asked
        let matrepr = MatRepr::from_array2(data.mat);
        let range_approx = RangeApprox::new(&matrepr, RangeApproxMode::RANK(rp));
        let q = range_approx.get_approximator().unwrap(); // args are rank , nbiter
        let residue = check_range_approx_repr(&matrepr, &q);
        log::info!(" subspace_iteration nom_l2 {:.2e} residue {:.2e} \n", norm_data, residue);
    } // end of test_range_approx_subspace_iteration_1


    #[test]
    fn test_range_approx_subspace_iteration_2() {
        log_init_test();
        //
        let mut data = RandomGaussianGenerator::<f64>::new().generate_matrix(Dim([30,500])).mat;
        // reduce rank to 26
        let new_row = data.row(2).to_owned();
        data.row_mut(3).assign(&new_row);
        data.row_mut(5).assign(&new_row);
        data.row_mut(7).assign(&new_row);
        data.row_mut(9).assign(&new_row);
        //
        let norm_data = norm_l2(&data.view());
        let rp = RangeRank { rank : 26 , nbiter : 2};
        let matrepr = MatRepr::from_array2(data);
        let range_approx = RangeApprox::new(&matrepr, RangeApproxMode::RANK(rp));
        let q = range_approx.get_approximator().unwrap();
        let residue = check_range_approx_repr(&matrepr, &q);
        log::info!(" subspace_iteration nom_l2 {:.2e} residue {:.2e} \n", norm_data, residue);
        assert!(residue < 1.0E-10);
    } // end of test_range_approx_subspace_iteration_2



#[test]
    fn check_tcsrmult_a() {
        //
        log_init_test();
        //
        log::info!("\n\n check_tcsrmultA");
            // matrix taken from wikipedia (4,5)
        let mat =  ndarray::arr2( & 
              [[ 1. , 0. , 0. , 0., 2. ],  // row 0
               [ 0. , 0. , 3. , 0. , 0. ],  // row 1
               [ 0. , 0. , 0. , 0. , 0. ],  // row 2
               [ 0. , 2. , 0. , 0. , 0. ]]  // row 3
        );
        // get same matri in a csr representation
        let csr_mat : CsMat<f64> = get_wiki_csr_mat_f64();
        // A is (4,4)
        let gmat  = RandomGaussianGenerator::<f64>::new().generate_matrix(Dim([4,4]));
        let mut prodmat = Array2::<f64>::zeros((5,4));
        prod::csc_mulacc_dense_colmaj(csr_mat.transpose_view(), gmat.mat.view(), prodmat.view_mut());
        // compare prod with standard computation
        let delta = norm_l2(&(mat.t().dot(&gmat.mat) - &prodmat).view());
        //
        log::debug!("check on usage , prod norm : {}, delta : {}", norm_l2(&prodmat.view()), delta);
        assert!(delta < 1.0E-10);
    } // end of check_tcsrmultA


    // TODO test with m >> n 
    
// tests for svd

#[test]
fn test_svd_wiki_rank_full() {
    //
    log_init_test();
    //
    log::info!("\n\n test_svd_wiki");
    // matrix taken from wikipedia (4,5)
    let mat =  ndarray::arr2( & 
      [[ 1. , 0. , 0. , 0., 2. ],  // row 0
      [ 0. , 0. , 3. , 0. , 0. ],  // row 1
      [ 0. , 0. , 0. , 0. , 0. ],  // row 2
      [ 0. , 2. , 0. , 0. , 0. ]]  // row 3
    );
    //
    let matrepr = MatRepr::from_array2(mat);
    let mut svdapprox = SvdApprox::new(&matrepr);
    let svdmode = RangeApproxMode::RANK(RangeRank{rank:4, nbiter:5});
    let svd_res = svdapprox.direct_svd(svdmode).unwrap();
    //
    let sigma = ndarray::arr1(&[ 3., (5f64).sqrt() , 2., 0.]);
    if let Some(computed_s) = svd_res.get_sigma() {
        assert!(computed_s.len() <= sigma.len());
        assert!(computed_s.len() >= 3);
        for i in 0..computed_s.len() {
            log::trace!{"sp  i  exact : {}, computed {}", sigma[i], computed_s[i]};
            let test;
            if  sigma[i] > 0. {
               test =  ((1. - computed_s[i]/sigma[i]).abs() as f32) < f32::EPSILON;
            }
            else {
               test =  ((sigma[i]-computed_s[i]).abs()  as f32) < f32::EPSILON;
            };
            assert!(test);
        }
    }
    else {
        std::panic!("test_svd_wiki_rank");
    }
} // end of test_svd_wiki


// get the wiki matrix in CsMat<f32> format
#[allow(unused)]
fn get_wiki_csr_mat_f32() -> CsMat<f32> {
    // matrix taken from wikipedia (4,5)
    // let mat =  ndarray::arr2( & 
    //   [[ 1. , 0. , 0. , 0., 2. ],  // row 0
    //   [ 0. , 0. , 3. , 0. , 0. ],  // row 1
    //   [ 0. , 0. , 0. , 0. , 0. ],  // row 2
    //   [ 0. , 2. , 0. , 0. , 0. ]]  // row 3
    // );
    let mut rows = Vec::<usize>::with_capacity(5);
    let mut cols = Vec::<usize>::with_capacity(5);
    let mut values = Vec::<f32>::with_capacity(5);
    rows.push(0);
    cols.push(0);
    values.push(1.);
    rows.push(0);
    cols.push(4);
    values.push(2.);
    // row 1    
    rows.push(1);
    cols.push(2);
    values.push(3.); 
    // row 3
    rows.push(3);
    cols.push(1);
    values.push(2.);     
    //
    let trimat = TriMatBase::<Vec<usize>, Vec<f32>>::from_triplets((4,5),rows, cols, values);
    let csr_mat : CsMat<f32> = trimat.to_csr();
    csr_mat
}  // end of get_wiki_csr_mat_f32



#[allow(unused)]
fn get_wiki_csr_mat_f64() -> CsMat<f64> {
    // matrix taken from wikipedia (4,5)
    // let mat =  ndarray::arr2( & 
    //   [[ 1. , 0. , 0. , 0., 2. ],  // row 0
    //   [ 0. , 0. , 3. , 0. , 0. ],  // row 1
    //   [ 0. , 0. , 0. , 0. , 0. ],  // row 2
    //   [ 0. , 2. , 0. , 0. , 0. ]]  // row 3
    // );
    let mut rows = Vec::<usize>::with_capacity(5);
    let mut cols = Vec::<usize>::with_capacity(5);
    let mut values = Vec::<f64>::with_capacity(5);
    rows.push(0);
    cols.push(0);
    values.push(1.);
    rows.push(0);
    cols.push(4);
    values.push(2.);
    // row 1    
    rows.push(1);
    cols.push(2);
    values.push(3.); 
    // row 3
    rows.push(3);
    cols.push(1);
    values.push(2.);     
    //
    let trimat = TriMatBase::<Vec<usize>, Vec<f64>>::from_triplets((4,5),rows, cols, values);
    let csr_mat : CsMat<f64> = trimat.to_csr();
    csr_mat
}  // end of get_wiki_csr_mat_f64


fn get_wiki_array2_f64() -> Array2<f64> {
    let mat =  ndarray::arr2( & 
              [[ 1. , 0. , 0. , 0., 2. ],  // row 0
              [ 0. , 0. , 3. , 0. , 0. ],  // row 1
              [ 0. , 0. , 0. , 0. , 0. ],  // row 2
              [ 0. , 2. , 0. , 0. , 0. ]]  // row 3
        );
    mat
} // end of get_wiki_array2_f64

#[test]
fn test_svd_wiki_csr_epsil () {
    //
    log_init_test();
    //
    log::info!("\n\n test_svd_wiki_csr_epsil");
    //
    let csr_mat : CsMat<f32> = get_wiki_csr_mat_f32();
    //
    let matrepr = MatRepr::from_csrmat(csr_mat);
    let mut svdapprox = SvdApprox::new(&matrepr);
    let svdmode = RangeApproxMode::EPSIL(RangePrecision{epsil:0.1 , step:5, max_rank : 10});
    let svd_res = svdapprox.direct_svd(svdmode).unwrap();
    //
    let sigma = ndarray::arr1(&[ 3., (5f32).sqrt() , 2., 0.]);
    if let Some(computed_s) = svd_res.get_sigma() {
        log::trace!{ "computed spectrum size {}", computed_s.len()};
        assert!(computed_s.len() <= sigma.len());
        assert!(computed_s.len() >= 3);
        for i in 0..computed_s.len() {
            log::trace!{"sp  i  exact : {}, computed {}", sigma[i], computed_s[i]};
            let test;
            if  sigma[i] > 0. {
               test =  ((1. - computed_s[i]/sigma[i]).abs() as f32) < 1.0E-5;
            }
            else {
               test =  ((sigma[i]-computed_s[i]).abs()  as f32) < 1.0E-5;
            };
            assert!(test);
        }
    }
    else {
        std::panic!("test_svd_wiki_csr_epsil");
    }
} // end of test_svd_wiki


// test rank approx for a csr representation
#[test]
fn test_svd_wiki_csr_rank() {
    //
    log_init_test();
    //
    log::info!("\n\n test_svd_wiki_csr_rank");
    //
    let csr_mat : CsMat<f32> = get_wiki_csr_mat_f32();
    //
    let matrepr = MatRepr::from_csrmat(csr_mat);
    let mut svdapprox = SvdApprox::new(&matrepr);
    //
    let svdmode = RangeApproxMode::RANK(RangeRank{rank:4, nbiter:5});
    let svd_res = svdapprox.direct_svd(svdmode).unwrap();
    let sigma = ndarray::arr1(&[ 3., (5f32).sqrt() , 2., 0.]);
    if let Some(computed_s) = svd_res.get_sigma() {
        log::trace!{ "computed spectrum size {}", computed_s.len()};
        assert!(computed_s.len() <= sigma.len());
        assert!(computed_s.len() >= 3);
        for i in 0..computed_s.len() {
            log::trace!{"sp  i  exact : {}, computed {}", sigma[i], computed_s[i]};
            let test;
            if  sigma[i] > 0. {
               test =  ((1. - computed_s[i]/sigma[i]).abs() as f32) < 1.0E-5;
            }
            else {
               test =  ((sigma[i]-computed_s[i]).abs()  as f32) < 1.0E-5;
            };
            assert!(test);
        }
    }
    else {
        std::panic!("test_svd_wiki_csr_epsil");
    }
}  // end of test_svd_wiki_csr_rank




#[test]
fn test_svd_wiki_full_epsil () {
    //
    log_init_test();
    //
    log::info!("\n\n test_svd_wiki");
    // matrix taken from wikipedia (4,5)
    let mat =  ndarray::arr2( & 
      [[ 1. , 0. , 0. , 0., 2. ],  // row 0
      [ 0. , 0. , 3. , 0. , 0. ],  // row 1
      [ 0. , 0. , 0. , 0. , 0. ],  // row 2
      [ 0. , 2. , 0. , 0. , 0. ]]  // row 3
    );
    //
    let matrepr = MatRepr::from_array2(mat);
    let mut svdapprox = SvdApprox::new(&matrepr);
    let svdmode = RangeApproxMode::EPSIL(RangePrecision{epsil:0.1 , step:5, max_rank : 4});
    let svd_res = svdapprox.direct_svd(svdmode).unwrap();
    //
    let sigma = ndarray::arr1(&[ 3., (5f64).sqrt() , 2., 0.]);
    if let Some(computed_s) = svd_res.get_sigma() {
        assert!(computed_s.len() >= 3);
        assert!(sigma.len() >= computed_s.len());
        for i in 0..computed_s.len() {
            log::trace!{"sp  i  exact : {}, computed {}", sigma[i], computed_s[i]};
            //
            let test;
            if  sigma[i] > 0. {
                test = ((1. - computed_s[i]/sigma[i]).abs() as f32) < f32::EPSILON;
            }
            else {
                test = ((sigma[i]-computed_s[i]).abs()  as f32) < f32::EPSILON ;
            };
            //
            assert!(test);
        }
    }
    else {
        std::panic!("test_svd_wiki_epsil");
    }
} // end of test_svd_wiki



#[test]
fn check_small_transpose_dense_mult_csr() {
   //
    log_init_test();
    // get wiki (4,5) matrix
    let csr_mat = get_wiki_csr_mat_f64();
    let gmat  = RandomGaussianGenerator::<f64>::new().generate_matrix(Dim([4,4]));
    // compute transpose(gmat.mat) *csr_mat
    let mult_res = small_transpose_dense_mult_csr(&gmat.mat,& csr_mat);
    // brute force
    let brute_res = &gmat.mat.t().dot(&get_wiki_array2_f64());
    let delta = norm_l2(&(mult_res - brute_res).view());
    //
    log::debug!("check_small_transpose_dense_mult_csr, delta : {}", delta);
    assert!(delta < 1.0E-10);
} // end of check_small_transpose_dense_mult_csr



#[test]
fn check_transpose_owned() {
    //
    log_init_test();
    //
    let mat = MatRepr::<f32>::from_csrmat(get_wiki_csr_mat_f32());
    let transposed = mat.transpose_owned();
    let transposed_csr = transposed.get_csr().unwrap();
    // check transposed is a csr
    assert!(transposed_csr.is_csr());
    // check csr_mat is conserved
    let check = mat.get_csr().unwrap().get(0,4).unwrap();
    log::debug!("old value should be 2  : {}", check);
    assert!((check- 2.).abs() < 1.0E-10); 
     // check transposed is correct
     let check = transposed_csr.get(4,0).unwrap(); 
     log::debug!("transposed value should be 2  : {}", check);
     assert!((check - 2.).abs() < 1.0E-10); 
      
} // end of check_transpose_owned


}  // end of module test