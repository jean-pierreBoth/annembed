//! unified matrix representation

use ndarray::{Array1, Array2, ArrayView1};

// pub to avoid to re-import everywhere explicitly
// pub use ndarray_linalg::{layout::MatrixLayout, svddc::JobSvd, Lapack, Scalar, QR};

use num_traits::float::*; // tp get FRAC_1_PI from FloatConst

use sprs::{CsMat, TriMat, prod};

use super::svdapprox::*;

/// an enum coding for the type of representation
pub enum MatType {
    FULL,
    CSR,
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
    data: MatMode<F>,
} // end of struct MatRepr

impl<F> MatRepr<F>
where
    F: Float
        + lax::Lapack
        + ndarray::ScalarOperand
        + sprs::MulAcc
        + for<'r> std::ops::MulAssign<&'r F>
        + Default
        + std::marker::Sync,
{
    /// initialize a MatRepr from an Array2
    #[inline]
    pub fn from_array2(mat: Array2<F>) -> MatRepr<F> {
        MatRepr {
            data: MatMode::FULL(mat),
        }
    }

    pub fn from_trimat(trimat: TriMat<F>) -> MatRepr<F> {
        MatRepr {
            data: MatMode::CSR(trimat.to_csr()),
        }
    }

    /// initialize a MatRepr from a CsMat
    #[inline]
    pub fn from_csrmat(mat: CsMat<F>) -> MatRepr<F> {
        assert!(mat.is_csr());
        MatRepr {
            data: MatMode::CSR(mat),
        }
    }

    /// a common interface to get matrix dimension. returns [nbrow, nbcolumn]
    pub fn shape(&self) -> [usize; 2] {
        match &self.data {
            MatMode::FULL(mat) => [mat.shape()[0], mat.shape()[1]],
            MatMode::CSR(csmat) => [csmat.shape().0, csmat.shape().1],
        }
    } // end of shape

    /// returns true if we have a row compressed representation
    pub fn is_csr(&self) -> bool {
        match &self.data {
            MatMode::FULL(_) => false,
            MatMode::CSR(_) => true,
        }
    } // end of is_csr

    /// returns a mutable reference to full matrice if data is given as full matrix, an Error otherwise
    pub fn get_full_mut(&mut self) -> Result<&mut Array2<F>, usize> {
        match &mut self.data {
            MatMode::FULL(mat) => Ok(mat),
            _ => Err(1),
        }
    } // end of get_full_mut

    /// returns a reference to full matrix if data is given as full matrix, an Error otherwise
    pub fn get_full(&self) -> Result<&Array2<F>, usize> {
        match &self.data {
            MatMode::FULL(mat) => Ok(mat),
            _ => Err(1),
        }
    } // end of get_full  

    pub fn get_csr(&self) -> Result<&CsMat<F>, usize> {
        match &self.data {
            MatMode::CSR(mat) => Ok(mat),
            _ => Err(1),
        }
    } // end of get_csr

    /// retrieve the CsMat (if mat is in this format) and consume MatRepr
    pub fn retrieve_csr(self) -> Result<CsMat<F>, usize> {
        match self.data {
            MatMode::CSR(mat) => Ok(mat),
            _ => Err(1),
        }
    }

    /// consume MatRepr and retrieve the Full mat (if mat is in this format)
    pub fn retrieve_array(self) -> Result<Array2<F>, usize> {
        match self.data {
            MatMode::FULL(mat) => Ok(mat),
            _ => Err(1),
        }
    }
    /// get a reference to matrix representation
    pub fn get_data(&self) -> &MatMode<F> {
        &self.data
    } // enf of get_data

    /// get a mutable reference to matrix representation
    pub fn get_data_mut(&mut self) -> &mut MatMode<F> {
        &mut self.data
    } // end of get_data_mut

    /// Matrix Vector multiplication. We use raw interface to get Blas.
    pub fn mat_dot_vector(&self, vec: &ArrayView1<F>) -> Array1<F> {
        match &self.data {
            MatMode::FULL(mat) => mat.dot(vec),
            MatMode::CSR(csmat) => {
                // allocate result
                let mut vres = Array1::<F>::zeros(csmat.rows());
                let vec_slice = vec.as_slice().unwrap();
                prod::mul_acc_mat_vec_csr(csmat.view(), vec_slice, vres.as_slice_mut().unwrap());
                vres
            }
        }
    } // end of matDotVector

    /// just multiplication by beta in a unified way
    pub fn scale(&mut self, beta: F) {
        match &mut self.data {
            MatMode::FULL(mat) => {
                *mat *= beta;
            }
            MatMode::CSR(csmat) => {
                csmat.scale(beta);
            }
        };
    } // end of scale

    /// return a transposed copy
    pub fn transpose_owned(&self) -> Self {
        match &self.data {
            MatMode::FULL(mat) => MatRepr::<F>::from_array2(mat.t().to_owned()),
            // in CSR mode we must reconvert to csr beccause the transposed view is csc
            MatMode::CSR(csmat) => MatRepr::<F>::from_csrmat(csmat.transpose_view().to_csr()),
        }
    } // end of transpose_owned

    /// return frobenius norm
    pub fn norm_frobenius(&self) -> F {
        match &self.data {
            MatMode::FULL(mat) => norm_frobenius_full(&mat.view()),
            MatMode::CSR(csmat) => norm_frobenius_csmat(&csmat.view()),
        }
    } // end of norm_frobenius
} // end of impl block for MatRepr
