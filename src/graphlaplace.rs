//! Graph Laplacian stuff

use ndarray::{Array, Array1, Array2};

use lax::{JobSvd, Lapack, layout::MatrixLayout};
// use ndarray_linalg::SVDDC;

use crate::tools::matrepr::*;
use crate::tools::svdapprox::*;

pub(crate) const FULL_MAT_REPR: usize = 5000;

pub(crate) const FULL_SVD_SIZE_LIMIT: usize = 5000;

/// We use a normalized symetric laplacian to go to the svd.
/// But we want the left eigenvectors of the normalized R(andom)W(alk) laplacian so we must keep track
/// of normalizer (rown L1 norms) used in D^{-1/2} * G * D^{-1/2} renormalization
#[derive(Clone)]
pub(crate) struct GraphLaplacian {
    // symetrized kernel. Exactly D^{-1/2} * G * D^{-1/2}
    sym_kernel: MatRepr<f32>,
    // the vector giving D of the symetrized graph (it is normed_scales * sqrt(kernel row))
    // kernel row is q_{\alpha, \epsil} in Berry-Harlim
    pub(crate) normalizer: Array1<f32>,
    //
    pub(crate) svd_res: Option<SvdResult<f32>>,
    // normed scales if not constant (constant means 1. everywhere).
    normed_scales: Option<Array1<f32>>,
    // The laplacian used to get Carre Du Champ
    pub(crate) laplacian: Option<MatRepr<f32>>,
}

impl GraphLaplacian {
    pub fn new(
        sym_kernel: MatRepr<f32>,
        normalizer: Array1<f32>,
        scales: Option<Array1<f32>>,
    ) -> Self {
        GraphLaplacian {
            sym_kernel,
            normalizer,
            svd_res: None,
            normed_scales: scales,
            laplacian: None,
        }
    } // end of new for GraphLaplacian

    #[inline]
    fn is_csr(&self) -> bool {
        self.sym_kernel.is_csr()
    } // end is_csr

    fn get_nbrow(&self) -> usize {
        self.normalizer.len()
    }

    // returns scales if any
    #[allow(unused)]
    fn get_scales(&self) -> Option<&Array1<f32>> {
        self.normed_scales.as_ref()
    }

    #[allow(unused)]
    pub fn get_laplacian(&self) -> Option<&MatRepr<f32>> {
        log::error!("not yet implemented");
        self.laplacian.as_ref()
    }

    #[allow(unused)]
    pub fn get_kernel(&self) -> &MatRepr<f32> {
        &self.sym_kernel
    }

    fn do_full_svd(&mut self) -> Result<SvdResult<f32>, String> {
        //
        log::info!("GraphLaplacian doing full svd");
        log::debug!("memory  : {:?}", memory_stats::memory_stats().unwrap());
        let b = self.sym_kernel.get_full_mut().unwrap();
        log::trace!(
            "GraphLaplacian ... size nbrow {} nbcol {} ",
            b.shape()[0],
            b.shape()[1]
        );
        //
        svd_f32(b)
    } // end of do_full_svd

    /// do a partial approxlated svd
    fn do_approx_svd(&mut self, asked_dim: usize) -> Result<SvdResult<f32>, String> {
        assert!(asked_dim >= 2);
        // get eigen values of normalized symetric lapalcian
        //
        //  switch to full or partial svd depending on csr representation and size
        // csr implies approx svd.
        log::info!(
            "got laplacian, going to approximated svd ... asked_dim :  {}",
            asked_dim
        );
        let mut svdapprox = SvdApprox::new(&self.sym_kernel);
        // TODO adjust epsil ?
        // we need one dim more beccause we get rid of first eigen vector as in dmap, and for slowly decreasing spectrum RANK approx is
        // better see Halko-Tropp
        let rank = 20;
        let nbiter = 5;
        log::trace!("asking svd, RangeRank rank : {}, nbiter : {}", rank, nbiter);
        //
        let svdmode = RangeApproxMode::RANK(RangeRank::new(rank, nbiter));
        let svd_res = svdapprox.direct_svd(svdmode);
        log::trace!("exited svd");
        if svd_res.is_err() {
            log::error!("svd approximation failed");
            std::panic!();
        }
        self.check_norms(svd_res.as_ref().unwrap());
        svd_res
    } // end if do_approx_svd

    pub fn do_svd(&mut self, asked_dim: usize) -> Result<SvdResult<f32>, String> {
        if !self.is_csr() && self.get_nbrow() <= FULL_SVD_SIZE_LIMIT {
            // try direct svd
            self.do_full_svd()
        } else {
            self.do_approx_svd(asked_dim)
        }
    } // end of init_from_sv_approx

    /// computes laplacian from kernel and scales
    pub fn compute_laplacian(&mut self) {
        // we must provide laplacian = Kernel - Identity/(scale[i] * scale[i]
        // and we have a symetrized Kernel by normalizer = scale * q.sqrt()
        if self.get_kernel().is_csr() {
            let kernel = self.get_kernel().get_csr().unwrap();
            let mut laplacian = kernel.clone();
            assert_eq!(laplacian.shape().0, laplacian.shape().1);
            let scales = self.normed_scales.as_ref().unwrap();
            let outer_iter = laplacian.outer_iterator_mut();
            // we de-symerize, see comments in the Full case
            for (row, mut row_vec) in outer_iter.enumerate() {
                for (col, val) in row_vec.iter_mut() {
                    *val *=
                        (scales[row] * self.normalizer[col]) / (scales[col] * self.normalizer[row]);

                    if row == col {
                        *val -= scales[row] * scales[row]; // diagnal term
                    }
                }
            }
            self.laplacian = Some(MatRepr::from_csrmat(laplacian));
        } else {
            // full matrix
            let kernel = self.get_kernel().get_full().unwrap();
            let mut laplacian = kernel.clone();
            let scales = self.normed_scales.as_ref().unwrap();
            assert_eq!(laplacian.shape()[0], laplacian.shape()[1]);
            // recall our normalizer is scale * sqrt(q) in Berry-Harlim notations
            // we must correct by qj.sqrt() / qi.sqrt() and q.sqrt() = normalizer / scale
            for i in 0..laplacian.shape()[0] {
                for j in 0..laplacian.shape()[1] {
                    laplacian[[i, j]] *=
                        (scales[i] * self.normalizer[j]) / (scales[j] * self.normalizer[i]);
                }
                laplacian[[i, i]] -= scales[i] * scales[i]; // diagnal term
            }
            self.laplacian = Some(MatRepr::from_array2(laplacian));
        }
        //
    }

    #[allow(unused)]
    pub(crate) fn check_norms(&self, svd_res: &SvdResult<f32>) {
        log::trace!("in of check_norms");
        //
        let u = svd_res.get_u_ref().unwrap();
        log::debug!("checking U norms , dim : {:?}", u.dim());
        let (nb_rows, nb_cols) = u.dim();
        for i in 0..nb_cols.min(3) {
            let norm = norm_frobenius_full(&u.column(i));
            log::debug!(" vector {} norm {:.2e} ", i, norm);
        }
        log::trace!("end of check_norms");
    }
} // end of impl GraphLaplacian

//=======================================================================================

//
// return s and u, used in symetric case
//
pub(crate) fn svd_f32(b: &mut Array2<f32>) -> Result<SvdResult<f32>, String> {
    let layout = MatrixLayout::C {
        row: b.shape()[0] as i32,
        lda: b.shape()[1] as i32,
    };
    let slice_for_svd_opt = b.as_slice_mut();
    if slice_for_svd_opt.is_none() {
        log::error!(
            "direct_svd Matrix cannot be transformed into a slice : not contiguous or not in standard order"
        );
        return Err(String::from("not contiguous or not in standard order"));
    }
    // use divide conquer (calls lapack gesdd), faster but could use svd (lapack gesvd)
    log::trace!("direct_svd calling svddc driver");
    let res_svd_b = f32::svddc(layout, JobSvd::Some, slice_for_svd_opt.unwrap());
    if res_svd_b.is_err() {
        log::error!("direct_svd, svddc failed");
    };
    // we have to decode res and fill in SvdApprox fields.
    // lax does encapsulte dgesvd (double) and sgesvd (single)  which returns U and Vt as vectors.
    // We must reconstruct Array2 from slices.
    // now we must match results
    // u is (m,r) , vt must be (r, n) with m = self.data.shape()[0]  and n = self.data.shape()[1]
    let res_svd_b = res_svd_b.unwrap();
    let r = res_svd_b.s.len();
    let m = b.shape()[0];
    // must convert from Real to Float ...
    let s: Array1<f32> = res_svd_b.s.iter().copied().collect::<Array1<f32>>();
    //
    // we have to decode res and fill in SvdApprox fields.
    // lax does encapsulte dgesvd (double) and sgesvd (single)  which returns U and Vt as vectors.
    // We must reconstruct Array2 from slices.
    // now we must match results
    // u is (m,r) , vt must be (r, n) with m = self.data.shape()[0]  and n = self.data.shape()[1]
    // must truncate to asked dim
    let s_u: Option<Array2<f32>>;
    if let Some(u_vec) = res_svd_b.u {
        let u_1 = Array::from_shape_vec((m, r), u_vec).unwrap();
        s_u = Some(u_1);
    } else {
        s_u = None;
    }
    //
    Ok(SvdResult {
        s: Some(s),
        u: s_u,
        vt: None,
    })
}

//==========================================================================

#[cfg(test)]
mod tests {

    //    cargo test graphlaplace  -- --nocapture
    //    RUST_LOG=annembed::tools::svdapprox=TRACE cargo test svdapprox  -- --nocapture

    use super::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // to check svd_f32
    #[test]
    fn test_svd_wiki_rank_svd_f32() {
        //
        log_init_test();
        //
        log::info!("\n\n test_svd_wiki");
        // matrix taken from wikipedia (4,5)

        let row_0: [f32; 5] = [1., 0., 0., 0., 2.];
        let row_1: [f32; 5] = [0., 0., 3., 0., 0.];
        let row_2: [f32; 5] = [0., 0., 0., 0., 0.];
        let row_3: [f32; 5] = [0., 2., 0., 0., 0.];

        let mut mat = ndarray::arr2(
            &[row_0, row_1, row_2, row_3], // row 3
        );
        //
        let epsil: f32 = 1.0E-5;
        let res = svd_f32(&mut mat).unwrap();
        let computed_s = res.get_sigma().as_ref().unwrap();
        let sigma = ndarray::arr1(&[3., (5f32).sqrt(), 2., 0.]);
        for i in 0..computed_s.len() {
            log::debug! {"sp  i  exact : {}, computed {}", sigma[i], computed_s[i]};
            let test = if sigma[i] > 0. {
                ((1. - computed_s[i] / sigma[i]).abs() as f32) < epsil
            } else {
                ((sigma[i] - computed_s[i]).abs() as f32) < epsil
            };
            assert!(test);
        }
    }
} // end of mod test
