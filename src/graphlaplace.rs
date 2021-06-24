//! Graph Laplacian stuff


use ndarray::{Array1, Array2, Array};

use lax::{layout::MatrixLayout, UVTFlag, SVDDC_};

use crate::tools::svdapprox::*;




const FULL_SVD_SIZE_LIMIT : usize = 2000;

/// We use a normalized symetric laplacian to go to the svd.
/// But we want the left eigenvectors of the normalized R(andom)W(alk) laplacian so we must keep track
/// of degrees (rown L1 norms)
pub(crate) struct GraphLaplacian {
    // symetrized graph. Exactly D^{-1/2} * G * D^{-1/2}
    sym_laplacian: MatRepr<f32>,
    // the vector giving D of the symtrized graph
    pub(crate) degrees: Array1<f32>,
    // 
    _s : Option<Array1<f32>>,
    //
    _u : Option<Array2<f32>>
}


impl GraphLaplacian {


    pub fn new(sym_laplacian: MatRepr<f32>, degrees: Array1<f32>) -> Self {
        GraphLaplacian{sym_laplacian, degrees, _s : None, _u: None}
    } // end of new for GraphLaplacian



#[inline]
    fn is_csr(&self) -> bool {
        self.sym_laplacian.is_csr()
    } // end is_csr

    fn get_nbrow(&self) -> usize {
        self.degrees.len()
    }

    fn do_full_svd(&mut self) -> Result<SvdResult<f32>, String> {
        //
        log::trace!("GraphLaplacian doing full svd");
        let b = self.sym_laplacian.get_full_mut().unwrap();
        log::trace!("GraphLaplacian ... size nbrow {} nbcol {} ", b.shape()[0], b.shape()[1]);

        let layout = MatrixLayout::C { row: b.shape()[0] as i32, lda: b.shape()[1] as i32 };
        let slice_for_svd_opt = b.as_slice_mut();
        if slice_for_svd_opt.is_none() {
            println!("direct_svd Matrix cannot be transformed into a slice : not contiguous or not in standard order");
            return Err(String::from("not contiguous or not in standard order"));
        }
        // use divide conquer (calls lapack gesdd), faster but could use svd (lapack gesvd)
        log::trace!("direct_svd calling svddc driver");
        let res_svd_b = f32::svddc(layout,  UVTFlag::Some, slice_for_svd_opt.unwrap());
        if res_svd_b.is_err()  {
            log::info!("GraphLaplacian do_full_svd svddc failed");
            return Err(String::from("GraphLaplacian svddc failed"));
        };
        // we have to decode res and fill in SvdApprox fields.
        // lax does encapsulte dgesvd (double) and sgesvd (single)  which returns U and Vt as vectors.
        // We must reconstruct Array2 from slices.
        // now we must match results
        // u is (m,r) , vt must be (r, n) with m = self.data.shape()[0]  and n = self.data.shape()[1]
        let res_svd_b = res_svd_b.unwrap();
        let r = res_svd_b.s.len();
        let m = b.shape()[0];
        // must truncate to asked dim
        let s : Array1<f32> = res_svd_b.s.iter().map(|x| *x).collect::<Array1<f32>>();
        //
        let s_u : Option<Array2<f32>>;
        if let Some(u_vec) = res_svd_b.u {
            s_u = Some(Array::from_shape_vec((m, r), u_vec).unwrap());
        }
        else {
            s_u = None;
        }
        Ok(SvdResult{s : Some(s), u: s_u, vt : None})
    }  // end of do_full_svd


    /// do a partial approxlated svd
    fn do_approx_svd(&mut self, asked_dim : usize) -> Result<SvdResult<f32>, String> {
        assert!(asked_dim >= 2);
        // get eigen values of normalized symetric lapalcian
        //
        //  switch to full or partial svd depending on csr representation and size
        // csr implies approx svd.
        log::info!("got laplacian, going to approximated svd ... asked_dim :  {}", asked_dim);
        let mut svdapprox = SvdApprox::new(&self.sym_laplacian);
        // TODO adjust epsil ?
        // we need one dim more beccause we get rid of first eigen vector as in dmap
        let svdmode = RangeApproxMode::EPSIL(RangePrecision::new(0.1, 25, asked_dim+5));
        let svd_res = svdapprox.direct_svd(svdmode);
        log::trace!("exited svd");
        if !svd_res.is_ok() {
            println!("svd approximation failed");
            std::panic!();
        }
        return svd_res;
    } // end if do_approx_svd



    pub fn do_svd(&mut self, asked_dim : usize) -> Result<SvdResult<f32>, String> {
        if !self.is_csr() && self.get_nbrow() <= FULL_SVD_SIZE_LIMIT {  // try direct svd
            self.do_full_svd()
        }
        else {
            self.do_approx_svd(asked_dim)
        }
     
    } // end of init_from_sv_approx


} // end of impl GraphLaplacian
