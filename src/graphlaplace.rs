//! Graph Laplacian stuff

use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};
use sprs::{CsMat, TriMatBase};

use ndarray_linalg::SVDDC;

use crate::tools::{nodeparam::*, svdapprox::*};

const FULL_MAT_REPR: usize = 5000;

const FULL_SVD_SIZE_LIMIT: usize = 5000;

/// We use a normalized symetric laplacian to go to the svd.
/// But we want the left eigenvectors of the normalized R(andom)W(alk) laplacian so we must keep track
/// of degrees (rown L1 norms)
pub(crate) struct GraphLaplacian {
    // symetrized graph. Exactly D^{-1/2} * G * D^{-1/2}
    sym_laplacian: MatRepr<f32>,
    // the vector giving D of the symtrized graph
    pub(crate) degrees: Array1<f32>,
    //
    _s: Option<Array1<f32>>,
    //
    _u: Option<Array2<f32>>,
}

impl GraphLaplacian {
    pub fn new(sym_laplacian: MatRepr<f32>, degrees: Array1<f32>) -> Self {
        GraphLaplacian {
            sym_laplacian,
            degrees,
            _s: None,
            _u: None,
        }
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
        log::info!("GraphLaplacian doing full svd");
        let b = self.sym_laplacian.get_full_mut().unwrap();
        log::trace!(
            "GraphLaplacian ... size nbrow {} nbcol {} ",
            b.shape()[0],
            b.shape()[1]
        );

        let slice_for_svd_opt = b.as_slice_mut();
        if slice_for_svd_opt.is_none() {
            println!("direct_svd Matrix cannot be transformed into a slice : not contiguous or not in standard order");
            return Err(String::from("not contiguous or not in standard order"));
        }
        // use divide conquer (calls lapack gesdd), faster but could use svd (lapack gesvd)
        log::trace!("direct_svd calling svddc driver");
        let res_svd_b = b.svddc(JobSvd::Some);
        if res_svd_b.is_err() {
            log::info!("GraphLaplacian do_full_svd svddc failed");
            return Err(String::from("GraphLaplacian svddc failed"));
        };
        // we have to decode res and fill in SvdApprox fields.
        // lax does encapsulte dgesvd (double) and sgesvd (single)  which returns U and Vt as vectors.
        // We must reconstruct Array2 from slices.
        // now we must match results
        // u is (m,r) , vt must be (r, n) with m = self.data.shape()[0]  and n = self.data.shape()[1]
        let res_svd_b = res_svd_b.unwrap();
        // must truncate to asked dim
        let s: Array1<f32> = res_svd_b.1;
        //
        Ok(SvdResult {
            s: Some(s),
            u: res_svd_b.0,
            vt: None,
        })
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
        let mut svdapprox = SvdApprox::new(&self.sym_laplacian);
        // TODO adjust epsil ?
        // we need one dim more beccause we get rid of first eigen vector as in dmap, and for slowly decreasing spectrum RANK approx is
        // better see Halko-Tropp
        let svdmode = RangeApproxMode::RANK(RangeRank::new(20, 5));
        let svd_res = svdapprox.direct_svd(svdmode);
        log::trace!("exited svd");
        if !svd_res.is_ok() {
            println!("svd approximation failed");
            std::panic!();
        }
        return svd_res;
    } // end if do_approx_svd

    pub fn do_svd(&mut self, asked_dim: usize) -> Result<SvdResult<f32>, String> {
        if !self.is_csr() && self.get_nbrow() <= FULL_SVD_SIZE_LIMIT {
            // try direct svd
            self.do_full_svd()
        } else {
            self.do_approx_svd(asked_dim)
        }
    } // end of init_from_sv_approx
} // end of impl GraphLaplacian

// the function computes a symetric laplacian graph for svd with transition probabilities taken from NodeParams
// We will then need the lower non zero eigenvalues and eigen vectors.
// The best justification for this is in Diffusion Maps.
//
// Store in a symetric matrix representation dense of CsMat with for spectral embedding
// Do the Svd to initialize embedding. After that we do not need any more a full matrix.
//      - Get maximal incoming degree and choose either a CsMat or a dense Array2.
//
// See also Veerman A Primer on Laplacian Dynamics in Directed Graphs 2020 arxiv https://arxiv.org/abs/2002.02605

pub(crate) fn get_laplacian(initial_space: &NodeParams) -> GraphLaplacian {
    //
    log::debug!("in get_laplacian");
    //
    let nbnodes = initial_space.get_nb_nodes();
    // get stats
    let max_nbng = initial_space.get_max_nbng();
    let node_params = initial_space;
    // TODO define a threshold for dense/sparse representation
    if nbnodes <= FULL_MAT_REPR {
        log::debug!("get_laplacian using full matrix");
        let mut transition_proba = Array2::<f32>::zeros((nbnodes, nbnodes));
        // we loop on all nodes, for each we want nearest neighbours, and get scale of distances around it
        for i in 0..node_params.params.len() {
            // remind to index each request
            let node_param = node_params.get_node_param(i);
            // CAVEAT diagonal transition 0. or 1. ? Choose 0. as in t-sne umap LargeVis
            for j in 0..node_param.edges.len() {
                let edge = node_param.edges[j];
                transition_proba[[i, edge.node]] = edge.weight;
            } // end of for j
        } // end for i
        log::trace!("full matrix initialized");
        // now we symetrize the graph by taking mean
        // The UMAP formula (p_i+p_j - p_i *p_j) implies taking the non null proba when one proba is null,
        // so UMAP initialization is more packed.
        let mut symgraph = (&transition_proba + &transition_proba.view().t()) * 0.5;
        // now we go to the symetric laplacian D^-1/2 * G * D^-1/2 but get rid of the I - ...
        // cf Yan-Jordan Fast Approximate Spectral Clustering ACM-KDD 2009
        //  compute sum of row and renormalize. See Lafon-Keller-Coifman
        // Diffusions Maps appendix B
        // IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,VOL. 28, NO. 11,NOVEMBER 2006
        let diag = symgraph.sum_axis(Axis(1));
        for i in 0..nbnodes {
            let mut row = symgraph.row_mut(i);
            for j in 0..nbnodes {
                row[[j]] /= (diag[[i]] * diag[[j]]).sqrt();
            }
        }
        //
        log::trace!("\n allocating full matrix laplacian");
        let laplacian = GraphLaplacian::new(MatRepr::from_array2(symgraph), diag);
        laplacian
    } else {
        log::debug!("Embedder using csr matrix");
        // now we must construct a CsrMat to store the symetrized graph transition probablity to go svd.
        // and initialize field initial_space with some NodeParams
        let mut edge_list = HashMap::<(usize, usize), f32>::with_capacity(nbnodes * max_nbng);
        for i in 0..node_params.params.len() {
            let node_param = node_params.get_node_param(i);
            for j in 0..node_param.edges.len() {
                let edge = node_param.edges[j];
                edge_list.insert((i, edge.node), node_param.edges[j].weight);
            } // end of for j
        }
        // now we iter on the hasmap symetrize the graph, and insert in triplets transition_proba
        let mut diagonal = Array1::<f32>::zeros(nbnodes);
        let mut rows = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
        let mut cols = Vec::<usize>::with_capacity(nbnodes * 2 * max_nbng);
        let mut values = Vec::<f32>::with_capacity(nbnodes * 2 * max_nbng);

        for ((i, j), val) in edge_list.iter() {
            assert!(i != j);
            let sym_val;
            if let Some(t_val) = edge_list.get(&(*j, *i)) {
                sym_val = (val + t_val) * 0.5;
            } else {
                sym_val = *val;
            }
            rows.push(*i);
            cols.push(*j);
            values.push(sym_val);
            diagonal[*i] += sym_val;
            //
            rows.push(*j);
            cols.push(*i);
            values.push(sym_val);
            diagonal[*j] += sym_val;
        }
        // as in FULL Representation we avoided the I diagnoal term which cancels anyway
        // Now we reset non diagonal terms to D^-1/2 G D^-1/2  i.e  val[i,j]/(D[i]*D[j])^1/2
        for i in 0..rows.len() {
            let row = rows[i];
            let col = cols[i];
            if row != col {
                values[i] = values[i] / (diagonal[row] * diagonal[col]).sqrt();
            }
        }
        //
        log::trace!("allocating csr laplacian");
        let laplacian = TriMatBase::<Vec<usize>, Vec<f32>>::from_triplets(
            (nbnodes, nbnodes),
            rows,
            cols,
            values,
        );
        let csr_mat: CsMat<f32> = laplacian.to_csr();
        let laplacian = GraphLaplacian::new(MatRepr::from_csrmat(csr_mat), diagonal);
        laplacian
    } // end case CsMat
      //
} // end of get_laplacian
