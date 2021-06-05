//! umap-like embedding from GrapK

#![allow(dead_code)]
// #![recursion_limit="256"]

use num_traits::{Float, NumAssign};
use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView1, Axis, Array};
use ndarray_linalg::{Lapack, Scalar};
use sprs::{CsMat, TriMatBase};

use lax::{layout::MatrixLayout, UVTFlag, SVDDC_};

use quantiles::{ckms::CKMS};     // we could use also greenwald_khanna

// threading needs
use rayon::prelude::*;
use parking_lot::{RwLock};
use std::sync::Arc;

use rand::{Rng, thread_rng};
use rand::distributions::{Uniform};

use std::time::Duration;
use cpu_time::ProcessTime;

use crate::fromhnsw::*;
use crate::tools::svdapprox::*;


/// do not consider probabilities under PROBA_MIN, thresolded!!
const PROBA_MIN: f32 = 1.0E-5;


// We need this structure to compute entropy od neighbour distribution
/// This structure stores gathers parameters of a node:
///  - its local scale
///  - list of edges. The f32 field constains distance (increasing order) or directed (decreasing) proba of edge going out of each node
///    (distance and proba) to its nearest neighbours as referenced in field neighbours of KGraph.
///
/// Identity of neighbour node must be fetched in KGraph structure to spare memory
#[derive(Clone)]
struct NodeParam {
    scale: f32,
    edges: Vec<OutEdge<f32>>,
}

impl NodeParam {
    pub fn new(scale: f32, edges: Vec<OutEdge<f32>>) -> Self {
        NodeParam { scale, edges }
    }

    /// for a given node index return corresponding edge if it is in neighbours, None else 
    pub fn get_edge(&self, i : NodeIdx) -> Option<&OutEdge<f32>> {
        self.edges.iter().find( |&&edge| edge.node == i)
    }  // end of is_around

    /// perplexity. Hill number cf Leinster
    pub fn get_perplexity(&self) -> f32 {
        let h : f32 = self.edges.iter().map(|&x| -x.weight * x.weight.ln()).sum();
        h.exp()
    }
} // end of NodeParam

/// We maintain NodeParam for each node as it enables scaling in the embedded space and cross entropy minimization.
struct NodeParams {
    params: Vec<NodeParam>,
}

impl NodeParams {
    pub fn get_node_param(&self, node: NodeIdx) -> &NodeParam {
        return &self.params[node];
    }

    pub fn get_nb_nodes(&self) -> usize {
        self.params.len()
    }
} // end of NodeParams




//==================================================================================================================


/// We use a normalized symetric laplacian to go to the svd.
/// But we want the left eigenvectors of the normalized R(andom)W(alk) laplacian so we must keep track
/// of degrees (rown L1 norms)
struct LaplacianGraph {
    // symetrized graph. Exactly D^{-1/2} * G * D^{-1/2}
    sym_laplacian: MatRepr<f32>,
    // the vector giving D of the symtrized graph
    degrees: Array1<f32>,
    // 
    s : Option<Array1<f32>>,
    //
    u : Option<Array2<f32>>
}


impl LaplacianGraph {


    pub fn new(sym_laplacian: MatRepr<f32>, degrees: Array1<f32>) -> Self {
        LaplacianGraph{sym_laplacian, degrees, s : None, u: None}
    } // end of new for LaplacianGraph



#[inline]
    fn is_csr(&self) -> bool {
        self.sym_laplacian.is_csr()
    } // end is_csr

    fn get_nbrow(&self) -> usize {
        self.degrees.len()
    }

    fn do_full_svd(&mut self) -> Result<SvdResult<f32>, String> {
        //
        log::trace!("LaplacianGraph doing full svd");
        let b = self.sym_laplacian.get_full_mut().unwrap();
        log::trace!("LaplacianGraph ... size nbrow {} nbcol {} ", b.shape()[0], b.shape()[1]);

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
            log::info!("LaplacianGraph do_full_svd svddc failed");
            return Err(String::from("LaplacianGraph svddc failed"));
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
        log::debug!("got laplacian, going to approximated svd ... asked_dim :  {}", asked_dim);
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



    fn do_svd(&mut self, asked_dim : usize) -> Result<SvdResult<f32>, String> {
        if !self.is_csr() && self.get_nbrow() <= 300 {  // try direct svd
            self.do_full_svd()
        }
        else {
            self.do_approx_svd(asked_dim)
        }
     
    } // end of init_from_sv_approx


} // end of impl LaplacianGraph




//=====================================================================================


/// main parameters driving Embeding
#[derive(Clone, Copy)]
pub struct EmbedderParams {
    pub asked_dim : usize,
    pub b : f64,
    pub scale_rho : f64,
    pub grad_step : f64,
    pub max_grad_iter : usize,
} // end of EmbedderParams


impl EmbedderParams {
    pub fn new()  -> Self {
        let asked_dim = 2;
        let b = 1.;
        let grad_step = 0.1;
        let scale_rho = 1.;
        let max_grad_iter = 100;
        EmbedderParams{asked_dim, b, scale_rho, grad_step, max_grad_iter}
    }

} // end of impl EmbedderParams


/// The structure corresponding to the embedding process
/// It must be initiamized by the graph extracted from Hnsw according to the choosen strategy
/// and the asked dimension for embedding
pub struct Embedder<'a,F> {
    /// graph constrcuted with fromhnsw module
    kgraph: &'a KGraph<F>,
    /// parameters
    parameters : EmbedderParams,
    /// tells if we used approximated svd (with CSR mode)
    approximated_svd : bool,
    /// contains edge probabilities according to the probabilized graph constructed before laplacian symetrization
    /// It is this representation that is used for cross entropy optimization!
    initial_space: Option<NodeParams>,
    /// initial embedding (option for degugging analyzing)
    initial_embedding : Option<Array2<F>>,
    /// final embedding
    embedding: Option<Array2<F>>,
} // end of Embedder


impl<'a,F> Embedder<'a,F>
where
    F: Float + Lapack + Scalar + ndarray::ScalarOperand + Send + Sync,
{
    /// constructor from a graph and asked embedding dimension
    pub fn new(kgraph : &'a KGraph<F>, parameters : EmbedderParams) -> Self {
        Embedder::<F>{kgraph, parameters , approximated_svd : false, initial_space:None, initial_embedding : None, embedding:None}
    } // end of new

    pub fn get_asked_dimension(&self) -> usize {
        self.parameters.asked_dim
    }

    pub fn get_scale_rho(&self) -> f64 {
        self.parameters.scale_rho
    }

    pub fn get_b(&self) -> f64 {
        self.parameters.b
    }

    pub fn get_grad_step(&self) -> f64 {
        self.parameters.grad_step
    }

    pub fn get_max_grad_iter(&self) -> usize {
        self.parameters.max_grad_iter

    }

    /// do the embedding
    pub fn embed(&mut self) -> Result<usize, usize> {
        //
        let kgraph_stats = self.kgraph.get_kraph_stats();
        // construction of initial neighbourhood, scales and weight of edges from distances.
        self.initial_space = Some(NodeParams{params : self.construct_initial_space()});
        // we can initialize embedding with diffusion maps or pure random.
        // initial embedding via diffusion maps
//        let initial_embedding = self.get_dmap_initial_embedding(self.asked_dimension);
        // if we use random initialization we must have a coherent box size.
        let initial_embedding = self.get_random_init(1.);
        // we nedd to construct field initial_space has been contructed in get_laplacian 
        // cross entropy optimization
        let embedding_res = self.entropy_optimize(self.get_b(), self.get_grad_step(), &initial_embedding);
        // optional store dump initial embedding
        self.initial_embedding = Some(initial_embedding);
        //
        match embedding_res {
            Ok(embedding) => {
                self.embedding = Some(embedding);
                return Ok(1);
            }
            _ => {
                log::info!("Embedder::embed : embedding optimization failed");
                return Err(1);
            }        
        }

        //
    } /// end embed


    /// returns the embedded data
    pub fn get_emmbedded(&self) -> Option<&Array2<F>> {
        return self.embedding.as_ref();
    }

     /// returns the initial embedding. Storage is optional TODO
     pub fn get_initial_embedding(&self) -> Option<&Array2<F>> {
        return self.initial_embedding.as_ref();
    }   

    /// get random initialization in a square of side size
    fn get_random_init(&mut self, size:f32) -> Array2<F> {
        log::trace!("Embedder get_random_init with size {:.2e}", size);
        //
        let nb_nodes = self.initial_space.as_ref().unwrap().get_nb_nodes();
        let mut initial_embedding = Array2::<F>::zeros((nb_nodes, self.get_asked_dimension()));
        let unif = Uniform::<f32>::new(-size/2. , size/2.);
        let mut rng = thread_rng();
        for i in 0..nb_nodes {
            for j in 0..self.get_asked_dimension() {
                initial_embedding[[i,j]] = F::from(rng.sample(unif)).unwrap();
            }
        }
        //
        initial_embedding
    }  // end of get_random_init


    /// determines scales in initial space and proba of edges.
    /// Construct node params for later optimization
    // after this function we do not need field kgraph anymore!
    fn construct_initial_space(&self) -> Vec::<NodeParam> {
        let nbnodes = self.kgraph.get_nb_nodes();
        let mut perplexity_q : CKMS<f32> = CKMS::<f32>::new(0.001);
        let mut scale_q : CKMS<f32> = CKMS::<f32>::new(0.001);
        // get stats
        let mut node_params = Vec::<NodeParam>::with_capacity(nbnodes);
        // TODO can be // with rayon taking care of indexation
        let neighbour_hood = self.kgraph.get_neighbours();
        for i in 0..neighbour_hood.len() {
            let node_param = self.get_scale_from_proba_normalisation(&neighbour_hood[i]);
            scale_q.insert(node_param.scale);
            perplexity_q.insert(node_param.get_perplexity());
//            log::trace!(" perplexity node {}  : {:.2e}", i , node_param.get_perplexity());
            assert_eq!(node_param.edges.len(), neighbour_hood[i].len());
            node_params.push(node_param);
        }
        // dump info on quantiles
        println!("\n constructed initial space");
        println!("\n\n scales quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
        scale_q.query(0.05).unwrap().1, scale_q.query(0.5).unwrap().1, 
        scale_q.query(0.95).unwrap().1, scale_q.query(0.99).unwrap().1);
        println!("");
        //
        println!("\n\n perplexity quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
        perplexity_q.query(0.05).unwrap().1, perplexity_q.query(0.5).unwrap().1, 
        perplexity_q.query(0.95).unwrap().1, perplexity_q.query(0.99).unwrap().1);
        println!("");    
        //
        node_params
    }  // end of construction of node params



    // this function initialize and returns embedding by a svd (or else?)
    // We are intersested in first eigenvalues (excpeting 1.) of transition probability matrix
    // i.e last non null eigenvalues of laplacian matrix!!
    // It is in fact diffusion Maps at time 0
    //
    fn get_dmap_initial_embedding(&mut self, asked_dim: usize) -> Array2<F> {
        //
        assert!(asked_dim >= 2);
        // get eigen values of normalized symetric lapalcian
        let mut laplacian = self.get_laplacian();
        //
        log::debug!("got laplacian, going to svd ... asked_dim :  {}", asked_dim);
        let svd_res = laplacian.do_svd(asked_dim+5).unwrap();
        // As we used a laplacian and probability transitions we eigenvectors corresponding to lower eigenvalues
        let lambdas = svd_res.get_sigma().as_ref().unwrap();
        // singular vectors are stored in decrasing order according to lapack for both gesdd and gesvd. 
        if lambdas.len() > 2 && lambdas[1] > lambdas[0] {
            panic!("svd spectrum not decreasing");
        }
        // we examine spectrum
        // our laplacian is without the term I-G , we use directly G symetrized so we consider upper eigenvalues
        log::info!(" first 3 eigen values {:.2e} {:.2e} {:2e}",lambdas[0], lambdas[1] , lambdas[2]);
        // get info on spectral gap
        log::info!(" last eigenvalue computed rank {} value {:.2e}", lambdas.len()-1, lambdas[lambdas.len()-1]);
        //
        log::debug!("keeping columns from 1 to : {}", asked_dim);
        // We get U at index in range first_non_zero-max_dim..first_non_zero
        let u = svd_res.get_u().as_ref().unwrap();
        log::debug!("u shape : nrows: {} ,  ncols : {} ", u.nrows(), u.ncols());
        // we can get svd from approx range so that nrows and ncols can be number of nodes!
        let mut embedded = Array2::<F>::zeros((u.nrows(), asked_dim));
        // according to theory (See Luxburg or Lafon-Keller diffusion maps) we must go back to eigen vectors of rw laplacian.
        // Appendix A of Coifman-Lafon Diffusion Maps. Applied Comput Harmonical Analysis 2006.
        // moreover we must get back to type F
        let sum_diag = laplacian.degrees.into_iter().sum::<f32>(); 
        for i in 0..u.nrows() {
            let row_i = u.row(i);
            let weight_i = (laplacian.degrees[i]/sum_diag).sqrt();
            for j in 0..asked_dim {
                // divide j value by diagonal and convert to F. TODO could take l_{i}^{t} as in dmap?
                embedded[[i, j]] =
                    F::from_f32(row_i[j+1] / weight_i).unwrap();
            }
        }
        log::trace!("ended get_dmap_initial_embedding");
        return embedded;
    } // end of get_dmap_initial_embedding



    // the function computes a symetric laplacian graph for svd.
    // We will then need the lower non zero eigenvalues and eigen vectors.
    // The best justification for this is in Diffusion Maps.
    //
    // Store in a symetric matrix representation dense of CsMat with for spectral embedding
    // Do the Svd to initialize embedding. After that we do not need any more a full matrix.
    //      - Get maximal incoming degree and choose either a CsMat or a dense Array2.
    //
    // Let x a point y_i its neighbours
    //     after simplification weight assigned can be assumed to be of the form exp(-alfa * (d(x, y_i))
    //     the problem is : how to choose alfa, this is done in get_scale_from_proba_normalisation
    // See also Veerman A Primer on Laplacian Dynamics in Directed Graphs 2020 arxiv https://arxiv.org/abs/2002.02605

    fn get_laplacian(&mut self) -> LaplacianGraph {
        log::trace!("in Embedder::get_laplacian");
        //
        let nbnodes = self.kgraph.get_nb_nodes();
        // get stats
        let max_nbng = self.kgraph.get_max_nbng();
        let node_params = self.initial_space.as_ref().unwrap();
        // TODO define a threshold for dense/sparse representation
        if nbnodes <= 300 {
            log::debug!("Embedder using full matrix");
            let mut transition_proba = Array2::<f32>::zeros((nbnodes, nbnodes));
            // we loop on all nodes, for each we want nearest neighbours, and get scale of distances around it
            for i in 0..node_params.params.len() {
                // remind to index each request
                log::trace!(" scaling node {}", i);
                let node_param = node_params.get_node_param(i);
//                log::trace!(" perplexity node {}  : {:.2e}", i , node_param.get_perplexity());
                // CAVEAT diagonal transition 0. or 1. ? Choose 0. as in t-sne umap LargeVis
                transition_proba[[i, i]] = 0.;
                for j in 0..node_param.edges.len() {
                    let edge = node_param.edges[j];
                    transition_proba[[i, edge.node]] = edge.weight;
                } // end of for j
            } // end for i
            log::trace!("scaling of nodes done");            
            // now we symetrize the graph by taking mean
            // The UMAP formula (p_i+p_j - p_i *p_j) implies taking the non null proba when one proba is null,
            // so UMAP initialization is more packed.
            let mut symgraph = (&transition_proba + &transition_proba.view().t()) * 0.5;
            // now we go to the symetric laplacian D^-1/2 * G * D^-1/2 but get rid of the I - ... cf Jostdan
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
            let laplacian = LaplacianGraph::new(MatRepr::from_array2(symgraph), diag);
            laplacian
        } else {
            log::debug!("Embedder using csr matrix");
            self.approximated_svd = true;
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
                assert!(*i != *j);  // we do not store null distance for self (loop) edge, its proba transition is always set to 0. CAVEAT
                let sym_val;
                if let Some(t_val) = edge_list.get(&(*j, *i)) {
                    sym_val = (val + t_val) * 0.5;
                } else {
                    sym_val = *val;
                }
                diagonal[*i] += sym_val;
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
            // now we push terms (i,i) in csr
            for i in 0..nbnodes {
                rows.push(i);
                cols.push(i);
                values.push(0.);
            }
            // Now we reset non diagonal terms to I-D^-1/2 G D^-1/2  i.e 1. - val[i,j]/(D[i]*D[j])^1/2
            for i in 0..rows.len() {
                let row = rows[i];
                let col = cols[i];
                if row == col {
                    values[i] = 1. - values[i] / (diagonal[row] * diagonal[col]).sqrt();
                }
                else {
                    values[i] = - values[i] / (diagonal[row] * diagonal[col]).sqrt();
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
            let laplacian = LaplacianGraph::new(MatRepr::from_csrmat(csr_mat),diagonal);
            laplacian
        } // end case CsMat
          //
    } // end of into_matrepr_for_svd


    // given neighbours of a node we choose scale to satisfy a normalization constraint.
    // p_i = exp[- beta * (d(x,y_i) - d(x, y_1)/ local_scale ]
    // We return beta/local_scale
    // as function is monotonic with respect to scale, we use dichotomy.
    fn get_scale_from_umap(&self, norm: f64, neighbours: &Vec<OutEdge<F>>) -> (f32, Vec<f32>) {
        // p_i = exp[- beta * (d(x,y_i)/ local_scale) ]
        let nbgh = neighbours.len();
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut dist = neighbours
            .iter()
            .map(|n| n.weight.to_f32().unwrap())
            .collect::<Vec<f32>>();
        let f = |beta: f32| {
            dist.iter()
                .map(|d| (-(d - rho_x) * beta).exp())
                .sum::<f32>()
        };
        // f is decreasing
        // TODO we could also normalize as usual?
        let beta = dichotomy_solver(false, f, 0f32, f32::MAX, norm as f32);
        // TODO get quantile info on beta or corresponding entropy ? β should be not far from 1?
        // reuse rho_y_s to return proba of edge
        for i in 0..nbgh {
            dist[i] = (-(dist[i] - rho_x) * beta).exp();
        }
        // in this state neither sum of proba adds up to 1 neither is any entropy (Shannon or Renyi) normailed.
        (1. / beta, dist)
    } // end of get_scale_from_umap


    // Simplest function where we know really what we do and why. Get a normalized proba with constraint.
    // given neighbours of a node we choose scale to satisfy a normalization constraint.
    // p_i = exp[- beta * (d(x,y_i)/ local_scale)]  and then normalized to 1.
    // local_scale can be adjusted so that ratio of last proba to first proba >= epsil.
    // This function returns the local scale (i.e mean distance of a point to its nearest neighbour)
    // and vector of proba weight to nearest neighbours. Min
    fn get_scale_from_proba_normalisation(&self, neighbours: &Vec<OutEdge<F>>) -> NodeParam {
//        log::trace!("in Embedder::get_scale_from_proba_normalisation");
        // p_i = exp[- beta * (d(x,y_i)/ local_scale) * lambda]
        let nbgh = neighbours.len();
        // determnine mean distance to nearest neighbour at local scale
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut rho_y_s = Vec::<f32>::with_capacity(neighbours.len() + 1);
        for i in 0..nbgh {
            let y_i = neighbours[i].node; // y_i is a NodeIx = usize
            rho_y_s.push(self.kgraph.neighbours[y_i][0].weight.to_f32().unwrap());
            // we rho_x, initial_scales
        } // end of for i
        rho_y_s.push(rho_x);
        let mean_rho = rho_y_s.iter().sum::<f32>() / (rho_y_s.len() as f32);
        // we set scale so that transition proba do not vary more than PROBA_MIN between first and last neighbour
        // exp(- (first_dist -last_dist)/scale) >= PROBA_MIN
        // TODO do we need some optimization with respect to this 1 ? as we have lambda for high variations
        let mut scale =  self.parameters.scale_rho as f32 * mean_rho;
        // now we adjust scale so that the ratio of proba of last neighbour to first neighbour do not exceed epsil.
        let first_dist = neighbours[0].weight.to_f32().unwrap();
        let last_dist = neighbours[nbgh - 1].weight.to_f32().unwrap();
        assert!(first_dist > 0. && last_dist > 0.);
        assert!(last_dist >= first_dist);
        //
        if last_dist > first_dist {
            let lambda = (last_dist - first_dist) / (scale * (-PROBA_MIN.ln()));
            if lambda > 1. {
                log::trace!("too small scale rescaling with lambda = {}", lambda);
                // we rescale mean_rho to avoid too large range of probabilities in neighbours.
                scale = scale * lambda;
            }
            let mut probas_edge = neighbours
                .iter()
                .map(|n| OutEdge::<f32>::new(n.node, (-n.weight.to_f32().unwrap() / scale).exp()))
                .collect::<Vec<OutEdge<f32>>>();
            //
            log::trace!("scale : {:.2e} proba gap {:.2e}", scale, probas_edge[probas_edge.len() - 1].weight / probas_edge[0].weight);
            assert!(probas_edge[probas_edge.len() - 1].weight / probas_edge[0].weight >= PROBA_MIN);
            let sum = probas_edge.iter().map(|e| e.weight).sum::<f32>();
            for i in 0..nbgh {
                probas_edge[i].weight = probas_edge[i].weight / sum;
            }
            return NodeParam::new(mean_rho, probas_edge);
        } else {
            // all neighbours are at the same distance!
            let probas_edge = neighbours
                .iter()
                .map(|n| OutEdge::<f32>::new(n.node, 1.0 / nbgh as f32))
                .collect::<Vec<OutEdge<f32>>>();
            return NodeParam::new(scale, probas_edge);
        }
    } // end of get_scale_from_proba_normalisation



    /// get embedding of a given node
    pub fn get_node_embedding(&self, node : NodeIdx) -> ArrayView1<F> {
        self.embedding.as_ref().unwrap().row(node)
    }

    // minimize divergence between embedded and initial distribution probability
    // We use cross entropy as in Umap. The edge weight function must take into acccount an initial density estimate and a scale.
    // The initial density makes the embedded graph asymetric as the initial graph.
    // The optimization function thus should try to restore asymetry and local scale as far as possible.

    fn entropy_optimize(&self, b: f64, grad_step : f64, initial_embedding : &Array2<F>) -> Result<Array2<F>, String> {
        //
        log::info!("in Embedder::entropy_optimize");
        //
        if self.initial_space.is_none() {
            log::error!("Embedder::entropy_optimize : initial_space not constructed, exiting");
            return Err(String::from(" initial_space not constructed, no NodeParams"));
        }
        let ce_optimization = EntropyOptim::new(self.initial_space.as_ref().unwrap(), b, grad_step, initial_embedding);
        // compute initial value of objective function
        let start = ProcessTime::now();
        let initial_ce = ce_optimization.ce_compute();
        let cpu_time: Duration = start.elapsed();
        println!(" initial cross entropy value {:.2e},  in time {:?}", initial_ce, cpu_time);
        // We manage some iterations on gradient computing
        let grad_step_init = grad_step;
        log::info!("grad_step_init : {:.2e}", grad_step_init);
        //
        log::debug!("in Embedder::entropy_optimize  ... gradient iterations");
        let start = ProcessTime::now();
        for iter in 1..=self.get_max_grad_iter() {
            // loop on edges
            let grad_step = grad_step_init/(iter as f64).sqrt();
            let start = ProcessTime::now();
            ce_optimization.gradient_iteration(grad_step);
            let cpu_time: Duration = start.elapsed();
            println!("ce after grad iteration time {:?} grad iter {:.2e}",  cpu_time, ce_optimization.ce_compute());
            log::debug!("ce after grad iteration time {:?} grad iter {:.2e}",  cpu_time, ce_optimization.ce_compute());
        }
        let cpu_time: Duration = start.elapsed();
        println!(" gradient iterations cpu_time {:?}",  cpu_time);
        let final_ce = ce_optimization.ce_compute();
        println!(" final cross entropy value {:.2e}", final_ce);
        //
        Ok(ce_optimization.get_embedded())
        //
    } // end of entropy_optimize



} // end of impl Embedder

//==================================================================================================================


/// All we need to optimize entropy discrepancy
/// A list of edge with its weight, an array of scale for each origin node of an edge, proba (weight) of each edge
/// and coordinates in embedded_space with lock protection for //
struct EntropyOptim<'a, F> {
    /// initial space by neighbourhood of each node
    node_params: &'a NodeParams,    /// for each edge , initial node, end node, proba (weight of edge) 24 bytes
    edges : Vec<(NodeIdx, OutEdge<f32>)>,
    /// scale for each node
    initial_scales : Vec<f32>,
    /// embedded coordinates of each node, under RwLock to // optimization     nbnodes * (embedded dim * f32 + lock size))
    embedded : Vec<Arc<RwLock<Array1<F>>>>,
    /// embedded_scales
    embedded_scales : Vec<f32>,
    ///
    b : f64,
    ///
    grad_step : f64,
} // end of EntropyOptim




impl <'a, F> EntropyOptim<'a,F> 
    where F: Float +  NumAssign + std::iter::Sum + num_traits::cast::FromPrimitive + Send + Sync {
    //
    pub fn new(node_params : &'a NodeParams, b: f64, grad_step : f64, initial_embed : &Array2<F>) -> Self {
        log::info!("entering EntropyOptim::new");
        // TODO what if not the same number of neighbours!!
        let nbng = node_params.params[0].edges.len();
        let nbnodes = node_params.get_nb_nodes();
        let mut edges = Vec::<(NodeIdx, OutEdge<f32>)>::with_capacity(nbnodes*nbng);
        let mut initial_scales =  Vec::<f32>::with_capacity(nbnodes);
        // construct field edges
        for i in 0..nbnodes {
            initial_scales.push(node_params.params[i].scale);
            for j in 0..node_params.params[i].edges.len() {
                edges.push((i, node_params.params[i].edges[j]));
            }
        }
        // construct embedded, initial embed can be droped now
        let mut embedded = Vec::<Arc<RwLock<Array1<F>>>>::new();
        let nbrow  = initial_embed.nrows();
        for i in 0..nbrow {
            embedded.push(Arc::new(RwLock::new(initial_embed.row(i).to_owned())));
        }
        // compute embedded scales
//        let embedded_scales = estimate_embedded_scales_from_first_neighbour(node_params, b, initial_embed);
        let embedded_scales = estimate_embedded_scales_from_first_neighbour(node_params, b, initial_embed);
        // get qunatile on embedded scales
        let mut scales_q = CKMS::<f32>::new(0.001);
        for s in &embedded_scales {
            scales_q.insert(*s);
        }
        println!("\n\n embedded scales quantiles at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}", 
        scales_q.query(0.05).unwrap().1, scales_q.query(0.5).unwrap().1, 
        scales_q.query(0.95).unwrap().1, scales_q.query(0.99).unwrap().1);
        println!("");    
        //
        EntropyOptim { node_params,  edges, initial_scales, embedded, embedded_scales, b, grad_step}
        // construct field embedded
    }  // end of new 


    // return result as an Array2<F> cloning data to return result to struct Embedder
    fn get_embedded(& self) -> Array2<F> {
        let nbrow = self.embedded.len();
        let nbcol = self.embedded[0].read().len();
        let mut embedding_res = Array2::<F>::zeros((nbrow, nbcol));
        // TODO version 0.15 provides move_into and push_row
        for i in 0..nbrow {
            let row = self.embedded[i].read();
            for j in 0..nbcol {
                embedding_res[[i,j]] = row[j];
            }
        }
        return embedding_res;
    }


    // computes croos entropy between initial space and embedded distribution. 
    // necessary to monitor optimization
    fn ce_compute(&self) -> f64 {
        log::trace!("\n entering EntropyOptim::ce_compute");
        //
        let mut ce_entropy = 0.;
        for edge in self.edges.iter() {
            let node_i = edge.0;
            let node_j = edge.1.node;
            assert!(node_i != node_j);
            let weight_ij = edge.1.weight as f64;
            let weight_ij_embed = cauchy_edge_weight(&self.embedded[node_i].read(), 
                    self.embedded_scales[node_i] as f64, self.b,
                    &self.embedded[node_j].read()).to_f64().unwrap();
            if weight_ij_embed > 0. {
                ce_entropy += -weight_ij * weight_ij_embed.ln();
            }
            if weight_ij_embed < 1. {
                ce_entropy += - (1. - weight_ij) * (1. - weight_ij_embed).ln();
            }            
            if !ce_entropy.is_finite() {
                log::debug!("weight_ij {} weight_ij_embed {}", weight_ij, weight_ij_embed);
                std::panic!();
            }
        }
        //
        ce_entropy
    } // end of ce_compute



    // threaded version for computing cross entropy between initial distribution and embedded distribution with Cauchy law.
    fn ce_compute_threaded(&self) -> f64 {
        log::trace!("\n entering EntropyOptim::ce_compute_threaded");
        //
        let ce_entropy = self.edges.par_iter()
            .fold( || 0.0f64, | entropy : f64, edge| entropy + {
                let node_i = edge.0;
                let node_j = edge.1.node;
                let weight_ij = edge.1.weight as f64;
                let weight_ij_embed = cauchy_edge_weight(&self.embedded[node_i].read(), 
                        self.initial_scales[node_i] as f64, self.b,
                        &self.embedded[node_j].read()).to_f64().unwrap();
                let mut term = 0.;
                if weight_ij_embed > 0. {
                    term += -weight_ij * weight_ij_embed.ln();
                }
                if weight_ij_embed < 1. {
                    term += - (1. - weight_ij) * (1. - weight_ij_embed).ln();
                }
                term
            })
            .sum::<f64>();
        return ce_entropy;
    }





    // TODO : pass functions corresponding to edge_weight and grad_edge_weight as arguments to test others weight function
    /// This function optimize cross entropy for Shannon cross entropy
    fn ce_optim_edge_shannon(&self, edge_idx : usize, grad_step : f64)
    where
        F: Float + NumAssign + std::iter::Sum + num_traits::cast::FromPrimitive
    {
        // get coordinate of node
        let node_i = self.edges[edge_idx].0;
        // we locks once and directly a write lock as conflicts should be small, many edges, some threads. see Recht Hogwild!
        let mut y_i = self.embedded[node_i].write();
        let y_i_len = y_i.len();
        let mut gradient = Array1::<F>::zeros(y_i_len);
        //
        let edge_out = self.edges[edge_idx];
        let node_j = edge_out.1.node;
        assert!(node_i != node_j);
        let weight = edge_out.1.weight as f64;
        assert!(weight <= 1.);
        let scale = self.embedded_scales[node_i] as f64;
        { // get read lock 
            let mut y_j = self.embedded[node_j].write();
            // compute l2 norm of y_j - y_i
            let d_ij : f64 = y_i.iter().zip(y_j.iter()).map(|(vi,vj)| (*vi-*vj)*(*vi-*vj)).sum::<F>().to_f64().unwrap();
            let d_ij_scaled = d_ij/(scale*scale);
            let cauchy_weight = 1./ (1. + d_ij_scaled.powf(self.b));
            // this coeff is common for P and 1.-P part
            let coeff_attraction : f64;
            let coeff_repulsion : f64;
            if d_ij_scaled > 0. {
                coeff_attraction = 2. * self.b * cauchy_weight * d_ij_scaled.powf(self.b - 1.)/ (scale*scale);
                // repulsion never more than attraction
                coeff_repulsion =   coeff_attraction / (0.5 + d_ij_scaled*d_ij_scaled);
//                coeff_repulsion =   0.;
            } else { // keep a small repulsive force to detach points.
                coeff_attraction = 0.;
                coeff_repulsion =   2. * self.b * cauchy_weight / (scale * scale);
            }
            let coeff_ij = - weight * coeff_attraction + (1.-weight) * coeff_repulsion;
            for k in 0..y_i.len() {
                // clipping makes data statistically staying in box!
                if d_ij_scaled > 0. {
                    gradient[k] = clip((y_j[k] - y_i[k]) * F::from_f64(coeff_ij).unwrap(), 4.);
                }
                else { // this is a problem if weight is not 1
                    log::trace!("attraction : null dist random push");
                    let xsi = 2 * (thread_rng().gen::<u32>() % 2) - 1;   // CAVEAT to // mutex...
                    let push = (xsi  as f64) * weight * scale;
                    gradient[k] = F::from(push).unwrap();
                }
                y_i[k] -= gradient[k] * F::from_f64(grad_step).unwrap();
                y_j[k] += gradient[k] * F::from_f64(grad_step).unwrap();
            }
            log::trace!("norm attracting coeff {:.2e} gradient {:.2e}", coeff_ij, l2_norm(&gradient.view()).to_f64().unwrap());
            // now we loop on negative sampling filtering out nodes that are either node_i or are in node_i neighbours.
            let mut neg_node : NodeIdx;
            let nb_neg = 10;
            for _k in 0..nb_neg {
                neg_node = thread_rng().gen_range(0..self.embedded_scales.len());
                if neg_node != node_i && neg_node != node_j {
                    if self.node_params.get_node_param(node_i).get_edge(neg_node).is_some() {
                        log::trace!("neg node is around node i");
                        continue;
                    }
                    let y_k = self.embedded[neg_node].read();
                    // compute the common part of coeff as in function grad_common_coeff
                    let d_ik : f64 = y_i.iter().zip(y_k.iter()).map(|(vi,vj)| (*vi-*vj)*(*vi-*vj)).sum::<F>().to_f64().unwrap();
                    let d_ik_scaled = d_ik/(scale*scale);
                    let cauchy_weight = 1./ (1. + d_ik_scaled.powf(self.b));
                    let coeff : f64 = 2. * self.b * cauchy_weight * d_ik_scaled.powf(self.b - 1.)/ (scale*scale);
                    let coeff_repulsion : f64;
                    if d_ik > 0. {
                        coeff_repulsion = coeff /( 0.5 + d_ik_scaled * d_ik_scaled);  // !!

                    } else {
                        log::trace!("repulsion null dist random push");
                        coeff_repulsion =  2. * self.b;
                    }
                    let weight = 0.;   // assume weight is 0 else check if k is in neighbourhood of i, but this happens with proba nb_ng/nbnodes
                    let coeff_ik = coeff_repulsion * ( 1. - weight as f64);
                    for l in 0..y_i.len() {
                        if d_ik_scaled > 0. {
                            gradient[l] = clip((y_k[l] - y_i[l]) * F::from_f64(coeff_ik).unwrap(),4.);
                        }
                        else {
                            let xsi = 2 * (thread_rng().gen::<u32>() % 2) - 1;   // CAVEAT to // mutex...
                            let push = (xsi  as f64) * (1. - weight) * scale;
                            gradient[l] = F::from(push).unwrap();                            
                        }
                        y_i[l] = -gradient[l] * F::from_f64(grad_step).unwrap();
                    }
                    log::trace!("norm repulsive  coeff gradient {:.2e} {:.2e}", coeff_ik , l2_norm(&gradient.view()).to_f64().unwrap());
                }
            }  // end of neg sampling
        }
    } // end of ce_optim_from_point



    // TODO to be called in // all was done for
    fn gradient_iteration(&self, grad_step : f64) {
        for i in 0..self.edges.len() {
            self.ce_optim_edge_shannon(i, grad_step);
        }
    } // end of gradient_iteration



    fn gradient_iteration_threaded(&self, grad_step : f64) {
        (0..self.edges.len()).into_par_iter().for_each( |i| self.ce_optim_edge_shannon(i, grad_step));
    } // end of gradient_iteration_threaded
    
    
}  // end of impl EntropyOptim


//===============================================================================================================


// restrain value
fn clip<F>(f : F, max : f32) -> F 
    where     F: Float + num_traits::FromPrimitive  {
    let f_r = f.to_f32().unwrap();
    let truncated = if f_r > max {
        log::trace!("truncated >");
        max
    }
    else if f_r < -max {
        log::trace!("truncated <");
        -max
    }
    else {
        f_r
    };
    return F::from(truncated).unwrap();
}   // end clip



/// computes the weight of an embedded edge.
/// scale correspond at density observed at initial point in original graph (hence the asymetry)
fn cauchy_edge_weight<F>(initial_point: &Array1<F>, scale: f64, b : f64, other: &Array1<F>) -> F
where
    F: Float + std::iter::Sum + num_traits::FromPrimitive
{
    let dist = initial_point
        .iter()
        .zip(other.iter())
        .map(|(i, f)| (*i - *f) * (*i - *f))
        .sum::<F>();
    let mut dist_f64 = dist.to_f64().unwrap() / (scale*scale);
    //
    dist_f64 = dist_f64.powf(b);
    assert!(dist_f64 > 0.);
    //
    let weight =  1. / (1. + dist_f64);
    let mut weight_f = F::from_f64(weight).unwrap();
    if !(weight_f < F::one()) {
        weight_f = F::one() - F::epsilon();
        log::trace!("cauchy_edge_weight fixing dist_f64 {:2.e}", dist_f64);
    }
    assert!(weight_f < F::one());
    assert!(weight_f.is_normal());      
    return weight_f;
} // end of cauchy_edge_weight




fn l2_dist<F>(y1: &ArrayView1<'_, F> , y2 : &ArrayView1<'_, F>) -> F 
where F :  Float + std::iter::Sum + num_traits::cast::FromPrimitive {
    //
    y1.iter().zip(y2.iter()).map(|(v1,v2)| (*v1 - *v2) * (*v1- *v2)).sum()
}  // end of l2_dist




fn l2_norm<F>(v: &ArrayView1<'_, F>) -> F 
where F :  Float + std::iter::Sum + num_traits::cast::FromPrimitive {
    //
    v.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt()
}  // end of l2_norm



// in this function we compute scale in embedded space as a mean of ratio appearing in determination of sign of gradient.
// in fact likelyhood ratio. So p/(1-p) should be equilibrated. typically p1 > 0.5 for the first points of node neighbours.
// This is ensured by scale in initial space.
fn estimate_embedded_scales_from_first_neighbour<F> (node_params : &NodeParams, b : f64, embed : &Array2<F>) -> Vec<f32> 
    where  F : Float + std::iter::Sum + num_traits::cast::FromPrimitive {
    let nbnodes = node_params.params.len();
    let mut embedded_scales = Vec::<f32>::with_capacity(nbnodes);
    for i in 0..nbnodes {
        let node_param = node_params.get_node_param(i);
        let mut num = 0.;
        let mut den = 0.;
        let new_scale : f64;
        for j in 0..node_param.edges.len() {
            let ref_edge = node_param.edges[j];
            let p1 = ref_edge.weight as f64;
            num += p1* (l2_dist(&embed.row(i), &embed.row(ref_edge.node)).to_f64().unwrap()).powf(2.*b);
            den += 1.- p1;
        } 
        new_scale = (num/den).powf(0.5/b);
        log::trace!("embedded scale for node {} : {:.2e}", i , new_scale);
        embedded_scales.push(new_scale as f32);
    }
    embedded_scales
} // end of estimate_scales_from_first_neighbours



// in embedded space (in unit ball) the scale is chosen as the scale at corresponding point / divided by mean initial scales.
fn estimate_embedded_scale_from_initial_scales(initial_scales :&Vec<f32>) -> Vec<f32> {
    let mean_scale : f32 = initial_scales.iter().sum::<f32>() / (initial_scales.len() as f32);
    let embedded_scale : Vec<f32> = initial_scales.iter().map(|x| x/mean_scale).collect();
    //
    for i in 0..embedded_scale.len() {
        log::trace!("embedded scale for node {} : {:.2e}", i , embedded_scale[i]);
    }
    //
    embedded_scale
}  // end of estimate_embedded_scale_from_initial_scales



/// search a root for f(x) = target between lower_r and upper_r. The flag increasing specifies the variation of f. true means increasing
fn dichotomy_solver<F>(increasing: bool, f: F, lower_r: f32, upper_r: f32, target: f32) -> f32
where
    F: Fn(f32) -> f32,
{
    //
    if lower_r >= upper_r {
        panic!(
            "dichotomy_solver failure low {} greater than upper {} ",
            lower_r, upper_r
        );
    }
    let range_low = f(lower_r).max(f(upper_r));
    let range_upper = f(upper_r).min(f(lower_r));
    if f(lower_r).max(f(upper_r)) < target || f(upper_r).min(f(lower_r)) > target {
        panic!(
            "dichotomy_solver target not in range of function range {}  {} ",
            range_low, range_upper
        );
    }
    //
    if f(upper_r) < f(lower_r) && increasing {
        panic!("f not increasing")
    } else if f(upper_r) > f(lower_r) && !increasing {
        panic!("f not decreasing")
    }
    // target in range, proceed
    let mut middle = 1.;
    let mut upper = upper_r;
    let mut lower = lower_r;
    //
    let mut nbiter = 0;
    while (target - f(middle)).abs() > 1.0E-5 {
        if increasing {
            if f(middle) > target {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        // increasing type
        else {
            // decreasing case
            if f(middle) > target {
                lower = middle;
            } else {
                upper = middle;
            }
        } // end decreasing type
        middle = (lower + upper) * 0.5;
        nbiter += 1;
        if nbiter > 100 {
            panic!(
                "dichotomy_solver do not converge, err :  {} ",
                (target - f(middle)).abs()
            );
        }
    } // end of while
    return middle;
}

mod tests {

//    cargo test embedder  -- --nocapture


    #[allow(unused)]
    use super::*;


    use rand::distributions::{Uniform};


    // have a warning with and error without ?
    #[allow(unused)]
    use hnsw_rs::prelude::*;
    #[allow(unused)]
    use hnsw_rs::hnsw::Neighbour;

    #[allow(unused)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }


    #[test]
    fn test_dichotomy_inc() {
        let f = |x: f32| x * x;
        //
        let beta = dichotomy_solver(true, f, 0., 5., 2.);
        println!("beta : {}", beta);
        assert!((beta - 2.0f32.sqrt()).abs() < 1.0E-4);
    } // test_dichotomy_inc
    #[test]
    fn test_dichotomy_dec() {
        let f = |x: f32| 1.0f32 / (x * x);
        //
        let beta = dichotomy_solver(false, f, 0.2, 5., 1. / 2.);
        println!("beta : {}", beta);
        assert!((beta - 2.0f32.sqrt()).abs() < 1.0E-4);
    } // test_dichotomy_dec


    #[allow(unused)]
    fn gen_rand_data_f32(nb_elem: usize , dim:usize) -> Vec<Vec<f32>> {
        let mut data = Vec::<Vec<f32>>::with_capacity(nb_elem);
        let mut rng = thread_rng();
        let unif =  Uniform::<f32>::new(0.,1.);
        for i in 0..nb_elem {
            let val = 2. * i as f32 * rng.sample(unif);
            let v :Vec<f32> = (0..dim).into_iter().map(|_|  val * rng.sample(unif)).collect();
            data.push(v);
        }
        data
    }
    
    #[test]
    fn mini_embed_full() {
        log_init_test();
        // generate datz
        let nb_elem = 500;
        let embed_dim = 20;
        let data = gen_rand_data_f32(nb_elem, embed_dim);
        let data_with_id = data.iter().zip(0..data.len()).collect();
        // hnsw construction
        let ef_c = 50;
        let max_nb_connection = 50;
        let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
        let mut hns = Hnsw::<f32, DistL1>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL1{});
        // to enforce the asked number of neighbour
        hns.set_keeping_pruned(true);
        hns.parallel_insert(&data_with_id);
        hns.dump_layer_info();
        // go to kgraph
        let knbn = 10;
        let mut kgraph = KGraph::<f32>::new();
        log::info!("calling kgraph.init_from_hnsw_all");
        let res = kgraph.init_from_hnsw_all(&hns, knbn);
        if res.is_err() {
            panic!("init_from_hnsw_all  failed");
        }
        log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
        let _kgraph_stats = kgraph.get_kraph_stats();
        let mut embed_params = EmbedderParams::new();
        embed_params.asked_dim = 5;
        let mut embedder = Embedder::new(&kgraph, embed_params);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
    } // end of mini_embed_full



} // end of tests
