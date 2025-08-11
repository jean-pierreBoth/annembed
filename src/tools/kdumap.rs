//! specific adhoc dmap kernel for initializing annembed

// Construct the representation of graph as a collections of probability-weighted edges
// determines scales in initial space and proba of edges for the neighbourhood of every point.
// Construct node params for later optimization
// after this function Embedder structure do not need field kgraph anymore
// This function relies on get_scale_from_proba_normalisation function which construct proabability-weighted edge around each node.
//

use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use quantiles::ckms::CKMS;
use sprs::{CsMat, TriMatBase}; // we could use also greenwald_khanna

use rayon::prelude::*;

use rand::Rng;

use crate::fromhnsw::kgraph::KGraph;
use crate::graphlaplace::*;
use crate::prelude::PROBA_MIN;
use crate::tools::{nodeparam::*, svdapprox::*};

pub(crate) fn to_proba_edges<F>(kgraph: &KGraph<F>, scale_rho: f32, beta: f32) -> NodeParams
where
    F: Float
        + num_traits::cast::FromPrimitive
        + std::marker::Sync
        + std::marker::Send
        + std::fmt::UpperExp
        + std::iter::Sum,
{
    //
    let mut perplexity_q: CKMS<f32> = CKMS::<f32>::new(0.001);
    let mut scale_q: CKMS<f32> = CKMS::<f32>::new(0.001);
    let mut weight_q: CKMS<f32> = CKMS::<f32>::new(0.001);
    let neighbour_hood = kgraph.get_neighbours();
    // a closure to compute scale and perplexity
    let scale_perplexity = |i: usize| -> (usize, Option<(f32, NodeParam)>) {
        if !neighbour_hood[i].is_empty() {
            let node_param =
                get_scale_from_proba_normalisation(kgraph, scale_rho, beta, &neighbour_hood[i]);
            let perplexity = node_param.get_perplexity();
            (i, Some((perplexity, node_param)))
        } else {
            (i, None)
        }
    };
    let mut opt_node_params: Vec<(usize, Option<(f32, NodeParam)>)> =
        Vec::<(usize, Option<(f32, NodeParam)>)>::new();
    let mut node_params: Vec<NodeParam> = (0..neighbour_hood.len())
        .map(|_| NodeParam::default())
        .collect();
    //
    (0..neighbour_hood.len())
        .into_par_iter()
        .map(scale_perplexity)
        .collect_into_vec(&mut opt_node_params);
    // now we process serial information related to opt_node_params
    let mut max_nbng = 0;
    for opt_param in &opt_node_params {
        match opt_param {
            (i, Some(param)) => {
                perplexity_q.insert(param.0);
                scale_q.insert(param.1.scale);
                // choose random edge to audit
                let j = rand::rng().random_range(0..param.1.edges.len());
                weight_q.insert(param.1.edges[j].weight);
                max_nbng = param.1.edges.len().max(max_nbng);
                assert_eq!(param.1.edges.len(), neighbour_hood[*i].len());
                node_params[*i] = param.1.clone();
            }
            (i, None) => {
                log::error!("to_proba_edges , node rank {}, has no neighbour, use hnsw.set_keeping_pruned(true)", i);
                log::error!("to_proba_edges , node rank {}, has no neighbour, use hnsw.set_keeping_pruned(true)", i);
                std::process::exit(1);
            }
        };
    }
    // dump info on quantiles
    log::debug!("\n constructed initial space");
    log::debug!(
        "\n scales quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}",
        scale_q.query(0.05).unwrap().1,
        scale_q.query(0.5).unwrap().1,
        scale_q.query(0.95).unwrap().1,
        scale_q.query(0.99).unwrap().1
    );
    //
    log::debug!(
        "\n edge weight quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}",
        weight_q.query(0.05).unwrap().1,
        weight_q.query(0.5).unwrap().1,
        weight_q.query(0.95).unwrap().1,
        weight_q.query(0.99).unwrap().1
    );
    //
    log::debug!(
        "\n perplexity quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}",
        perplexity_q.query(0.05).unwrap().1,
        perplexity_q.query(0.5).unwrap().1,
        perplexity_q.query(0.95).unwrap().1,
        perplexity_q.query(0.99).unwrap().1
    );
    log::debug!("");
    //
    NodeParams::new(node_params, max_nbng)
} // end of construction of node params

// Simplest function where we know really what we do and why.
// Given a graph, scale and exponent parameters transform a list of distance-edge to neighbours into a list of proba-edge.
//
// Given neighbours of a node we choose scale to satisfy a normalization constraint.
// p_i = exp[- (d(x,y_i) - shift)/ local_scale)**beta]
// with shift = d(x,y_0) and local_scale = mean_dist_to_first_neighbour * scale_rho
//  and then normalized to 1.
//
// We do not set an edge from x to itself. So we will have 0 on the diagonal matrix of transition probability.
// This is in accordance with t-sne, umap and so on. The main weight is put on first neighbour.
//
// This function returns the local scale (i.e mean distance of a point to its nearest neighbour)
// and vector of proba weight to nearest neighbours.
//
fn get_scale_from_proba_normalisation<F>(
    kgraph: &KGraph<F>,
    scale_rho: f32,
    beta: f32,
    neighbours: &[OutEdge<F>],
) -> NodeParam
where
    F: Float + num_traits::cast::FromPrimitive + Sync + Send + std::fmt::UpperExp + std::iter::Sum,
{
    //
    //        log::trace!("in get_scale_from_proba_normalisation");
    let nbgh = neighbours.len();
    assert!(nbgh > 0);
    // determnine mean distance to nearest neighbour at local scale, reason why we need kgraph as argument.
    let rho_x = neighbours[0].weight.to_f32().unwrap();
    let mut rho_y_s = Vec::<f32>::with_capacity(neighbours.len() + 1);
    //
    for neighbour in neighbours {
        let y_i = neighbour.node; // y_i is a NodeIx = usize
        rho_y_s.push(kgraph.get_neighbours()[y_i][0].weight.to_f32().unwrap());
    } // end of for i
      //
    rho_y_s.push(rho_x);
    let mean_rho = rho_y_s.iter().sum::<f32>() / (rho_y_s.len() as f32);
    // we set scale so that transition proba do not vary more than PROBA_MIN between first and last neighbour
    // exp(- (first_dist -last_dist)/scale) >= PROBA_MIN
    // TODO do we need some optimization with respect to this 1 ? as we have lambda for high variations
    let scale = scale_rho * mean_rho;
    // now we adjust scale so that the ratio of proba of last neighbour to first neighbour do not exceed epsil.
    let mut all_equal = false;
    let first_dist = neighbours[0].weight.to_f32().unwrap();
    // it happens first dist = 0 , Cf Higgs Boson data
    let last_n = neighbours
        .iter()
        .rfind(|&n| n.weight.to_f32().unwrap() > 0.);
    if last_n.is_none() {
        // means all distances are 0! (encountered in Higgs Boson bench)
        all_equal = true;
    }
    //
    let remap_weight = |w: F, shift: f32, scale: f32, beta: f32| {
        (-((w.to_f32().unwrap() - shift).max(0.) / scale).powf(beta)).exp()
    };
    //
    if !all_equal {
        let last_dist = last_n.unwrap().weight.to_f32().unwrap();
        if last_dist > first_dist {
            //
            let mut probas_edge = neighbours
                .iter()
                .map(|n| {
                    OutEdge::<f32>::new(
                        n.node,
                        remap_weight(n.weight, first_dist, scale, beta).max(PROBA_MIN),
                    )
                })
                .collect::<Vec<OutEdge<f32>>>();
            //
            let proba_range = probas_edge[probas_edge.len() - 1].weight / probas_edge[0].weight;
            log::trace!(" first dist {:2e} last dist {:2e}", first_dist, last_dist);
            log::trace!("scale : {:.2e} , first neighbour proba {:2e}, last neighbour proba {:2e} proba gap {:.2e}", scale, probas_edge[0].weight, 
                            probas_edge[probas_edge.len() - 1].weight,
                            proba_range);
            if proba_range < PROBA_MIN {
                log::error!(" first dist {:2e} last dist {:2e}", first_dist, last_dist);
                log::error!("scale : {:.2e} , first neighbour proba {:2e}, last neighbour proba {:2e} proba gap {:.2e}", scale, probas_edge[0].weight, 
                                probas_edge[probas_edge.len() - 1].weight,
                                proba_range);
            }
            assert!(
                proba_range >= PROBA_MIN,
                "proba range {:.2e} too low edge proba, increase scale_rho or reduce beta",
                proba_range
            );
            //
            let sum = probas_edge.iter().map(|e| e.weight).sum::<f32>();
            for p in probas_edge.iter_mut().take(nbgh) {
                p.weight /= sum;
            }
            return NodeParam::new(scale, probas_edge);
        } else {
            all_equal = true;
        }
    }
    if all_equal {
        // all neighbours are at the same distance!
        let probas_edge = neighbours
            .iter()
            .map(|n| OutEdge::<f32>::new(n.node, 1.0 / nbgh as f32))
            .collect::<Vec<OutEdge<f32>>>();
        NodeParam::new(scale, probas_edge)
    } else {
        log::error!("fatal error in get_scale_from_proba_normalisation, should not happen!");
        std::panic!("incoherence error");
    }
} // end of get_scale_from_proba_normalisation

//
//TODO: superseded by methods of DiffusionMaps
//
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
    log::info!("in fn get_laplacian using alfa = 0.5",);
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
        GraphLaplacian::new(MatRepr::from_array2(symgraph), diag)
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
                values[i] /= (diagonal[row] * diagonal[col]).sqrt();
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
        GraphLaplacian::new(MatRepr::from_csrmat(csr_mat), diagonal)
    } // end case CsMat
      //
} // end of get_laplacian
