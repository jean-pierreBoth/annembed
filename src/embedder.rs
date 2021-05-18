//! umap-like embedding from GrapK

#![allow(dead_code)]
// #![recursion_limit="256"]

use num_traits::Float;
use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Lapack, Scalar};
use sprs::{CsMat, TriMatBase};

use crate::fromhnsw::*;
use crate::tools::svdapprox::*;

// We need this structure to compute entropy od neighbour distribution
/// This structure stores for each node its local scale and proba to its nearest neighbours as referenced in
/// field neighbours of KGraph. Field neighbours gives us acces to neighbours id and original distances.
/// So this field gives a local scale and transition proba.
/// Identity of neighbour node must be fetched in KGraph structure to spare memory
#[derive(Clone)]
struct NodeParam {
    scale: f32,
    probas: Vec<f32>,
}

impl NodeParam {
    pub fn new(scale: f32, probas: Vec<f32>) -> Self {
        NodeParam { scale, probas }
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
} // end of NodeParams

/// All we need to optimize entropy discrepancy
/// For each node i , its knbg neighbours in neighbours[i] , its initial scale and transition proba in initial_space
/// and in embedded_space
struct EntropyOptim {
    neighbours: Vec<Vec<NodeIdx>>,
    initial_space: Option<NodeParams>,
    embedded_space: Option<NodeParams>,
}

/// We use a normalized symetric laplacian to go to the svd.
/// But we want the left eigenvectors of the normalized R(andom)W(alk) laplacian so we must keep track
/// of degrees (rown norms)
pub struct LaplacianGraph {
    sym_laplacian: MatRepr<f32>,
    degrees: Array1<f32>,
}

/// The structure corresponding to the embedding process
/// It must be initiamized by the graph extracted from Hnsw according to the choosen strategy
/// and the asked dimension for embedding
pub struct Emmbedder<'a, F> {
    kgraph: &'a KGraph<F>,
    asked_dimension: usize,
    initial_space: Option<NodeParams>,
    embedded_space: Option<NodeParams>,
    embedding: Option<Array2<F>>,
} // end of Emmbedder

impl<F> Emmbedder<'_, F>
where
    F: Float + Lapack + Scalar + ndarray::ScalarOperand + Send + Sync,
{
    // this function initialize embedding by a svd (or else?)
    // We are intersested in first eigenvalues (excpeting 1.) of transition probability matrix
    // i.e last non null eigenvalues of laplacian matrix!!
    // It is in fact diffusion Maps at time 0
    fn get_dmap_initial_embedding(&mut self, asked_dim: usize) -> Array2<F> {
        // get eigen values of normalized symetric lapalcian
        let laplacian = self.get_laplacian();
        let mut svdapprox = SvdApprox::new(&laplacian.sym_laplacian);
        // TODO adjust epsil ?
        let svdmode = RangeApproxMode::EPSIL(RangePrecision::new(0.1, 5));
        let svd_res = svdapprox.direct_svd(svdmode);
        if !svd_res.is_ok() {
            println!("svd approximation failed");
            std::panic!();
        }
        // As we used a laplacian and probability transitions we eigenvectors corresponding to lower eigenvalues
        let lambdas = svdapprox.get_sigma().as_ref().unwrap();
        // singular vectors are stored in decrasing order according to lapack for both gesdd and gesvd
        if lambdas.len() > 2 && lambdas[1] > lambdas[0] {
            panic!("svd spectrum not decreasing");
        }
        // get first non null lamba
        let first_non_zero_opt = lambdas.iter().rev().position(|&x| x > 0.);
        if !first_non_zero_opt.is_some() {
            println!("cannot find positive eigenvalue");
            std::panic!();
        } else {
            let first_non_zero = lambdas.len() - 1 - first_non_zero_opt.unwrap(); // is in [0..len()-1]
            log::info!(
                "last non null eigenvalue at rank : {}, value : {}",
                first_non_zero,
                lambdas[first_non_zero]
            );
            assert!(first_non_zero >= asked_dim);
            let max_dim = asked_dim.min(first_non_zero + 1); // is in [1..len()]
                                                             // We get U at index in range first_non_zero-max_dim..first_non_zero
            let u = svdapprox.get_u().as_ref().unwrap();
            assert_eq!(u.nrows(), self.kgraph.get_nbng());
            let mut embedded = Array2::<F>::zeros((u.nrows(), max_dim));
            // according to theory (See Luxburg or Lafon-Keller diffusion maps) we must go back to eigen vectors of rw laplacian.
            // moreover we must get back to type F
            let sum_diag = laplacian.degrees.into_iter().sum::<f32>().sqrt();
            let j_weights: Vec<f32> = laplacian
                .degrees
                .into_iter()
                .map(|x| x.sqrt() / sum_diag)
                .collect();
            for i in 0..u.nrows() {
                let row_i = u.row(i);
                for j in 0..asked_dim {
                    // divide j value by diagonal and convert to F
                    embedded[[i, j]] =
                        F::from_f32(row_i[first_non_zero - j] / j_weights[i]).unwrap();
                }
            }
            // according to theory (See Luxburg or Lafon-Keller diffusion maps) we must go back to eigen vectors of rw laplacian.
            // moreover we must get back to type F
            return embedded;
        }
        //
    } // end of get_initial_embedding

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

    fn get_laplacian(&self) -> LaplacianGraph {
        let nbnodes = self.kgraph.get_nb_nodes();
        // get stats
        let graphstats = self.kgraph.get_kraph_stats();
        let nbng = self.kgraph.get_nbng();
        let mut node_params = Vec::<NodeParam>::with_capacity(nbnodes);
        // TODO define a threshold for dense/sparse representation
        if nbnodes <= 30000 {
            let mut transition_proba = Array2::<f32>::zeros((nbnodes, nbnodes));
            // we loop on all nodes, for each we want nearest neighbours, and get scale of distances around it
            // TODO can be // with rayon taking care of indexation
            let neighbour_hood = self.kgraph.get_neighbours();
            for i in 0..neighbour_hood.len() {
                // remind to index each request
                let node_param = self.get_scale_from_proba_normalisation(&neighbour_hood[i]);
                assert_eq!(node_param.probas.len(), neighbour_hood[i].len());
                for j in 0..neighbour_hood[i].len() {
                    let edge = neighbour_hood[i][j];
                    transition_proba[[i, edge.node]] = node_param.probas[j];
                } // end of for j
                node_params.push(node_param);
            } // end for i
              // now we symetrize the graph by taking mean
              // The UMAP formula (p_i+p_j - p_i *p_j) implies taking the non null proba when one proba is null,
              // so UMAP initialization is more packed.
            let mut symgraph = (&transition_proba + &transition_proba.view().t()) * 0.5;
            // now we go to the symetric laplacian. compute sum of row and renormalize. See Lafon-Keller-Coifman
            // Diffusions Maps appendix B
            // IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,VOL. 28, NO. 11,NOVEMBER 2006
            let diag = symgraph.sum_axis(Axis(1));
            for i in 0..nbnodes {
                let mut row = symgraph.row_mut(i);
                for j in 0..nbnodes {
                    row /= -(diag[[i]] * diag[[j]]).sqrt();
                }
                row[[i]] = 1.;
            }
            //
            let laplacian = LaplacianGraph {
                sym_laplacian: MatRepr::from_array2(symgraph),
                degrees: diag,
            };
            laplacian
        } else {
            // now we must construct a CsrMat to store the symetrized graph transition probablity to go svd.
            let neighbour_hood = self.kgraph.get_neighbours();
            // TODO This can be made // with a chashmap
            let mut edge_list = HashMap::<(usize, usize), f32>::with_capacity(nbnodes * nbng);
            for i in 0..neighbour_hood.len() {
                let node_param = self.get_scale_from_proba_normalisation(&neighbour_hood[i]);
                assert_eq!(node_param.probas.len(), neighbour_hood[i].len());
                for j in 0..neighbour_hood[i].len() {
                    let edge = neighbour_hood[i][j];
                    edge_list.insert((i, edge.node), node_param.probas[j]);
                } // end of for j
                node_params.push(node_param);
            }
            // now we iter on the hasmap symetrize the graph, and insert in triplets transition_proba
            let mut diagonal = Array1::<f32>::zeros(nbnodes);
            let mut rows = Vec::<usize>::with_capacity(nbnodes * 2 * nbng);
            let mut cols = Vec::<usize>::with_capacity(nbnodes * 2 * nbng);
            let mut values = Vec::<f32>::with_capacity(nbnodes * 2 * nbng);

            for ((i, j), val) in edge_list.iter() {
                assert!(*i != *j);
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
            // Now we reset non diagonal terms to 1 - val[i,j]/(D[i]*D[j])^1/2 add diagonal term to 1.
            for i in 0..rows.len() {
                let row = rows[i];
                let col = cols[i];
                values[i] = 1. - values[i] / (diagonal[row] * diagonal[col]).sqrt();
            }
            for i in 0..nbnodes {
                rows.push(i);
                cols.push(i);
                values.push(1.);
            }
            let laplacian = TriMatBase::<Vec<usize>, Vec<f32>>::from_triplets(
                (nbnodes, nbnodes),
                rows,
                cols,
                values,
            );
            let csr_mat: CsMat<f32> = laplacian.to_csr();
            let laplacian = LaplacianGraph {
                sym_laplacian: MatRepr::from_csrmat(csr_mat),
                degrees: diagonal,
            };
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
    // local_scale can be adjusted so that ratio of last praba to first proba >= epsil.
    // This function returns the local scale (i.e mean distance of a point to its nearest neighbour)
    // and vector of proba weight to nearest neighbours. Min
    fn get_scale_from_proba_normalisation(&self, neighbours: &Vec<OutEdge<F>>) -> NodeParam {
        // p_i = exp[- beta * (d(x,y_i)/ local_scale) * lambda]
        const PROBA_MIN: f32 = 1.0E-5;
        let nbgh = neighbours.len();
        // determnine mean distance to nearest neighbour at local scale
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut rho_y_s = Vec::<f32>::with_capacity(neighbours.len() + 1);
        for i in 0..nbgh {
            let y_i = neighbours[i].node; // y_i is a NodeIx = usize
            rho_y_s.push(self.kgraph.neighbours[y_i][0].weight.to_f32().unwrap());
            // we rho_x, scales
        } // end of for i
        rho_y_s.push(rho_x);
        let mut mean_rho = rho_y_s.iter().sum::<f32>() / (rho_y_s.len() as f32);
        // now we adjust mean_rho so that the ratio of proba of last neighbour to first neighbour do not exceed epsil.
        let first_dist = neighbours[0].weight.to_f32().unwrap();
        let last_dist = neighbours[nbgh - 1].weight.to_f32().unwrap();
        if last_dist > first_dist {
            let lambda = (last_dist - first_dist) / (mean_rho * (-PROBA_MIN.ln()));
            if lambda > 1. {
                // we rescale mean_rho to avoid too large range of probabilities in neighbours.
                mean_rho = mean_rho * lambda;
            }
            let mut probas = neighbours
                .iter()
                .map(|n| (-n.weight.to_f32().unwrap() / mean_rho).exp())
                .collect::<Vec<f32>>();
            assert!(probas[probas.len() - 1] / probas[0] <= PROBA_MIN);
            let sum = probas.iter().sum::<f32>();
            for i in 0..nbgh {
                probas[i] = probas[i] / sum;
            }
            return NodeParam::new(mean_rho, probas);
        } else {
            // all neighbours are at the same distance!
            let probas = neighbours
                .iter()
                .map(|_| (1.0 / nbgh as f32))
                .collect::<Vec<f32>>();
            return NodeParam::new(mean_rho, probas);
        }
    } // end of get_scale_from_proba_normalisation
    // minimize divergence between embedded and initial distribution probability
    // We use cross entropy as in Umap. The edge weight function must take into acccount an initial density estimate and a scale.
    // The initial density makes the embedded graph asymetric as the initial graph.
    // The optimization function thus should try to restore asymetry and local scale as far as possible.

    fn entropy_optimize(initial: &Array2<F>) -> Result<usize, String> {
        // compute initial value of objective function

        // loop on points calling ce_optim_from_point
        Ok(1)
    } // end of entropy_optimize

    // TODO : pass functions corresponding to edge_weight and grad_edge_weight as arguments to test others weight function
    // TODO This is clearly a function that must be threaded!
    /// This function optimize cross entropy for Shannon cross entropy
    fn ce_optim_from_node(&self, node: NodeIdx)
    where
        F: Float + std::iter::Sum + num_traits::cast::FromPrimitive,
    {
        //
        //        let gradient = Array1::<F>::zeros(source.len());
        let node_params = self.initial_space.as_ref().unwrap();
        let node_param = node_params.get_node_param(node);
        // loop on connected neighbours taking into account P and 1.-P for each of its neighbours
        // for that we need read only access to kgraph and not a &mut on self

        // negative loop with sampling on points that are not initial neighbours

        // mode initial point
    } // end of ce_optim_from_point
} // end of impl Emmbedder

/// computes the weight of an embedded edge.
/// scale correspond at density observed at initial point in original graph (hence the asymetry)
fn cauchy_edge_weight<F>(initial_point: &Array1<F>, scale: F, other: &Array1<F>) -> F
where
    F: Float + std::iter::Sum,
{
    let dist = initial_point
        .iter()
        .zip(other.iter())
        .map(|(i, f)| (*i - *f) * (*i - *f))
        .sum::<F>();
    return dist / scale;
} // end of embedded_edge_weight

// gradient of embedded_edge_weight

fn grad_cauchy_edge_weight<F>(
    initial_point: &Array1<F>,
    scale: F,
    other: &Array1<F>,
    gradient: &mut Array1<F>,
) where
    F: Float + std::iter::Sum + num_traits::cast::FromPrimitive,
{
    //
    assert_eq!(gradient.len(), initial_point.len());
    //
    for i in 0..gradient.len() {
        gradient[i] = -F::from_f32(2.).unwrap() * (other[i] - initial_point[i])
            / (scale * cauchy_edge_weight(initial_point, scale, initial_point));
    }
} // end of grad_embedded_weight

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

    use super::*;
    #[allow(dead_code)]
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
} // end of tests
