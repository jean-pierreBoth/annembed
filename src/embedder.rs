//! umap-like embedding from GrapK

#![allow(dead_code)]

use num_traits::{Float};

use ndarray::{Array2};
use ndarray_linalg::{Scalar, Lapack};


use crate::fromhnsw::*;

pub struct Emmbedder <'a, F> {
    kgraph : &'a KGraph<F>, 
}

impl <F> Emmbedder<'_, F> 
    where F :  Float + Lapack + Scalar  + ndarray::ScalarOperand + Send + Sync {


    // this function compute a (generalised laplacian), do an approximated svd of it and project data on associated eigenvectors
    fn get_initial_embedding(&mut self) -> Array2<F> {
        let embedded = Array2::<F>::zeros((3,2));
        //
        embedded
    }

    fn graph_symmetrization() {

    } // end of graph_symmetrization

    /// computes a generalized laplacian with weights taking into account density of points.
    /// Veerman A Primer on Laplacian Dynamics in Directed Graphs 2020 arxiv https://arxiv.org/abs/2002.02605
    fn graph_laplacian() {

    }
    // minimize divergence between embedded and initial distribution probability
    fn entropy_optimize() {

    }

    // convert into neighbourhood probabilities
    //      - We must define rescaling/thesholding/renormalization strategy around each point
    // Store in a matrix representation with for spectral embedding
    //      - Get maximal incoming degree and choose either a CsMat or a dense Array2. 
    //
    // Let x a point y_i its neighbours
    //     after simplification weight assigned can be assumed to be of the form exp(-alfa * (d(x, y_i) - d(x, y_1)))
    //     the problem is : how to choose alfa
    fn into_matrepr(&self) {
        let nbnodes = self.kgraph.get_nb_nodes();
        // get stats
        let graphstats = self.kgraph.get_kraph_stats();
        let max_in_degree = graphstats.get_max_in_degree();
        let mut scale_params = Vec::<f32>::with_capacity(nbnodes);


       // TODO define a threshold for dense/sparse representation
        if max_in_degree > nbnodes / 10 {
            let mut transition_proba = Array2::<f32>::zeros((nbnodes,nbnodes));
    
            // we loop on all nodes, for each we want nearest neighbours, and get scale of distances around it
            // TODO can be // with rayon taking care of indexation
            let neighbour_hood = self.kgraph.get_neighbours();
            for i in 0..neighbour_hood.len() {
                // remind to index each request
                let (scale, probas) = self.get_scale_from_normalisation(1., &neighbour_hood[i]);
                scale_params.push(scale);
                assert_eq!(probas.len(), neighbour_hood[i].len());
                for j in 0..neighbour_hood[i].len() {
                    let edge = neighbour_hood[i][j];
                    transition_proba[[i,edge.node ]] = probas[j];
                } // end of for j

            }  // end for i
        }   
        else {
            panic!("csmat representation not yet done");
        }
        //
    }  // end of into_matrepr




    // this function choose (beta) scale so that at mid range among neighbours we have a proba of 1/k
    // so that  k/2 neighbours have proba > 1/K and the other half have proba less than k/2 
    // so the proba of neighbours do not sum up to 1 (is between nbgnh/2, nbgh]but split above median range.
    // in this state neither sum of proba adds up to 1 neither is any entropy (Shannon or Renyi) normalized.
    //
    // Returns mean distance to first neighbour around current point and probabilities to current neighbours.
    fn get_scale_from_neighbourhood(&self, neighbours : &Vec<OutEdge<F>>) -> (f32, Vec::<f32>) {
        // p_i = exp[- beta * (d(x,y_i) - d(x,y_1).min(local_scale))] 
        let nbgh = neighbours.len();
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut rho_min = f32::MAX;
        let mut rho_y_s = Vec::<f32>::with_capacity(neighbours.len());
        rho_y_s.push(rho_x);
        for i in 0..nbgh {
            let y_i = neighbours[i].node;      // y_i is a NodeIx = usize
            let rho_i = self.kgraph.neighbours[y_i][0].weight.to_f32().unwrap();
            rho_min = rho_min.min(rho_i);
            rho_y_s.push(rho_i);
            // we rho_x, scales
        }  // end of for i
        let nbgh_2 = nbgh/2;
        let rho_median = neighbours[nbgh_2].weight.to_f32().unwrap();
        // compute average of nearest neighbour distance around our point.
        let mean_rho = rho_y_s.iter().sum::<f32>()/ (rho_y_s.len() as f32);
        // now we have our rho for the current point, it takes into account local scale.
        // if rho_x > mean_rho distance from x to its neighbour will be penalized and first term will not be 1
        // as is the case if rho_x < mean_rho
        let rho = mean_rho.min(rho_x);
        let scale : f32;
        if rho_median > rho {
            // now we set scale so that k/2 neighbour is at proba 1/2 ?
            scale = (2. * nbgh as f32).ln() / (rho_median - rho);
        }
        else { // must find first index such that neighbours[i].weight > rho
            log::info!("get_scale_from_neighbourhood : point with rho_median  {} <= mean local scale {} , point far apart others", rho_median, mean_rho);
            let mut rank =  nbgh_2;
            for j in nbgh_2..nbgh {
                if neighbours[j].weight.to_f32().unwrap() > rho {
                    rank = j;
                    break;
                }
            }
            assert!(rank < nbgh);
            scale = neighbours[rank].weight.to_f32().unwrap();
        }
        // now we set scale so that k/2 neighbour is at proba 1/2 ?
        // reuse rho_y_s to return proba of edge
        for i in 0..nbgh {
            let rho_xi = neighbours[i].weight.to_f32().unwrap();
            rho_y_s[i] = (- (rho_xi- rho) * scale).exp(); 
        }
        rho_y_s.truncate(nbgh);        
        (mean_rho, rho_y_s)
    }  // end of get_scale_from_neighbourhood





    // choose scale to satisfy a normalization constraint. 
    // p_i = exp[- beta * (d(x,y_i)/ local_scale) ] 
    // We return beta/local_scale
    // as function is monotonic with respect to scale, we use dichotomy.
    fn get_scale_from_normalisation(&self, norm : f64 , neighbours : &Vec<OutEdge<F>>)  -> (f32, Vec::<f32>) {
      // p_i = exp[- beta * (d(x,y_i)/ local_scale) ] 
        let nbgh = neighbours.len();
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut rho_y_s = Vec::<f32>::with_capacity(neighbours.len() + 1);
        rho_y_s.push(rho_x);
        for i in 0..nbgh {
            let y_i = neighbours[i].node;      // y_i is a NodeIx = usize
            rho_y_s.push(self.kgraph.neighbours[y_i][0].weight.to_f32().unwrap());
            // we rho_x, scales
        }  // end of for i
        // if rho_x > mean_rho distance from x to its neighbour will be penalized and first term will not be 1
        // as is the case if rho_x < mean_rho
        // we must have a rho that is dependant on neighbourhood and less than rho_x, that implies much.
        let mean_rho = rho_y_s.iter().sum::<f32>() / (rho_y_s.len() as f32);
        // question cold be:  far is mean-rho from rho_min
        // now we set scale so that  ∑ p_{i} = norm 
        // for beta = 0 sum is nbgh and for β = infinity,  sum is 0. If norm is not between nbgh and 0 we have an error, else
        // as ∑ p_{i} is decreasing with respect to beta we dichotomize
        // normalize dists before dichotomy solver to make it more stable
        let dist = neighbours.iter().map( |n| n.weight.to_f32().unwrap()/mean_rho).collect::<Vec<f32>>();
        let f  = |beta : f32|  { dist.iter().map(|d| (-d * beta).exp()).sum::<f32>() };
        // f is decreasing
        let beta = dichotomy_solver(false, f, 0f32, f32::MAX, norm as f32);
        // TODO get quantile info on beta or corresponding entropy ?
        // reuse rho_y_s to return proba of edge
        for i in 0..nbgh {
            rho_y_s[i] = (- dist[i] * beta).exp(); 
        }
        rho_y_s.truncate(nbgh);
        // in this state neither sum of proba adds up to 1 neither is any entropy (Shannon or Renyi) normailed.
        (mean_rho, rho_y_s)
    } // end of get_scale_from_normalisation



} // end of impl Emmbedder



fn dichotomy_solver<F>(increasing : bool, f : F, lower_r : f32 , upper_r : f32, target : f32) -> f32 
            where F : Fn(f32) -> f32 {
    //
    if lower_r >= upper_r {
        panic!("dichotomy_solver failure low {} greater than upper {} ", lower_r, upper_r);
    }
    let range_low = f(lower_r).max(f(upper_r));
    let range_upper = f(upper_r).min(f(lower_r));
    if f(lower_r).max(f(upper_r)) < target || f(upper_r).min(f(lower_r)) > target {
        panic!("dichotomy_solver target not in range of function range {}  {} ", range_low, range_upper);     
    }
    // 
    if f(upper_r) < f(lower_r) && increasing {
        panic!("f not increasing")
    }
    else if f(upper_r) > f(lower_r) && !increasing {
        panic!("f not decreasing")
    }
    // target in range, proceed
    let mut middle = 1.;
    let mut upper = upper_r;
    let mut lower = lower_r;
    //
    let mut nbiter = 0;
    while (target-f(middle)).abs() > 1.0E-5 {
        if increasing {
            if f(middle) > target { upper = middle; }  else { lower = middle; }
        } // increasing type
        else { // decreasing case
            if f(middle) > target { lower = middle; }  else { upper = middle; }
        } // end decreasing type
        middle = (lower+upper)*0.5;
        nbiter += 1;
        if nbiter > 100 {
            panic!("dichotomy_solver do not converge, err :  {} ", (target-f(middle)).abs() );
        }
    }  // end of while
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
    
        let f = |x : f32 | {x*x};
        //
        let beta = dichotomy_solver(true, f, 0. , 5. , 2.);
        println!("beta : {}", beta);
        assert!( (beta - 2.0f32.sqrt()).abs() < 1.0E-4);
    }  // test_dichotomy_inc
    
    
    
    #[test]
    fn test_dichotomy_dec() {
    
        let f = |x : f32 | {1.0f32/ (x*x)};
        //
        let beta = dichotomy_solver(false, f, 0.2 , 5. , 1./2.);
        println!("beta : {}", beta);
        assert!( (beta - 2.0f32.sqrt()).abs() < 1.0E-4);
    }  // test_dichotomy_dec
    
    
    } // end of tests