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
                let (scale, probas) = self.get_scale_from_proba_normalisation(&neighbour_hood[i]);
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





    // choose scale to satisfy a normalization constraint. 
    // p_i = exp[- beta * (d(x,y_i)/ local_scale) ] 
    // We return beta/local_scale
    // as function is monotonic with respect to scale, we use dichotomy.
    fn get_scale_from_umap(&self, norm : f64 , neighbours : &Vec<OutEdge<F>>)  -> (f32, Vec::<f32>) {
      // p_i = exp[- beta * (d(x,y_i)/ local_scale) ] 
        let nbgh = neighbours.len();
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut dist = neighbours.iter().map( |n| n.weight.to_f32().unwrap()).collect::<Vec<f32>>();
        let f  = |beta : f32|  { dist.iter().map(|d| (-(d-rho_x) * beta).exp()).sum::<f32>() };
        // f is decreasing 
        // TODO we could also normalize as usual? 
        let beta = dichotomy_solver(false, f, 0f32, f32::MAX, norm as f32);
        // TODO get quantile info on beta or corresponding entropy ? β should be not far from 1?
        // reuse rho_y_s to return proba of edge
        for i in 0..nbgh {
            dist[i] = (- (dist[i]-rho_x) * beta).exp(); 
        }
        // in this state neither sum of proba adds up to 1 neither is any entropy (Shannon or Renyi) normailed.
        (1./beta, dist)
    } // end of get_scale_from_umap


    // Simplest function where we know really what we do and why. Get a normalized proba with constraint.
    // p_i = exp[- beta * (d(x,y_i)/ local_scale)]  and then normalized to 1.
    // local_scale can be adjusted so that ratio of last praba to first proba >= epsil.
    fn get_scale_from_proba_normalisation(&self, neighbours : &Vec<OutEdge<F>>)  -> (f32, Vec::<f32>) {
        // p_i = exp[- beta * (d(x,y_i)/ local_scale) * lambda] 
        let epsil : f32 = 1.0E-5;
        let nbgh = neighbours.len();
        // determnine mean distance to nearest neighbour at local scale
        let rho_x = neighbours[0].weight.to_f32().unwrap();
        let mut rho_y_s = Vec::<f32>::with_capacity(neighbours.len() + 1);
        for i in 0..nbgh {
            let y_i = neighbours[i].node;      // y_i is a NodeIx = usize
            rho_y_s.push(self.kgraph.neighbours[y_i][0].weight.to_f32().unwrap());
            // we rho_x, scales
        }  // end of for i
        rho_y_s.push(rho_x);
        let mut mean_rho = rho_y_s.iter().sum::<f32>() / (rho_y_s.len() as f32);
        // now we adjust mean_rho so that the ratio of proba of last neighbour to first neighbour do not exceed epsil.
        let first_dist = neighbours[0].weight.to_f32().unwrap();
        let last_dist = neighbours[nbgh-1].weight.to_f32().unwrap();
        if  last_dist > first_dist {
            let lambda = (last_dist-first_dist)/(mean_rho * (-epsil.ln()));
            if lambda > 1. {
                // we rescale mean_rho to avoid too large range of probabilities in neighbours.
                mean_rho = mean_rho*lambda;
            }
            let mut probas = neighbours.iter().map( |n| (-n.weight.to_f32().unwrap()/mean_rho).exp()).collect::<Vec<f32>>();
            assert!(probas[probas.len()-1]/probas[0] <= epsil);
            let sum = probas.iter().sum::<f32>();
            for i in 0..nbgh {
                probas[i] = probas[i]/sum; 
            }
            return (mean_rho, probas);
        } else {
            // all neighbours are at the same distance!
            let probas = neighbours.iter().map( |_| (1.0 / nbgh as f32)).collect::<Vec<f32>>();
            return (mean_rho, probas)
        }
    } // end of get_scale_from_proba_normalisation
  
  
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