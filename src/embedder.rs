//! umap-like embedding from GrapK

#![allow(dead_code)]

use num_traits::{Float};

use ndarray::{Dim, Array, Array1, Array2, Dimension};
use ndarray_linalg::{Scalar, Lapack};

use crate::tools::*;

use crate::fromhnsw::*;

pub struct Emmbedder <'a, F> {
    kgraph : &'a KGraph<F>, 
}

impl <F> Emmbedder<'_, F> 
    where F :  Float + Lapack + Scalar  + ndarray::ScalarOperand  {


    // this function compute a (generalised laplacian), do an approximated svd of it and project data on associated eigenvectors
    fn get_initial_embedding(&mut self) -> Array2<F> {
        let embedded : Array2<F>;
        //
        embedded
    }

    fn graph_symmetrization() {

    } // end of graph_symmetrization

    // minimize divergence between embedded and initial distribution probability
    fn entropy_optimize() {

    }


} // end of impl Emmbedder