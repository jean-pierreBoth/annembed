//! This modules extracts a neighborhood around a point from the hnsw structure
//! Given a point in the upper structure of Hnsw we recursively collect its neighbours down to the lower layers.
//! The objective is to get homology characterization of neighbourhood of a point.

use anyhow::{anyhow};

use num_traits::{Float};
use num_traits::cast::FromPrimitive;



//use std::cmp::Ordering;

use std::collections::HashMap;
// use indexmap::set::*;

#[allow(unused)]
use ndarray::{Array2};


use hnsw_rs::prelude::*;
use hnsw_rs::hnsw::{DataId};


/// The zoom can be extracted from a Vec<T> or 
pub struct Zoom<F:Float + FromPrimitive + Clone+Send+Sync> {
    /// center of extraction
    center : Point<F>,
    /// number of hops inside one layer before going down. For now 1.
    nbhop : usize,
    /// optional cutoff
    cutoff : Option<f64>,
    /// max dit encountered
    max_dist : f64,
} // end of Zoom


impl <F> Zoom<F>
    where F : Float + FromPrimitive + Clone + Send + Sync + std::fmt::UpperExp {

    pub fn new<T,D>(hnsw : &Hnsw<T,D>, center : &Point<T>, nbhop : usize)  -> Self  
        where   T: Clone + Send + Sync,
                D: Distance<T> + Send + Sync {
        //
        log::debug!("entering Zoom::new");
        
        // at what layer is point?
        // if layer is too low we must go up to begin exploration
        
        // one we have the layer from which we search we explore around and iterate downwards.
        panic!("not yet");
    } // end of new
} // end of impl block for Zoom