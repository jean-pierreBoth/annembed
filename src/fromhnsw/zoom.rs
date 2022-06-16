//! This modules extracts a neighborhood around a point from the hnsw structure
//! Given a point in the upper structure of Hnsw we recursively collect its neighbours down to the lower layers.
//! The objective is to get homology characterization of neighbourhood of a point.
//! The neighbourhood should not contain too many points as the distance between points will be needed.  
//! To get a larger view use the kgproj module.


/// As we will need to compute distances between points in the neighbourhood we need to retrieve
/// the original data inside a Point<T>, hence the PointIndexation.get_point() function in crate hnsw.

 
use anyhow::{anyhow};

use num_traits::{Float};
use num_traits::cast::FromPrimitive;



//use std::cmp::Ordering;

use std::collections::{VecDeque, HashMap};

// use indexmap::set::*;

#[allow(unused)]
use ndarray::{Array2};


use hnsw_rs::prelude::*;
use hnsw_rs::hnsw::{DataId};


/// The zoom can be extracted from a Vec<T> or 
pub struct Zoom<F:Float + FromPrimitive + Clone+Send+Sync> {
    /// center of extraction
    center : Vec<F>,
    /// optional cutoff
    cutoff : Option<f64>,
    /// max dit encountered
    max_dist : f64,
} // end of Zoom


impl <F> Zoom<F>
    where F : Float + FromPrimitive + Clone + Send + Sync + std::fmt::UpperExp {

    pub fn new<D>(hnsw : &Hnsw<F,D>, center : &Vec<F>)  -> Self  
        where   D: Distance<F> + Send + Sync {
        //
        log::debug!("entering Zoom::new");
        // at what layer is point?
        let nearest = hnsw.search(&center, 3, 50);
        let p_id = nearest[0].p_id;
        log::debug!("Zoom::new nearest at layer {:?}, distance : {} , next dist {} ", p_id, nearest[0].distance, nearest[1].distance);
        let point_indexation = hnsw.get_point_indexation();
        let nearest_point= point_indexation.get_point(&nearest[0].p_id).unwrap();
        // get parent of nearest point if possible
        let nearest_point_neighbour = nearest_point.get_neighborhood_id();
        let from_layer = if p_id.0 < point_indexation.get_max_level_observed().min(2) {
            p_id.0 as usize  + 1
        }
        else {
            p_id.0 as usize
        };
        let mut max_dist = 0.;
        let distance_f = hnsw.get_distance();
        log::debug!("setting from layer to {}", from_layer);
        let upper = nearest_point_neighbour[from_layer][0];
        let from_point = point_indexation.get_point(&upper.p_id).unwrap();
        let threshold_dist =  distance_f.eval(center, from_point.get_v());
        log::debug!("upper dist to center : {}", threshold_dist);
        // now we examine what is around from_point and below
        let mut hash = HashMap::<DataId, Vec<F>>::new();
        let mut encountered = VecDeque::<PointId>::new();
        encountered.push_back(from_point.get_point_id());
        while let Some(p_id) = encountered.pop_back() {
            log::debug!("popping p_id : {:?}", p_id);
            // push into encountered neighbours at lower layer.
            let layer = p_id.0 as usize; 
            // treat neighbours at same layer.
            let neighbours = from_point.get_neighborhood_id();
            for n in &neighbours[layer] {
                // now we loop on neighbours at layer l
                if let Some(data) = point_indexation.get_point(&n.p_id) {
                    // check for possible threshold
                    let new_dist = distance_f.eval(data.get_v(), center);
                    if new_dist > max_dist {
                        max_dist = max_dist.max(new_dist); 
                    }   
                    log::debug!("new_dist : {}, max_dist : {} p_id {:?}", new_dist, max_dist, n.p_id);
                    if max_dist > 1.0E5 {
                        panic!("too large");
                    }
                    hash.insert(n.get_origin_id(), data.get_v().to_vec());
                }
                else {
                    log::error!("PointIndexation::get_point failed at {:?}", n.p_id);
                    panic!("PointIndexation::get_point failed");
                }
            }
            // lower layers are stored in the encountered stack
            for l in 0..layer {
                for n in &neighbours[l] {
                    // now we loop on neighbours at layer l
                    log::debug!("pushing in stack {:?}", n.p_id);
                    encountered.push_back(n.p_id);
                }                
            }
            log::debug!("encountered size : {}", encountered.len());
            if log::log_enabled!(log::Level::Debug) && hash.len() % 50 == 0 {
                log::debug!(" zoom size : {}, queue size : {}", hash.len(), encountered.len());
            }
        } // end of while
        // one we have the layer from which we search we explore around and iterate downwards.

        log::debug!("end of while suceeded, nbpoint {}, max distance {}", hash.len(), max_dist);
        
        Zoom{ center : center.clone(), cutoff: None, max_dist : max_dist as f64}
    } // end of new
} // end of impl block for Zoom


#[cfg(test)]
mod tests {

//    cargo test fromhnsw  -- --nocapture
//    cargo test  fromhnsw::tests::test_graph_projection -- --nocapture
//    RUST_LOG=annembed::fromhnsw=TRACE cargo test fromhnsw -- --nocapture

use super::*;


use rand::distributions::{Uniform};
use rand::prelude::*;

#[cfg(test)]
fn log_init_test() {
    let res = env_logger::builder().is_test(true).try_init();
    if res.is_err() {
        println!("could not init log");
    }
}  // end of log_init_test



fn gen_rand_data_f32(nb_elem: usize , dim:usize) -> Vec<Vec<f32>> {
    let mut data = Vec::<Vec<f32>>::with_capacity(nb_elem);
    let mut rng = thread_rng();
    let unif =  Uniform::<f32>::new(0.,1.);
    for i in 0..nb_elem {
        let val = 10. * i as f32 * rng.sample(unif);
        let v :Vec<f32> = (0..dim).into_iter().map(|_|  val * rng.sample(unif)).collect();
        data.push(v);
    }
    data
} // end of gen_rand_data_f32


#[test]
fn test_rand_zoom() {
    log_init_test();
    //
    let nb_elem = 10000;
    let dim = 30;
    let knbn = 10;
    //
    println!("\n\n test_rand_zoom nb_elem {:?}", nb_elem);
    //
    let data = gen_rand_data_f32(nb_elem, dim);
    let data_with_id : Vec<(&Vec<f32>, usize)> = data.iter().zip(0..data.len()).collect();

    let ef_c = 50;
    let max_nb_connection = 64;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let mut hns = Hnsw::<f32, DistL1>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL1{});
    // to enforce the asked number of neighbour
    hns.set_keeping_pruned(true);
    hns.parallel_insert(&data_with_id);
    hns.dump_layer_info();
    //
    let center = data_with_id[0].0;
    let zoom = Zoom::<f32>::new(&hns, center);
}  // end of test_rand_zoom

}  // end of tests