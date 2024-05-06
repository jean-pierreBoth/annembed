//! Definition of graph projection on less densley populated layers of a Hnsw structure.
//! This can be used in hierarchical initialization of the embedding.
//!
//! We extract from the Hnw structure a graph as in [KGraph](../KGraph) but
//! also construct a smaller graph by considering only upper layers of the Hnsw structure.  
//! Links from each point of the lower (denser) layers to the smaller graph are stored
//! so we can infer how well the smaller graph represent the whole graph.
//!

use anyhow::anyhow;

use num_traits::cast::FromPrimitive;
use num_traits::Float;

// to dump to ripser
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::path::Path;

use std::cmp::Ordering;
use std::sync::Arc;

use indexmap::set::*;
use quantiles::ckms::CKMS;
use std::collections::HashMap; // we could use also greenwald_khanna

use hnsw_rs::prelude::*;

use super::kgraph::*;
use crate::tools::nodeparam::*;

/// Construct a projection Graph from Hnsw data on layers above a given layers.  
/// Maintain for each point in the Hnsw structure nearest point in projected structure.
/// Possibly stores matrix of distances between filtered points
///
pub struct KGraphProjection<F> {
    /// we consider projection on points on layers above and including this layer
    layer: usize,
    /// graph on which we project
    small_graph: KGraph<F>,
    /// for each data out of the filtered data, we keep an edge to nearest data in the filtered set
    proj_data: HashMap<NodeIdx, OutEdge<F>>,
    /// larger graph that is projected
    large_graph: KGraph<F>,
} // end of struct Projection<F>

impl<F> KGraphProjection<F>
where
    F: Float + FromPrimitive + Send + Sync + std::fmt::UpperExp + std::iter::Sum,
{
    //  - first we construct graph consisting in upper (less populated) layers
    //  - Then we project : for points of others layers store the shorter edge from point to graph just constructed
    //  - at last we construct graph for the lower (more populated layers)
    //
    /// construct graph from layers above layer, projects data of another layers on point in layers above layer arg
    /// nbng is the maximum number of neighours to keep. It should be comparable to
    /// the parameter *max_nb_conn* used in the Hnsw structure.
    pub fn new<T, D>(hnsw: &Hnsw<T, D>, nbng: usize, layer: usize) -> Self
    where
        T: Clone + Send + Sync,
        D: Distance<T> + Send + Sync,
    {
        log::debug!("KGraphProjection new  layer : {}", layer);
        let mut nb_point_to_collect = 0;
        let mut nb_point_below_nbng: usize = 0;
        let mut mean_deficient_neighbour_size: usize = 0;
        let max_nb_conn = hnsw.get_max_nb_connection() as usize;
        let max_level_observed = hnsw.get_max_level_observed() as usize;
        log::debug!("max level observed : {}", max_level_observed);
        // check number of points kept in
        if layer >= max_level_observed {
            log::error!(
                "KGraphProjection::new, layer argument greater than nb_layer!!, layer : {}",
                layer
            );
            println!(
                "KGraphProjection::new, layer argument greater than nb_layer!!, layer : {}",
                layer
            );
        }
        for l in (layer..=max_level_observed).rev() {
            nb_point_to_collect += hnsw.get_point_indexation().get_layer_nb_point(l);
            log::trace!(
                " layer : {}, nb points to collect : {}",
                l,
                nb_point_to_collect
            );
        }
        if nb_point_to_collect <= 0 {
            log::error!("!!!!!!!!!!!! KGraphProjection cannot collect points !!!!!!!!!!!!!, check layer argument");
            println!("!!!!!!!!!!!! KGraphProjection cannot collect points !!!!!!!!!!!!!, check layer argument");
            std::process::exit(1);
        }
        //
        let layer_u8 = layer as u8;
        log::debug!(
            "Projection : number of point to collect : {}",
            nb_point_to_collect
        );
        let mut upper_graph_neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point_to_collect);
        for _i in 0..nb_point_to_collect {
            upper_graph_neighbours.push(Vec::<OutEdge<F>>::new());
        }
        let mut index_set = IndexSet::<DataId>::with_capacity(nb_point_to_collect);
        let mut _nb_point_upper_graph = 0;
        // We want to construct an indexation of nodes valid for the whole graph and for the upper layers.
        // So we begin indexation from upper layers
        for l in (layer..=max_level_observed).rev() {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                let (_, _) = index_set.insert_full(point.get_origin_id());
            }
        }
        // now we have an indexation of upper levels (less populatated levels), we clone present state of index_set
        let mut upper_index_set = index_set.clone();
        // we now continue to have the whole indexation
        for l in (0..layer).rev() {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                let (_, _) = index_set.insert_full(point.get_origin_id());
            }
        }
        // now we have the whole indexation, we construct upper(less populated) graph
        for l in (layer..=max_level_observed).rev() {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            //
            while let Some(point) = layer_iter.next() {
                // now point is an Arc<Point<F>>
                // point_id , we must reindex point_id for ProjectedDistances structure
                let origin_id = point.get_origin_id();
                let p_id = point.get_point_id();
                // remap _point_id
                let (index, newindex) = upper_index_set.insert_full(origin_id);
                assert_eq!(newindex, false);
                let neighbours_hnsw = point.get_neighborhood_id();
                // get neighbours of point in the same layer
                let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn);
                for m in layer..=max_level_observed {
                    for j in 0..neighbours_hnsw[m].len() {
                        let n_origin_id = neighbours_hnsw[m][j].get_origin_id();
                        let n_p_id = neighbours_hnsw[m][j].p_id;
                        if n_p_id.0 as usize >= layer {
                            // get index of !!! (and not get) IndexMap interface is error prone
                            let neighbour_idx = upper_index_set.get_index_of(&n_origin_id).unwrap();
                            vec_tmp.push(OutEdge::<F> {
                                node: neighbour_idx,
                                weight: F::from_f32(neighbours_hnsw[m][j].distance).unwrap(),
                            });
                        }
                    } // end of for j
                } // end of for m
                vec_tmp.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
                assert!(vec_tmp.len() <= 1 || vec_tmp[0].weight <= vec_tmp[1].weight); // temporary , check we did not invert order
                                                                                       // keep only the asked size. Could we keep more ?
                if vec_tmp.len() < nbng {
                    nb_point_below_nbng += 1;
                    mean_deficient_neighbour_size += vec_tmp.len();
                    log::trace!("neighbours must have {} neighbours, got only {}. layer {}  , pos in layer : {}", nbng, vec_tmp.len(),  p_id.0, p_id.1);
                    if vec_tmp.len() == 0 {
                        let p_id = point.get_point_id();
                        log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer : {} ", p_id.0, p_id.1);
                        continue;
                    }
                }
                vec_tmp.truncate(nbng);
                _nb_point_upper_graph += 1;
                // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
                upper_graph_neighbours[index] = vec_tmp;
                // TODO compute distance with preceding points. Fills distances ????????????
            } // end of while
        } // end loop on upper layers
          // at this step we have upper Graph (and could store distances between points if not too many points)
          // yet we do not have global projection mapping to NodeId of upper graph
          // so we need to loop on lower layers
        log::trace!(
            "number of points in upper graph : {} ",
            _nb_point_upper_graph
        );
        // now we construct projection
        let mut nb_point_without_projection = 0;
        let mut proj_data: HashMap<NodeIdx, OutEdge<F>> = HashMap::new();
        let mut points_with_no_projection = Vec::<Arc<Point<T>>>::new();
        for l in 0..layer {
            log::trace!("scanning projections of layer {}", l);
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                // we do as in KGraph.init_from_hnsw_layer
                let neighbours_hnsw = point.get_neighborhood_id();
                let best_distance = F::infinity();
                // get nearest point in upper layers.
                // possibly use a BinaryHeap?
                let mut best_edge = OutEdge::<F> {
                    node: usize::MAX,
                    weight: best_distance,
                };
                for m in layer..=max_level_observed {
                    for j in 0..neighbours_hnsw[m].len() {
                        let n_origin_id = neighbours_hnsw[m][j].get_origin_id();
                        let n_p_id = neighbours_hnsw[m][j].p_id;
                        if n_p_id.0 >= layer_u8
                            && F::from(neighbours_hnsw[m][j].distance).unwrap() < best_edge.weight
                        {
                            // store edge with remapping dataid to nodeidx
                            let neighbour_index =
                                upper_index_set.get_index_of(&n_origin_id).unwrap();
                            best_edge = OutEdge::<F> {
                                node: neighbour_index,
                                weight: F::from_f32(neighbours_hnsw[m][j].distance).unwrap(),
                            };
                        }
                    } // end of for j
                }
                // we have best edge, insert it in
                if best_edge.weight < F::infinity() && best_edge.node < usize::MAX {
                    let index = index_set.get_index_of(&point.get_origin_id()).unwrap();
                    proj_data.insert(index, best_edge);
                } else {
                    let p_id = point.get_point_id();
                    points_with_no_projection.push(Arc::clone(&point));
                    log::trace!(" no projection for point pid {}  {}", p_id.0, p_id.1);
                    nb_point_without_projection += 1;
                }
                // now we must
            } // end while
        } // end of search in densely populated layers
        log::info!(
            "number of points without edge defining a projection= {} ",
            nb_point_without_projection
        );
        let nb_projection_fixed = fix_points_with_no_projection(
            &mut proj_data,
            &index_set,
            &points_with_no_projection,
            layer,
        );
        log::info!(
            "number of points without edge defining a projection after fixing = {} ",
            nb_point_without_projection - nb_projection_fixed
        );
        //
        // define projection by identity on upper layers
        for l in layer..=max_level_observed {
            log::trace!("scanning projections of layer {}", l);
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            while let Some(point) = layer_iter.next() {
                let index = upper_index_set
                    .get_index_of(&point.get_origin_id())
                    .unwrap();
                let best_edge = OutEdge::<F> {
                    node: index,
                    weight: F::from_f32(0.).unwrap(),
                };
                proj_data.insert(index, best_edge);
            }
        }
        //
        log::info!(
            "\n\n number of points with less than : {} neighbours = {} ",
            nbng,
            nb_point_below_nbng
        );
        let upper_graph = KGraph::<F> {
            max_nbng: nbng,
            nbnodes: upper_graph_neighbours.len(),
            neighbours: upper_graph_neighbours,
            node_set: upper_index_set,
        };
        log::info!("\n\n getting stats from reduced graph : ");
        log::info!("\n ================================");
        let _graph_stats = upper_graph.get_kraph_stats();
        //
        // construct the dense whole  graph, lower layers or more populated layers
        //
        log::debug!(" constructing the whole graph");
        nb_point_to_collect = hnsw.get_nb_point();
        let mut graph_neighbours = Vec::<Vec<OutEdge<F>>>::with_capacity(nb_point_to_collect);
        for _i in 0..nb_point_to_collect {
            graph_neighbours.push(Vec::<OutEdge<F>>::new());
        }
        nb_point_below_nbng = 0;
        for l in 0..=max_level_observed {
            let mut layer_iter = hnsw.get_point_indexation().get_layer_iterator(l);
            //
            while let Some(point) = layer_iter.next() {
                let origin_id = point.get_origin_id();
                let index = index_set.get_index_of(&origin_id).unwrap();
                let neighbours_hnsw = point.get_neighborhood_id();
                // get neighbours of point in the same layer  possibly use a BinaryHeap?
                let mut vec_tmp = Vec::<OutEdge<F>>::with_capacity(max_nb_conn);
                for m in 0..layer {
                    for j in 0..neighbours_hnsw[m].len() {
                        let n_origin_id = neighbours_hnsw[m][j].get_origin_id();
                        // points are already indexed , or panic!
                        let neighbour_idx = index_set.get_index_of(&n_origin_id).unwrap();
                        vec_tmp.push(OutEdge::<F> {
                            node: neighbour_idx,
                            weight: F::from_f32(neighbours_hnsw[m][j].distance).unwrap(),
                        });
                    } // end of for j
                } // end of for m
                vec_tmp.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
                if vec_tmp.len() < nbng {
                    nb_point_below_nbng += 1;
                    mean_deficient_neighbour_size += vec_tmp.len();
                    let p_id = point.get_point_id();
                    log::trace!("neighbours must have {} neighbours, got only {}. layer {}  , pos in layer : {}", nbng, vec_tmp.len(),  p_id.0, p_id.1);
                    if vec_tmp.len() == 0 {
                        log::warn!(" graph will not be connected, isolated point at layer {}  , pos in layer : {} ", p_id.0, p_id.1);
                        continue;
                    }
                }
                vec_tmp.truncate(nbng);
                _nb_point_upper_graph += 1;
                // We insert neighborhood info at slot corresponding to index beccause we want to access points in coherence with neighbours referencing
                graph_neighbours[index] = vec_tmp;
            }
        } // end for on layers
          // we have both graph and projection
        let whole_graph = KGraph::<F> {
            max_nbng: nbng,
            nbnodes: graph_neighbours.len(),
            neighbours: graph_neighbours,
            node_set: index_set,
        };
        log::info!("\n \n getting stats from whole graph : ");
        log::info!("\n =====================================");
        let _graph_stats = whole_graph.get_kraph_stats();
        //
        if nb_point_below_nbng > 0 {
            log::info!("number of points with less than : {} neighbours = {},  mean size for deficient neighbourhhod {:.3e}", nbng, 
                nb_point_below_nbng, mean_deficient_neighbour_size as f64/nb_point_below_nbng as f64);
        }
        log::trace!("Projection exiting from new");
        //
        KGraphProjection {
            layer,
            small_graph: upper_graph,
            proj_data: proj_data,
            large_graph: whole_graph,
        }
    } // end of new

    /// get layer corresponding above which the projection is done. The layer is included in the projection.
    pub fn get_layer(&self) -> usize {
        self.layer
    } // end of get_layer

    /// returns the edge defining distance to nearest element in projected (small) graph.  
    /// The argument is a NodeIdx
    pub fn get_projection_by_nodeidx(&self, nodeidx: &NodeIdx) -> OutEdge<F> {
        *(self.proj_data.get(&nodeidx).unwrap())
    } // end of get_distance_to_projection

    /// returns the distance to nearest element in projected (small) graph
    /// The argument is a DataId
    pub fn get_distance_to_projection_by_dataid(&self, data_id: &DataId) -> F {
        let edge = self.proj_data.get(&data_id).unwrap();
        self.proj_data.get(&edge.node).unwrap().weight
    } // end of get_distance_to_projection

    /// return a reference to the small graph
    pub fn get_small_graph(&self) -> &KGraph<F> {
        &self.small_graph
    }

    /// returns a reference to the large graph
    pub fn get_large_graph(&self) -> &KGraph<F> {
        &self.large_graph
    }

    /// returns quantile stats on distances to projection point
    pub fn get_projection_distance_quant(&self) -> CKMS<f32> {
        let mut quant = CKMS::<f32>::new(0.001);
        for (_, edge) in self.proj_data.iter() {
            quant.insert(F::to_f32(&edge.weight).unwrap());
        }
        //
        quant
    }

    /// dump csc matrix of distances between pointsfrom projected graph.
    pub fn dump_sparse_mat_for_ripser(&self, fname: &str) -> Result<(), anyhow::Error> {
        let path = Path::new(fname);
        log::debug!("in to_ripser_sparse_dist : fname : {}", path.display());
        let fileres = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path);
        let file;
        if fileres.is_ok() {
            file = fileres.unwrap();
        } else {
            return Err(anyhow!("could not open file : {}", path.display()));
        }
        //
        let mut bufwriter = BufWriter::new(file);
        let res = self.small_graph.to_ripser_sparse_dist(&mut bufwriter);
        // TODO must launch ripser either with crate run_script or by making ripser a library using cxx. or Ripserer.jl
        //
        return res;
    } // end of compute_approximated_barcodes
} // end of impl block

// search a projection among projection of neighbours
fn get_point_approximate_projection<T, F>(
    point: &Arc<Point<T>>,
    proj_data: &mut HashMap<NodeIdx, OutEdge<F>>,
    index_set: &IndexSet<DataId>,
    layer: usize,
) -> Option<OutEdge<F>>
where
    T: Clone + Send + Sync,
    F: Float,
{
    //
    let neighbours_hnsw = point.get_neighborhood_id();
    // search a neighbour with a projection
    for l in 0..=layer {
        let nbng_l = neighbours_hnsw[l].len();
        for j in 0..nbng_l {
            let n_origin_id = neighbours_hnsw[l][j].get_origin_id();
            // has this point a projection ?
            let neighbour_idx = index_set.get_index_of(&n_origin_id).unwrap();
            if let Some(proj) = proj_data.get(&neighbour_idx) {
                return Some(proj.clone());
            }
        }
    }
    return None;
} // end of get_approximate_projection

// hack to approximate projections for problem points. We take the projection of neighbour
fn fix_points_with_no_projection<T, F>(
    proj_data: &mut HashMap<NodeIdx, OutEdge<F>>,
    index_set: &IndexSet<DataId>,
    points: &Vec<Arc<Point<T>>>,
    layer: usize,
) -> usize
where
    T: Clone + Send + Sync,
    F: Float,
{
    //
    let nb_points_noproj = points.len();
    let mut nb_fixed = 0;
    log::debug!(
        "fix_points_with_no_projection, nb points : {} ",
        nb_points_noproj
    );
    //
    for point in points.iter() {
        if let Some(edge) = get_point_approximate_projection(point, proj_data, index_set, layer) {
            let origin_id = point.get_origin_id();
            let idx = index_set.get_index_of(&origin_id).unwrap();
            proj_data.insert(idx, edge);
            nb_fixed += 1;
        }
    } // end of for on points
      //
    log::trace!("fix_points_with_no_projection nb fixed {}", nb_fixed);
    //
    nb_fixed
} // end of fix_points_with_no_projection

#[cfg(test)]
mod tests {

    //    cargo test fromhnsw  -- --nocapture
    //    cargo test  fromhnsw::tests::test_graph_projection -- --nocapture
    //    RUST_LOG=annembed::fromhnsw=TRACE cargo test fromhnsw -- --nocapture

    use super::*;

    use rand::distributions::Uniform;
    use rand::prelude::*;

    #[cfg(test)]
    fn log_init_test() {
        let res = env_logger::builder().is_test(true).try_init();
        if res.is_err() {
            println!("could not init log");
        }
    } // end of log_init_test

    fn gen_rand_data_f32(nb_elem: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::<Vec<f32>>::with_capacity(nb_elem);
        let mut rng = thread_rng();
        let unif = Uniform::<f32>::new(0., 1.);
        for i in 0..nb_elem {
            let val = 10. * i as f32 * rng.sample(unif);
            let v: Vec<f32> = (0..dim)
                .into_iter()
                .map(|_| val * rng.sample(unif))
                .collect();
            data.push(v);
        }
        data
    }

    #[test]
    fn test_graph_projection() {
        log_init_test();
        //
        let nb_elem = 80000;
        let dim = 30;
        let knbn = 10;
        //
        println!("\n\n test_graph_projection nb_elem {:?}", nb_elem);
        //
        let data = gen_rand_data_f32(nb_elem, dim);
        let data_with_id: Vec<(&Vec<f32>, usize)> = data.iter().zip(0..data.len()).collect();

        let ef_c = 50;
        let layer = 1;
        let max_nb_connection = 64;
        let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
        let mut hns =
            Hnsw::<f32, DistL1>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL1 {});
        // to enforce the asked number of neighbour
        hns.set_keeping_pruned(true);
        hns.parallel_insert(&data_with_id);
        hns.dump_layer_info();
        //
        let _graph_projection = KGraphProjection::<f32>::new(&hns, knbn, layer);
    } // end of test_graph_projection
} // end of mod tests
