//! Data extraction from a Hnsw structure to be processed by Julia Ripserer package with function
//! in crate/Julia/toripserer.jl
//!
//! - We extract points in a neighbourhood around a center point of interest and dump in bson format
//!  the distance matrix between points to be processed by Julia function localPersistency
//!
//! - We try to compute global homology from projection of data on less populated layers. We dump
//!  a csc matrix of projected cloud points to upper layers and dump the Csc matrix to be processed by Julia function projectedPersistency
//!
//!

use anyhow::anyhow;

// to dump to ripser
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::path::Path;

use ndarray::Array2;

use hnsw_rs::prelude::*;
//use hnsw_rs::hnsw::{DataId};

use super::kgproj::*;

pub struct ToRipserer<'a, 'b, T, D>
where
    T: Clone + Send + Sync + 'b,
    D: Distance<T> + Send + Sync,
{
    hnsw: &'a Hnsw<'b, T, D>,
} // end of ToRipserer

impl<'a, 'b, T, D> ToRipserer<'a, 'b, T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    pub fn new(hnsw: &'a Hnsw<'b, T, D>) -> Self {
        ToRipserer { hnsw: hnsw }
    }

    /// extract *knbn* neighbours around *center* and dump lower triangulal distance matrix
    /// in bson format for analysis with Julia Ripserer.  
    /// knbn can reasonably go up to 1000 points. On a i7 @2.3 Ghz laptop using one core, Ripserer will compute H0,H1 in 0.5s
    /// and H0,H1 and H2 in 55s. With 2000 points we get H0 and H1 in 22s
    pub fn extract_neighbourhood(
        &self,
        center: &Vec<T>,
        knbn: usize,
        ef_c: usize,
        outbson: &String,
    ) -> Result<(), anyhow::Error> {
        // search neighbours
        let point_indexation = self.hnsw.get_point_indexation();
        log::debug!("extract_neighbourhood asking for {} points", knbn);
        let neighbour_0 = self.hnsw.search(&center, knbn, ef_c);
        let nb_points = neighbour_0.len();
        log::debug!("got nb neighbours : {}", nb_points);
        let mut dist_mat = Array2::zeros((nb_points, nb_points));
        let distance = self.hnsw.get_distance();
        for i in 0..neighbour_0.len() {
            let data_i = point_indexation
                .get_point_data(&neighbour_0[i].p_id)
                .unwrap();
            for j in 0..i {
                let point_j = point_indexation
                    .get_point_data(&neighbour_0[j].p_id)
                    .unwrap();
                dist_mat[[i, j]] = distance.eval(&data_i, &point_j);
            }
        } // end of for i
        let vlen = nb_points * (nb_points + 1) / 2;
        let mut v = Vec::<f32>::with_capacity(vlen);
        for i in 0..neighbour_0.len() {
            for j in 0..i {
                v.push(dist_mat[[i, j]]);
            }
            v.push(0.);
        }
        // dump full matrix distance in lower triangular format
        log::debug!(
            "serializing , first values : {} {} {} , last : {}",
            v[0],
            v[1],
            v[2],
            v[v.len() - 2]
        );
        let serializer = bson::Serializer::new();
        let res = serde::ser::Serialize::serialize::<bson::Serializer>(&v, serializer);
        if !res.is_ok() {
            log::error!("serialization of distance matrix failed");
            return Err(anyhow!("serialization of distance matrix failed"));
        }
        let bson_v = res.unwrap();
        let path = Path::new(outbson);
        let fileres = OpenOptions::new().write(true).create(true).open(path);
        let file;
        if fileres.is_ok() {
            file = fileres.unwrap();
        } else {
            log::error!("could not open file : {}", path.display());
            return Err(anyhow!("could not open file : {}", path.display()));
        }
        let mut doc = bson::Document::new();
        doc.insert("limat", bson_v);
        let bufwriter = BufWriter::new(file);
        let res = doc.to_writer(bufwriter);
        if res.is_err() {
            log::error!("dump of bson failed: {}", res.clone().err().unwrap());
            return Err(anyhow!("dump of bson failed: {}", res.err().unwrap()));
        }
        //
        return Ok(());
    } // end of extract_neighbourhood

    /// A KGraphProjection is constructed from layers above (and including) argument *layer* from the hnsw structure.
    /// layer must be chosen so that there are not too many points for persistence computations to run
    /// but sufficiently so that every point in the full data is close to some point of the upper layers.
    /// The graph has knbn neighbours around each point and is stored as a CSC matrix.  
    ///     
    /// The quantiles on distances to projection is printed, the median can be considered as a measure of the quality of the projection.  
    ///
    /// Ripserer on CSC matrix is fast.
    /// On a i7 @2.3 Ghz laptop using one core Ripserer will compute H0,H1,H2 in 0.05s for a 1200*1200 with 18000 entries.  
    ///  
    /// We try to exploit the stability theorem of persistence diagrams as described in
    /// Nanda V. COMPUTATIONAL ALGEBRAIC TOPOLOGY Lecture Notes, paragraph 6.5. See [TDA](https://people.maths.ox.ac.uk/nanda/cat/TDANotes.pdf)
    ///
    pub fn extract_projection_to_ripserer(
        &self,
        knbn: usize,
        layer: usize,
        fname: &String,
    ) -> Result<(), anyhow::Error> {
        // construct a graph projection from layer
        let graph_projection = KGraphProjection::<f32>::new(&self.hnsw, knbn, layer);
        let quant = graph_projection.get_projection_distance_quant();
        if quant.count() > 0 {
            println!("\n\n projection distance from lower layers to upper layers");
            println!(
                "\n quantile at 0.05 : {:.2e} , 0.5 :  {:.2e}, 0.95 : {:.2e}, 0.99 : {:.2e}",
                quant.query(0.05).unwrap().1,
                quant.query(0.5).unwrap().1,
                quant.query(0.95).unwrap().1,
                quant.query(0.99).unwrap().1
            );
        }
        // testing output for ripser
        log::debug!("output fashion projection for ripser : {}", fname);
        let res = graph_projection.dump_sparse_mat_for_ripser(fname);
        if res.is_err() {
            log::error!("graph_projection dump_sparse_mat_for_ripser failed");
            return Err(anyhow!(
                "graph_projection dump_sparse_mat_for_ripser failed"
            ));
        }
        //
        return Ok(());
    } // end of extract_projection_to_ripserer
} // end of impl ToRipserer
