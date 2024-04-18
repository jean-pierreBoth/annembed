//! Estimate how many times each node is in neighbours of some node in graph.
//!
//! This is referred to as hubness of data.  
//! It is shown to be correlated to intrinsic dimension of data.
//!
//! A Reference on hubness is:
//! **Hubs in Space: Popular Nearest Neighbours in High Dimensional Data**  
//! *Radovanovic M., Nanopoulos A., Ivanovic I.. Journal Machine Learning 2010*
//! Cf [Hubs](https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf)
//!

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::atomic::{AtomicU32, Ordering};

use num_traits::cast::FromPrimitive;
use num_traits::Float;

use hdrhistogram::Histogram;
use indxvec::{Indices, Vecops};

use hnsw_rs::hnsw::DataId;

use super::kgraph::*;

pub struct Hubness<'a, F> {
    /// The graph we work for
    kgraph: &'a KGraph<F>,
    /// citation count in neighbourhoods for each node.
    counts: Vec<u32>,
} // end of Hubness

impl<'a, F> Hubness<'a, F>
where
    F: FromPrimitive + Float + std::fmt::UpperExp + Sync + Send + std::iter::Sum,
{
    pub fn new(kgraph: &'a KGraph<F>) -> Self {
        //
        let nb_nodes = kgraph.get_nb_nodes();
        let mut counts_atom = Vec::<AtomicU32>::with_capacity(nb_nodes);
        for _ in 0..nb_nodes {
            counts_atom.push(AtomicU32::new(0));
        }
        //
        let scan_node = |node: usize, counts_atom: &Vec<AtomicU32>| {
            let neighbours = kgraph.get_out_edges_by_idx(node);
            for edge in neighbours {
                let n = edge.node;
                // we increment hub count for n as it is cited in this edge
                // note fecth_add possible only on arch implementing atomic ops on u32
                counts_atom[n].fetch_add(1, Ordering::SeqCst);
            }
        };
        (0..nb_nodes)
            .into_par_iter()
            .for_each(|n| scan_node(n, &counts_atom));
        //
        let mut counts = Vec::<u32>::with_capacity(nb_nodes);
        for i in 0..nb_nodes {
            counts.push(counts_atom[i].load(Ordering::Relaxed));
        }
        //
        Hubness {
            kgraph: &kgraph,
            counts: counts,
        }
    } // end of new

    /// returns counts by index
    pub fn get_counts(&self) -> &Vec<u32> {
        &self.counts
    } // end of get_counts

    /// get standardized 3 moment of occurences (See Radovanovic paper cited above)
    /// [Hubs](https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf)
    pub fn get_standard3m(&self) -> f64 {
        //
        if self.counts.len() <= 1 {
            return 0.;
        }
        //
        let mu = self.counts.iter().sum::<u32>() as f64 / self.counts.len() as f64;
        //
        let mut sum2 = 0f64;
        let mut sum3 = 0.;
        let mut incr;
        for x in &self.counts {
            incr = (f64::from(*x) - mu) * (f64::from(*x) - mu);
            sum2 = sum2 + incr;
            sum3 = sum3 + incr * (f64::from(*x) - mu);
        }
        sum3 /= self.counts.len() as f64;
        let sigma = (sum2 / (self.counts.len() - 1) as f64).sqrt();
        let s3m = sum3 / sigma.powi(3);
        //
        return s3m;
    } // end of get_standard3m

    /// get an histogram of hubness counts and prints histogram summary
    /// quantiles for which thresholds are given are :  
    /// 0.1, 0.25, 0.5, 0.75, 0.9 , 0.99, 0.999, 0.9999
    pub fn get_hubness_histogram(&self) -> Result<Histogram<u32>, anyhow::Error> {
        // record histogram length from 1 to readmaxsize with slot of size readmaxsize/10**prec
        // lowest value arg in init must be >= 1
        let max_value = 2 * (self.counts.len() as f64).sqrt() as u64;
        let prec = 1u32;
        assert!(
            prec >= 1,
            "precision for histogram construction should range >= 1"
        );
        let histo = Histogram::<u32>::new_with_bounds(1, max_value, 1);
        if histo.is_err() {
            log::error!(
                "hubness::get_hubness_histogram, could not create histogram , error : {:?}",
                histo.as_ref().err()
            );
            return Err(anyhow::anyhow!("histogram construction failed"));
        }
        let mut histo = histo.unwrap();
        let mut nb_out_histo = 0u32;
        for v in &self.counts {
            let res = histo.record(*v as u64);
            if res.is_err() {
                nb_out_histo += 1;
            }
        }
        // display result
        if nb_out_histo > 0 {
            println!(
                "number of too large values : {}, maximum value : {}",
                nb_out_histo, max_value
            );
        }
        let quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999];
        let thresholds = quantiles
            .iter()
            .map(|f| histo.value_at_quantile(*f))
            .collect::<Vec<u64>>();
        //
        println!("\n hubness quantiles : ");
        println!("======================");
        println!("quantiles : {:?}", quantiles);
        println!("thresholds : {:?}", thresholds);
        println!("\n");
        //
        Ok(histo)
    } // end of get_hubness_histogram

    /// get the DataId of the nodes having first largest hubness
    pub fn get_largest_hubs_by_dataid(&self, first_asked: usize) -> Vec<(DataId, usize)> {
        let first = first_asked.min(self.counts.len());
        //
        let mut hubs_dataid = Vec::<(DataId, usize)>::with_capacity(first);
        // we must get index of first largest counts, get descending order
        let ranks = self.counts.rank(false);
        let index = ranks.invindex();
        //
        if log::log_enabled!(log::Level::Debug) {
            for i in 0..first {
                let data_id = self.kgraph.get_data_id_from_idx(index[i]).unwrap();
                log::debug!(
                    "index : {} , rank : {}, count : {} , data_id : {:?}",
                    i,
                    index[i],
                    self.counts[index[i]],
                    data_id
                );
            }
        }
        log::info!("get_largest_hubs_by_dataid, 10 largest hubness ");
        for i in 0..first {
            let data_id = self.kgraph.get_data_id_from_idx(index[i]).unwrap();
            if i <= 10 {
                log::info!(
                    "index : {} , rank : {}, count : {} , data_id : {:?}",
                    i,
                    index[i],
                    self.counts[index[i]],
                    data_id
                );
            }
            hubs_dataid.push((*data_id, self.counts[index[i]] as usize));
        }
        //
        hubs_dataid
    } // end of get_largest_hubs_by_dataid

    /// returns the hubness of a given DataId
    pub fn get_dataid_hubness(&self, data_id: &DataId) -> usize {
        let index = self.kgraph.get_idx_from_dataid(data_id).unwrap();
        self.counts[index] as usize
    } // end of get_dataid_hubness
} // end of impl block for Hubness
