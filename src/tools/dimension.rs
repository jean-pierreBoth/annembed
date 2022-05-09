//! dimension estimation

use anyhow::{anyhow};

use num_traits::{Float};
use crate::tools::nodeparam::*;


/// We implement the method described in :  
///     Maximum likelyhood estimation of intrinsic dimension.
///     Levina E. and Bickel P.J NIPS 2004.  [Levina-Bickel](https://www.stat.berkeley.edu/~bickel/mldim.pdf)
/// 
pub(crate) fn intrinsic_dimension_from_edges<F>(edges : &Vec<OutEdge<F>>) ->  Result<f64,anyhow::Error> 
                where F : Float  {
    let k_first: usize;
    let k_last : usize;
    if edges.len() >= 20 {
        // bickel use 10..20 as default
        k_first = 8;
        k_last = 19;
    }
    else if edges.len() > 3 {
        k_last = edges.len() - 1;
        k_first = 2;
    }
    else {
        log::error!("intrinsic_dimension_from_edges not enough edges");
        return Err(anyhow!("not enough neighbours"));
    }
    //
    let mut density : f64 = 0.;
    let d_estimate = |k| {
        let mut aux = 0.;
        for j in 1..k {
            if edges[j].weight.to_f64().unwrap() <= 0. || edges[k].weight.to_f64().unwrap() <= 0. {
                log::error!("null distances {:.3e}, {:.3e}", edges[j].weight.to_f64().unwrap(),  edges[k].weight.to_f64().unwrap());
            }
            aux += (edges[k].weight.to_f64().unwrap()/edges[j].weight.to_f64().unwrap()).ln();
        }
        if aux <= 0. {
            // we must take care of case with equal distances that give 0!!
//            log::debug!("density_k : aux {:.3e} , k = {}", aux, k);
            return -1.;
        }
        else {
            let density_k = (k as f64 - 1.)/aux;
         //   log::debug!("density_k : {:.3e}", density_k);
            return density_k;
        }
    };
    let mut nb_pos : u32 = 0;
    for k in k_first..=k_last {
        let d = d_estimate(k);
        if d > 0. {
            density += d_estimate(k);
            nb_pos += 1;
        }
    }
    if nb_pos > 0 {
        density = density / nb_pos as f64;
//        log::debug!("intrinsic_dimension_from_edges k_first : {}, k_last : {}, density : {:.3e}", k_first, k_last, density);
        return Ok(density);
    }
    else {
        return Err(anyhow!("not positive distances"));
    }
} // end of intrinsic_dimension_from_edges
