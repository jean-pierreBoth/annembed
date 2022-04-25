//! dimension estimation

use anyhow::{anyhow};

use num_traits::{Float};
use crate::tools::nodeparam::*;


/// We implement the method described in :  
///     Maximum likelyhood estimation of intrinsic dimension.
///     Levina E. and Bickel P.J NIPS 2004.  <https://www.stat.berkeley.edu/~bickel/mldim.pdf>
/// 
pub fn intrinsic_dimension_from_edges<F>(edges : &Vec<OutEdge<F>>) ->  Result<f64,anyhow::Error> 
                where F : Float  {
    let k_first: usize;
    let k_last : usize;
    if edges.len() > 20 {
        // bickel use 10..20 as default
        k_first = 10;
        k_last = 20;
    }
    else if edges.len() > 3 {
        k_last = edges.len() - 1;
        k_first = 2;
    }
    else {
        return Err(anyhow!("not enough neighbours"));
    }
    //
    log::debug!("intrinsic_dim, k_first :{},  k_last : {}", k_first, k_last);
    let mut density : f64 = 0.;
    let d_estimate = |k| {
        let mut aux = 0.;
        for j in 1..k {
            if edges[j].weight.to_f64().unwrap() <= 0. || edges[k].weight.to_f64().unwrap() <= 0. {
                log::error!("null distances {:.3e}, {:.3e}", edges[j].weight.to_f64().unwrap(),  edges[k].weight.to_f64().unwrap());
            }
            aux += (edges[k].weight.to_f64().unwrap()/edges[j].weight.to_f64().unwrap()).ln();
        }
        let density_k = (k as f64 - 1.)/aux;
        log::debug!("density_k : {:.3e}", density_k);
        return density_k;
    };
    for k in k_first..=k_last {
        density += d_estimate(k);
    }
    Ok(density / (k_last - k_first + 1) as f64)
} // end of intrinsic_dimension_from_edges
