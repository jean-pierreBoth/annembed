//! This module computes various entropy
//! 
//! 
//! 

#![allow(dead_code)]

use num_traits::{Float};
use core::ops::*;  // for  AddAssign + SubAssign + MulAssign + DivAssign + RemAssign 

/// A discrete probability
/// 
/// normalized set to true if normalization has been checked
/// 
pub struct DiscreteProba {
    p : Vec<f64>,
    entropy :Option<Vec<RenyiEntropy>>,
}

impl  DiscreteProba {

    pub fn new(p: &Vec<f64>) -> Self {
        let mut sum = 0f64;
        for x in p.iter() {
            if *x < 0. {
                log::error!("negative value in probability");
                std::panic!("negative value in probability");
            }
            else {
                sum +=x;
            }
        }
        let np = p.iter().map( |&x| x/sum ).collect();
        DiscreteProba { p: np , entropy : None}
    }
    



    /// compute for q = 1 it is Shannon entropy
    fn renyi_entropy_1(&self) -> f64 {
        let entropy = self.p.iter().map( |&v| if v > 0. { -v * v.ln() } else {0.}).sum();
        return entropy;
    }

    /// cmpute for q!= 1. 
    /// See Leinster. Entropy and diversity
    fn renyi_entropy_not_1(&self,  q: f64) -> f64 {
        let entropy : f64 = self.p.iter().map( |&v| if v > 0. { v.powf(q)} else {0.} ).sum();
        entropy.ln() / (1. - q)
    }


    /// compute Renyi entrpy of a for a vector of order
    pub fn renyi_entropy(&mut self, order: &[f64]) -> Vec<RenyiEntropy> {
        let mut entropy = Vec::<RenyiEntropy>::with_capacity(order.len());
        for q in order.iter() {
            if (*q-1.).abs() < 1.0E-5  {
                entropy.push(RenyiEntropy::new(1., self.renyi_entropy_1()));
            }
            else {
                entropy.push(RenyiEntropy::new(*q, self.renyi_entropy_not_1(*q)));
            }  
        }
        return entropy;
    } // end of entropy_renyi


    // compute relative entropy at q=1 
    fn relative_renyi_entropy_1(&self, other : &DiscreteProba) -> f64 {
        let entropy = self.p.iter().zip(other.p.iter()).map(|t| if *t.0 > 0. {*t.0 * (*t.1/ *t.0).ln() } else { 0.}).sum();
        entropy
    }

    //
    fn relative_renyi_entropy_q(&self, other : &DiscreteProba, q : f64) -> f64 {
        let entropy = self.p.iter().zip(other.p.iter()).map(|t| if *t.0 > 0. {*t.0 * (*t.1 / *t.0).powf(q) } else { 0.}).sum();
        entropy        
    }

    /// computes mean diversity of other with respect to self.
    /// ```math
    /// \sum_{i self.p_{i} != 0} self.p_{i} * \phi( other.p_{i}/ self.p_{i})
    /// ```
    pub fn relative_renyi_entropy(&self, other : &DiscreteProba, q: f64) ->f64 {
        if (q-1.).abs() < 1.0E-5 {
            return self.relative_renyi_entropy_1(other);
        } 
        else {
            return self.relative_renyi_entropy_q(other, q);
        }
    } // end of relative_entropy

} // end impl ProbaDistribution


fn near_to_1<F:Float>(f:F) -> bool {
    let one = num_traits::identities::one::<F>();
    let val = if (f - one).abs() < Float::epsilon() {
        true
    }
    else {
        false
    };
    val
}



    /// compute for q = 1 it is Shannon entropy
    fn renyi_entropy_gen<F:Float + AddAssign + SubAssign + MulAssign + DivAssign + std::iter::Sum>(p : Vec<F>) -> F {
        let entropy = p.iter().map( |&v| if v > F::zero() { -v * v.ln() } else {F::zero()}).sum();
        return entropy;
    }

//=================================================================================

#[derive(Clone,Copy, Debug)]
enum EntropyKind  {
    Renyi,
    Shannon,
}

///  Renyi entropy value
#[derive(Copy, Clone, Debug)]
pub struct RenyiEntropy {
    kind : EntropyKind,
    q : f64,
    value : f64,
}

impl RenyiEntropy {
    pub fn new(q:f64, value : f64) -> Self {
        RenyiEntropy{kind : EntropyKind::Renyi, q, value }
    }
}

//==================================================================================


/// Probabilistic neigheibourhood
/// If y1 ,   , yn are the points in neigbourhood aroud y0
/// lambda * d()

pub struct ProbaNeighbour {
    point_id : u32,
    /// list of neighbour's point_id
    neighbours_id : Vec<u32>,
    /// distance to each neighbour
    distance : Vec<f64>,
    /// probability transition deduced from distances
    proba : DiscreteProba,
    // rescale coefficient 
    lambda : f64
} // end of ProbaNeighbour


impl ProbaNeighbour {

}



//================================================================================  

#[cfg(test)]
mod tests {

    use rand::prelude::*;
    use super::*;

    use rand::distributions::{Distribution,Uniform};
    use rand_xoshiro::Xoshiro256PlusPlus;

#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]

fn test_proba_ann() {
    let unif_01 = Uniform::<f64>::new(0., 1.);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567 as u64);
    let p : Vec<f64>= (0..50).into_iter().map(|_| unif_01.sample(&mut rng)).collect();
    let mut proba = DiscreteProba::new(&p);
    let _entropy = proba.renyi_entropy(&[1., 2.]);

 
} // end of test_proba_ann



}