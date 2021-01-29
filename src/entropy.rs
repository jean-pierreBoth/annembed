//! This module computes various entropy
//! 
//! 

/// A discrete probability
/// 
/// normalized set to true if normalization has been checked
/// 
pub struct ProbaAnn {
    normalized : bool,
    p : Vec<f64>,
    entropy :Option<Vec<RenyiEntropy>>,
}

impl  ProbaAnn {

    /// normalize to 1.
    fn normalize(&mut self) {
        let norm:f64 = self.p.iter().sum();
        for v in self.p.iter_mut() {
            *v /= norm;
        }
        self.normalized = true;
    }

    /// compute for q = 1 it is Shannon entropy
    fn compute_1(&self) -> f64 {
        let entropy = self.p.iter().map( |&v| if v > 0. { -v * v.ln() } else {0.}).sum();
        return entropy;
    }

    /// cmpute for q!= 1. 
    /// See Leinster. Entropy and diversity
    fn compute_not_1(&self,  q: f64) -> f64 {
        let entropy : f64 = self.p.iter().map( |&v| if v > 0. { v.powf(q)} else {0.} ).sum();
        entropy.ln() / (1. - q)
    }

    /// compute Renyi entrpy of a for a vector of order
    pub fn compute_renyi(&mut self, order: &Vec<f64>) -> Vec<RenyiEntropy> {
        let mut entropy = Vec::<RenyiEntropy>::with_capacity(order.len());
        for q in order.iter() {
            if *q == 1. {
                entropy.push(RenyiEntropy{q: 1., value: self.compute_1()});
            }
            else {
                entropy.push(RenyiEntropy{q: *q, value: self.compute_not_1(*q)});
            }  
        }
        return entropy;
    } // end of computeRenyi

} // end impl ProbaDistribution

//=================================================================================

///  Renyi entropy value
#[derive(Copy, Clone, Debug)]
pub struct RenyiEntropy {
    q : f64,
    value : f64,
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
    proba : ProbaAnn,
    // rescale coefficient 
    lambda : f64
} // end of ProbaNeighbour


impl ProbaNeighbour {

}