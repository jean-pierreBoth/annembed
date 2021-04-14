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
    fn entropy_1(&self) -> f64 {
        let entropy = self.p.iter().map( |&v| if v > 0. { -v * v.ln() } else {0.}).sum();
        return entropy;
    }

    /// cmpute for q!= 1. 
    /// See Leinster. Entropy and diversity
    fn entropy_not_1(&self,  q: f64) -> f64 {
        let entropy : f64 = self.p.iter().map( |&v| if v > 0. { v.powf(q)} else {0.} ).sum();
        entropy.ln() / (1. - q)
    }


    /// compute Renyi entrpy of a for a vector of order
    pub fn entropy_renyi(&mut self, order: &Vec<f64>) -> Vec<RenyiEntropy> {
        let mut entropy = Vec::<RenyiEntropy>::with_capacity(order.len());
        for q in order.iter() {
            if *q == 1. {
                entropy.push(RenyiEntropy{q: 1., value: self.entropy_1()});
            }
            else {
                entropy.push(RenyiEntropy{q: *q, value: self.entropy_not_1(*q)});
            }  
        }
        return entropy;
    } // end of entropy_renyi


    // compute relative entropy at q=1 
    fn relative_entropy_1(&self, other : &ProbaAnn) -> f64 {
        let entropy = self.p.iter().zip(other.p.iter()).map(|t| if *t.0 > 0. {*t.0 * (*t.1/ *t.0).ln() } else { 0.}).sum();
        entropy
    }

    //
    fn relative_entropy_q(&self, other : &ProbaAnn, q : f64) -> f64 {
        let entropy = self.p.iter().zip(other.p.iter()).map(|t| if *t.0 > 0. {*t.0 * (*t.1 / *t.0).powf(q) } else { 0.}).sum();
        entropy        
    }

    /// computes mean diversity of other with respect to self.
    /// ```math
    /// \sum_{i self.p_{i} != 0} self.p_{i} * \phi( other.p_{i}/ self.p_{i})
    /// ```
    pub fn relative_entropy(&self, other : &ProbaAnn, q: f64) ->f64 {
        if q == 1. {
            return self.relative_entropy_1(other);
        } 
        else {
            return self.relative_entropy_q(other, q);
        }
    } // end of relative_entropy

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



//================================================================================  

#[cfg(test)]
mod tests {

}