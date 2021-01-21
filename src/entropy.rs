//! This module computes various entropy
//! 
//! 

/// A discrete probability
/// 
/// normalized set to true if normalization has been checked
/// 
pub struct ProbaDistribution {
    normalized : bool,
    p : Vec<f64>
    entropy :Option<Vec<RenyiEntropy>>,
}

impl ProbaDistribution {
    /// compute for q != 1 and 1.
    /// compute Renyi entrpy of a for a vector of order
    pub fn computeRenyi(&mut self, order:Vec<i32>) -> f64 {

    } // end of computeRenyi

    /// rescale function
} // end impl ProbaDistribution


/// 
pub struct RenyiEntropy {
    q : u32,
    value : f64,
}


/// Probabilistic neigheibourhood
/// If y1 ,   , yn are the points in neigbourhood aroud y0
/// lambda * d()

pub struct ProbaNeighbour {
    point_id : u32,
    neighbours_id : Vec<u32>,
    distance : Vec<f64>,
    proba : ProbaDistribution,
    // rescale coefficient 
    lambda : f64
} // end of ProbaNeighbour


impl ProbaNeighbour {

}