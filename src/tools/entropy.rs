//! This module computes entropy (mostly Renyi at present time) on a discrete probability
//! Vectors are dependant on the type F:Float + AddAssign + SubAssign + MulAssign + DivAssign + std::iter::Sum
//! which in fact imposes f32 or 64.
//!
//!

#![allow(dead_code)]

use core::ops::*;
use num_traits::Float; // for  AddAssign + SubAssign + MulAssign + DivAssign + RemAssign

//=================================================================================

#[derive(Clone, Copy, Debug)]
enum EntropyKind {
    Renyi,
    Shannon,
}

///  Renyi entropy value
#[derive(Copy, Clone, Debug)]
pub struct RenyiEntropy<F> {
    kind: EntropyKind,
    q: F,
    value: F,
}

impl<F> RenyiEntropy<F>
where
    F: Float + AddAssign + SubAssign + MulAssign + DivAssign + std::iter::Sum,
{
    pub fn new(q: F, value: F) -> Self {
        RenyiEntropy {
            kind: EntropyKind::Renyi,
            q,
            value,
        }
    }
}

//======================================================================================

/// A discrete probability
///
/// normalized set to true if normalization has been checked
///
pub struct DiscreteProba<F>
where
    F: Float + AddAssign + SubAssign + MulAssign + DivAssign + std::iter::Sum,
{
    p: Vec<F>,
    entropy: Option<Vec<RenyiEntropy<F>>>,
}

impl<F> DiscreteProba<F>
where
    F: Float + AddAssign + SubAssign + MulAssign + DivAssign + std::iter::Sum,
{
    pub fn new(p: &Vec<F>) -> Self {
        let mut sum = F::zero();
        let zero = F::zero();
        for x in p.iter() {
            if *x < zero {
                log::error!("negative value in probability");
                std::panic!("negative value in probability");
            } else {
                sum += *x;
            }
        }
        let np = p.iter().map(|&x| x / sum).collect();
        DiscreteProba {
            p: np,
            entropy: None,
        }
    }

    /// compute for q = 1 it is Shannon entropy
    fn renyi_entropy_1(&self) -> F {
        let zero = F::zero();
        let entropy = self
            .p
            .iter()
            .map(|&v| if v > zero { -v * v.ln() } else { zero })
            .sum();
        return entropy;
    }

    /// cmpute for q!= 1.
    /// See Leinster. Entropy and diversity
    fn renyi_entropy_not_1(&self, q: F) -> F {
        let zero = F::zero();
        let entropy: F = self
            .p
            .iter()
            .map(|&v| if v > zero { v.powf(q) } else { zero })
            .sum();
        entropy.ln() / (F::one() - q)
    }

    /// compute Renyi entrpy of a for a slice of order values
    pub fn renyi_entropy(&mut self, order: &[F]) -> Vec<RenyiEntropy<F>> {
        let mut entropy_v = Vec::<RenyiEntropy<F>>::with_capacity(order.len());
        for q in order.iter() {
            if near_to_1(*q) {
                let entropy = self.renyi_entropy_not_1(*q);
                entropy_v.push(RenyiEntropy::new(F::one(), entropy));
            } else {
                let entropy = self.renyi_entropy_not_1(*q);
                entropy_v.push(RenyiEntropy::new(*q, entropy));
            }
        }
        return entropy_v;
    } // end of entropy_renyi

    // compute relative entropy at q=1
    fn relative_renyi_entropy_1(&self, other: &DiscreteProba<F>) -> F {
        let zero = F::zero();
        let entropy = self
            .p
            .iter()
            .zip(other.p.iter())
            .map(|t| {
                if *t.0 > zero {
                    *t.0 * (*t.1 / *t.0).ln()
                } else {
                    zero
                }
            })
            .sum();
        //        let ent_f64 : f64 = num_traits::cast::cast::<F, f64>(entropy).unwrap();
        entropy
    }

    //
    fn relative_renyi_entropy_q(&self, other: &DiscreteProba<F>, q: F) -> F {
        let zero = F::zero();
        let entropy = self
            .p
            .iter()
            .zip(other.p.iter())
            .map(|t| {
                if *t.0 > zero {
                    *t.0 * (*t.1 / *t.0).powf(q)
                } else {
                    zero
                }
            })
            .sum();
        entropy
    }

    /// computes mean diversity of other with respect to self.
    ///
    ///
    /// $$ \sum_{self.p_{i} != 0} self.p_{i} * \phi(\frac{other.p_{i}}{self.p_{i}})$$
    ///
    ///
    pub fn relative_renyi_entropy(&self, other: &DiscreteProba<F>, q: F) -> F {
        if near_to_1(q) {
            return self.relative_renyi_entropy_1(other);
        } else {
            return self.relative_renyi_entropy_q(other, q);
        }
    } // end of relative_entropy
} // end impl ProbaDistribution

#[inline]
fn near_to_1<F: Float>(f: F) -> bool {
    let one = num_traits::identities::one::<F>();
    if (f - one).abs() < Float::epsilon() {
        true
    } else {
        false
    }
}

//==================================================================================

/// Probabilistic neigheibourhood
/// If y1 ,   , yn are the points in neigbourhood aroud y0
/// lambda * d()

pub struct ProbaNeighbour<F>
where
    F: Float + AddAssign + SubAssign + MulAssign + DivAssign + std::iter::Sum,
{
    point_id: u32,
    /// list of neighbour's point_id
    neighbours_id: Vec<u32>,
    /// distance to each neighbour
    distance: Vec<F>,
    /// probability transition deduced from distances
    proba: DiscreteProba<F>,
    // rescale coefficient
    lambda: f64,
} // end of ProbaNeighbour

impl<F> ProbaNeighbour<F> where
    F: Float + AddAssign + SubAssign + MulAssign + DivAssign + std::iter::Sum
{
}

//================================================================================

#[cfg(test)]
mod tests {

    use super::*;
    use rand::prelude::*;

    use rand::distributions::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]

    fn test_proba_ann() {
        let unif_01 = Uniform::<f64>::new(0., 1.);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567 as u64);
        let p: Vec<f64> = (0..50)
            .into_iter()
            .map(|_| unif_01.sample(&mut rng))
            .collect();
        let mut proba = DiscreteProba::new(&p);
        let _entropy = proba.renyi_entropy(&[1., 2.]);
    } // end of test_proba_ann
}
