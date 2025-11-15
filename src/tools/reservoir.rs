// reservoir sampling

use rand_distr::Distribution;
use rand_xoshiro::Xoshiro256PlusPlus;

//  1. Faster methods for Random Sampling J.S Vitter Comm ACM 1984
//  2. Kim-Hung Li Reservoir Sampling Algorithms : Comm ACM Vol 20, 4 December 1994
//  3. https://en.wikipedia.org/wiki/Reservoir_sampling

/// Algorithm L of Kim-Hung Li paper
#[allow(unused)]
pub(crate) fn unweighted_reservoir<T, IT>(
    size_asked: usize,
    mut in_terms: IT,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<T>
where
    IT: Iterator<Item = T>,
{
    let mut out_terms = Vec::<T>::with_capacity(size_asked.min(in_terms.size_hint().0));
    //
    for _ in 0..size_asked {
        match in_terms.next() {
            Some(item) => out_terms.push(item),
            None => break,
        }
    }
    let mut xsi: f64;
    xsi = rand_distr::StandardUniform.sample(rng);
    let mut w: f64 = (xsi.ln() / (size_asked as f64)).exp();
    let mut s = size_asked - 1;
    //
    loop {
        xsi = rand_distr::StandardUniform.sample(rng);
        s = (xsi.ln() / (1. - w).ln()).floor() as usize;
        match in_terms.nth(s) {
            Some(item) => {
                // update random index in out_terms
                xsi = rand_distr::StandardUniform.sample(rng);
                let idx = (size_asked as f64 * xsi).floor() as usize;
                out_terms[idx] = item;
                // update w
                xsi = rand_distr::StandardUniform.sample(rng);
                w = w * (xsi.ln() / (size_asked as f64)).exp();
            }
            None => break,
        }
    } // end while
    out_terms
}

#[cfg(test)]
mod tests {

    //    use std::iter::FromIterator;
    use super::*;
    use rand::SeedableRng;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_reservoir_sampling() {
        log_init_test();
        let mut rng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(4664397);
        let nb_asked = 5000;
        let in_terms = Vec::<usize>::from_iter::<std::ops::Range<usize>>(0..60000);
        let selected_terms = unweighted_reservoir(nb_asked, in_terms.iter(), &mut rng);
        let mean = selected_terms.iter().fold(0, |acc, t| acc + *t) as f64 / nb_asked as f64;
        log::debug!(" mean= {:.3e} ", mean);
        //
        assert_eq!(selected_terms.len(), nb_asked);
        // sort and print
    }
} // end of mod tests
