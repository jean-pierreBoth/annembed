// reservoir sampling

use rand_distr::Distribution;
use rand_xoshiro::Xoshiro256PlusPlus;

//  1. Faster methods for Random Sampling J.S Vitter Comm ACM 1984
//  2. Kim-Hung Li Reservoir Sampling Algorithms : Comm ACM Vol 20, 4 December 1994
//  3. https://en.wikipedia.org/wiki/Reservoir_sampling

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
    for i in 0..size_asked {
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
        s = s + (xsi.ln() / (1. - w).ln()).floor() as usize + 1;
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
