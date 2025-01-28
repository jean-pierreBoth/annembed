// restrain value

use num_traits::Float;

pub(crate) fn clip<F>(f: F, max: F) -> F
where
    F: Float + num_traits::FromPrimitive,
{
    if f > max {
        log::trace!("truncated >");
        max
    } else if f < -max {
        log::trace!("truncated <");
        -max
    } else {
        f
    }
} // end clip
