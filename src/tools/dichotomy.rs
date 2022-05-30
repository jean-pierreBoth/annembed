//! dichotomy search for monotnous function




/// search a root for f(x) = target between lower_r and upper_r. The flag increasing specifies the variation of f. true means increasing
pub(crate) fn dichotomy_solver<F>(increasing: bool, f: F, lower_r: f32, upper_r: f32, target: f32) -> Result<f32,f32>
where
    F: Fn(f32) -> f32,
{
    //
    if lower_r >= upper_r {
        panic!(
            "dichotomy_solver failure low {} greater than upper {} ",
            lower_r, upper_r
        );
    }
    let range_low = f(lower_r).max(f(upper_r));
    let range_upper = f(upper_r).min(f(lower_r));
    if f(lower_r).max(f(upper_r)) < target || f(upper_r).min(f(lower_r)) > target {
        panic!(
            "dichotomy_solver target not in range of function range {}  {} ",
            range_low, range_upper
        );
    }
    //
    if f(upper_r) < f(lower_r) && increasing {
        panic!("f not increasing")
    } else if f(upper_r) > f(lower_r) && !increasing {
        panic!("f not decreasing")
    }
    // target in range, proceed
    let mut middle = 1.;
    let mut upper = upper_r;
    let mut lower = lower_r;
    //
    let mut nbiter = 0;
    while (target - f(middle)).abs() > 1.0E-5 {
        if increasing {
            if f(middle) > target {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        // increasing type
        else {
            // decreasing case
            if f(middle) > target {
                lower = middle;
            } else {
                upper = middle;
            }
        } // end decreasing type
        middle = (lower + upper) * 0.5;
        nbiter += 1;
        if nbiter > 100 {
            return Err((target - f(middle)).abs());
        }
    } // end of while
    return Ok(middle);
}


//======================================================================


#[cfg(test)]
mod tests {


    use super::*;

    #[test]
    fn test_dichotomy_inc() {
        let f = |x: f32| x * x;
        //
        let beta = dichotomy_solver(true, f, 0., 5., 2.).unwrap();
        println!("beta : {}", beta);
        assert!((beta - 2.0f32.sqrt()).abs() < 1.0E-4);
    } // test_dichotomy_inc

    
    #[test]
    fn test_dichotomy_dec() {
        let f = |x: f32| 1.0f32 / (x * x);
        //
        let beta = dichotomy_solver(false, f, 0.2, 5., 1. / 2.).unwrap();
        println!("beta : {}", beta);
        assert!((beta - 2.0f32.sqrt()).abs() < 1.0E-4);
    } // test_dichotomy_dec

}