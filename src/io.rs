//! To take charge of io csv bincode...
//! 
//! 
//!

use num_traits::{Float};


use ndarray::{Array2};

use csv::*;


/// This function is mostly dedicated to write embedded data in very few dimensions
pub fn write_csv_array2<F>(mat : &Array2<F>, filename : &str) -> std::io::Result<usize>
            where F : Float {
    //
    let mut wtr = Writer::from_path(filename)?;
    let (nbrow, nbcol) = mat.dim();
    let mut line : Vec<String> = (0..(nbcol+1)).into_iter().map(|_| String::from("")).collect();
    for i in 0..nbrow {
        line[0] = i.to_string();
        for j in 0..nbcol {
            line[j] = mat[[i,j]].to_f32().unwrap().to_string();
        }
        wtr.write_record(&line)?;
    }
    wtr.flush()?;
    //
    return Ok(1);
} // end of dump_csv_array2