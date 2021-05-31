//! To take charge of io csv bincode...
//! 
//! 
//!

use num_traits::{Float};


use ndarray::{Array2};

use csv::*;


/// This function is mostly dedicated to write embedded data in very few dimensions
pub fn write_csv_labeled_array2<F>(csv_writer : &mut Writer<std::fs::File>, labels : &[u8], mat : &Array2<F>) -> std::io::Result<usize>
            where F : Float {
    //
    let (nbrow, nbcol) = mat.dim();
    let mut line : Vec<String> = ((0..=nbcol)).into_iter().map(|_| String::from("")).collect();
    for i in 0..nbrow {
        line[0] = labels[i].to_string();
        for j in 0..nbcol {
            line[1+j] = format!("{:.2e}", mat[[i,j]].to_f32().unwrap());
        }
        csv_writer.write_record(&line)?;
    }
    csv_writer.flush()?;
    //
    return Ok(1);
} // end of dump_csv_array2