//! To take charge of io csv bincode...
//! 
//! 
//!


use log::*;
use anyhow::anyhow;

use std::fs::OpenOptions;
use std::path::Path;
use std::io::{Read, BufReader, BufRead};

use num_traits::Float;
use std::str::FromStr;


use ndarray::Array2;

use csv::*;


/// This function is mostly dedicated to write embedded data in very few dimensions
pub fn write_csv_labeled_array2<F, T>(csv_writer : &mut Writer<std::fs::File>, labels : &[T], mat : &Array2<F>) -> std::io::Result<usize>
            where F : Float , T : ToString {
    //
    let (nbrow, nbcol) = mat.dim();
    let mut line : Vec<String> = ((0..=nbcol)).into_iter().map(|_| String::from("")).collect();
    for i in 0..nbrow {
        line[0] = labels[i].to_string();
        for j in 0..nbcol {
            line[1+j] = format!("{:.5e}", mat[[i,j]].to_f32().unwrap());
        }
        csv_writer.write_record(&line)?;
    }
    csv_writer.flush()?;
    //
    return Ok(1);
} // end of dump_csv_array2


/// This function dumps an array2 into a csf file 
pub fn write_csv_array2<F>(csv_writer : &mut Writer<std::fs::File>, mat : &Array2<F>) -> std::io::Result<usize>
            where F : Float {
    //
    let (nbrow, nbcol) = mat.dim();
    let mut line : Vec<String> = ((0..nbcol)).into_iter().map(|_| String::from("")).collect();
    for i in 0..nbrow {
        for j in 0..nbcol {
            line[j] = format!("{:.5e}", mat[[i,j]].to_f32().unwrap());
        }
        csv_writer.write_record(&line)?;
    }
    csv_writer.flush()?;
    //
    return Ok(1);
} // end of write_csv_array2


// count number of first lines beginning with '#' or '%'
pub(crate) fn get_header_size(filepath : &Path) -> anyhow::Result<usize> {
    //
    log::debug!("get_header_size");
    //
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("fn get_header_size : could not open file {:?}", filepath.as_os_str());
        println!("fn get_header_size : could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("fn get_header_size : could not open file {}", filepath.display()));            
    }
    let mut file = fileres?;
    let mut nb_header_lines = 0;
    let mut c = [0];
    let mut more = true;
    while more {
        file.read_exact(&mut c)?;
        if ['#', '%'].contains(&(c[0] as char)) {
            nb_header_lines += 1;
            loop {
                file.read_exact(&mut c)?;
                if c[0] == '\n' as u8 {
                    break;
                }
            }
        }
        else {
            more = false;
            log::debug!("file has {} nb headers lines", nb_header_lines);
        }
    }
    //
    Ok(nb_header_lines)
}// end of get_header_size


/// get data to embed from a csv file
/// Each line of the file must have a vector of float values with some standard csv delimiters.
/// A header is possible with lines beginning with '#' or '%'
pub fn get_toembed_from_csv<F> (filepath : &Path, delim : u8) -> anyhow::Result<Vec<Vec<F>>> 
    where F : FromStr + Float {
    //
    let nb_headers_line = get_header_size(&filepath)?;
    log::info!("directed_from_csv , got header nb lines {}", nb_headers_line);
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("ProcessingState reload_json : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let mut bufreader = BufReader::new(file);
    // skip header lines
    let mut headerline = String::new();
    for _ in 0..nb_headers_line {
        bufreader.read_line(&mut headerline)?;
    }
    //
    let mut nb_record = 0;      // number of record loaded
    let mut num_record : usize = 0;
    let mut nb_fields = 0;
    let mut toembed = Vec::<Vec<F>>::new();
    //
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(bufreader);
    for result in rdr.records() {
        num_record += 1;
        let record = result?;
        if log::log_enabled!(Level::Info) && nb_record <= 2 {
            log::debug!(" record num {:?}, {:?}", nb_record, record);
        }
        if nb_record == 0 {
            nb_fields = record.len();
            log::info!("nb fields = {}", nb_fields);
            if nb_fields < 2 {
                log::error!("found only one field in record, check the delimitor , got {:?} as delimitor ", delim as char);
                return Err(anyhow!("found only one field in record, check the delimitor , got {:?} as delimitor ", delim as char));
            }
        }
        else {
            if record.len() != nb_fields {
                println!("non constant number of fields at record {} first record has {}",num_record,  nb_fields);
                return Err(anyhow!("non constant number of fields at record {} first record has {}",num_record,  nb_fields));   
            }
            // We have a new vector with nb_fields to parse
            let mut v = Vec::<F>::with_capacity(nb_fields);
            for j in 0..nb_fields {
                let field = record.get(j).unwrap();
                // decode into Ix type
                if let Ok(val) = field.parse::<F>() {
                    v.push(val);
                }
                else {
                    log::error!("error decoding field {} of record  {}, field : {:?}",j, num_record, field);
                    return Err(anyhow!("error decoding field {} of record  {}, field : {:?}",j, num_record, field)); 
                }
            }
            toembed.push(v);
        }
        nb_record += 1;
    }
    Ok(toembed)
} // end of get_toembed_from_csv



//========================================================================================

#[cfg(test)]
mod tests {

//    cargo test io  -- --nocapture
//    cargo test io::tests::testname -- --nocapture
//    RUST_LOG=annembed::io=TRACE cargo test testname -- --nocapture



use super::*;

fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  // end of log_init_test


static TESTDIR : &str = "/home/jpboth/Rust/annembed/Tmp";

#[test]
fn load_csv() {
    log_init_test();
    //
    let path = Path::new(TESTDIR).join("toembed.csv");
    let fileres = OpenOptions::new().read(true).open(&path);
    if fileres.is_ok() {
        let toembed = get_toembed_from_csv::<f32>(&path, b',');
        assert!(toembed.is_ok());        
    }
} // end of load_csv


} // end of mod tests