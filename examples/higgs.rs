//! test of embedding for HIGGS boson data that consists in 11 millions of points in dimension 21.
//! The data is described and can be retrieved at <https://archive.ics.uci.edu/ml/datasets/HIGGS>.
//! An example of this data set processing is given in the paper by Amid and Warmuth
//! Cf <https://arxiv.org/abs/1910.00204>
//! 

use anyhow::{anyhow};

use std::io::{BufReader};
use std::fs::{OpenOptions};
use std::path::{PathBuf};


use csv::Writer;

use ndarray::{Array2, ArrayView};

use hnsw_rs::prelude::*;
use hnsw_rs::hnswio::{load_description, load_hnsw};

use annembed::prelude::*;




use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use annembed::fromhnsw::kgraph::{kgraph_from_hnsw_all};
use annembed::fromhnsw::kgproj::KGraphProjection;

const HIGGS_DIR : &'static str = "/home/jpboth/Data/";


/// return a vector of labels, and a list of vectors to embed
/// First field of record is label, then the 21 following field are the data. 
/// 11 millions records!
fn read_higgs_csv(fname : String) -> anyhow::Result<(Vec<u8>, Array2<f32>)> {
    //
    let nb_fields = 29;
    let to_parse = 22;
    let mut num_record : usize = 0;
    let filepath = PathBuf::from(fname);
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_higgs_csv {:?}", filepath.as_os_str());
        println!("read_higgs_csv {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let mut labels = Vec::<u8>::new();
    let mut data = Array2::<f32>::zeros((0,21));
    let mut rdr = csv::Reader::from_reader(bufreader);
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        num_record += 1;
        if num_record % 1_000_000 == 0 {
            log::info!("loaded {} millions record", num_record);
        }
        let record = result?;
        if record.len() != nb_fields {
            println!("record {} record has {} fields", num_record,  record.len());
            return Err(anyhow!("record {} record has {} fields", num_record,  record.len()));
        }
        let mut new_data = Vec::<f32>::with_capacity(21);
        for j in 0..to_parse {
            let field = record.get(j).unwrap();
            // decode into Ix type
            if let Ok(val) = field.parse::<f32>() {
                match j {
                    0 => { labels.push(if val > 0. { 1} else {0}); }
                    _ => { new_data.push(val); }
                };
            }
            else {
                log::debug!("error decoding field  of record {}", num_record);
                return Err(anyhow!("error decoding field 1of record  {}",num_record)); 
            }
        } // end for j
        assert_eq!(new_data.len(), 21);
        data.push_row(ArrayView::from(&new_data)).unwrap();
    }
    //
    assert_eq!(data.dim().0, labels.len());
    log::info!("number of records read : {}", data.len());
    //
    Ok((labels, data))
}  // end of read_higgs_csv



fn rescale(data : &mut Array2<f32>) -> Vec<Vec<f32>> {

    let (nb_row, nb_col) = data.dim();
    let mut datavec = Vec::<Vec<f32>>::with_capacity(nb_row);
    // 
    for j in 0..nb_col {
        let mut col = data.column_mut(j);
        let mean = col.mean().unwrap();
        let sigma = col.var(1.).sqrt();
        col.mapv_inplace(|x| (x - mean)/sigma );
    }
    // reformat in vetors
    for i in 0..nb_row {
        datavec.push(data.row(i).to_vec());
    }
    //
    return datavec;
}  // end of rescale


fn reload_higgshnsw<D>() -> std::io::Result<Hnsw::<f32, D>> 
    where D : Distance<f32> + std::default::Default + std::marker::Send + std::marker::Sync {
    // look for 
    log::info!("trying to reload Higgs.hnsw.graph and Higgs.hnsw.data");
    let graphfname = String::from("Higgs.hnsw.graph");
    let graphpath = PathBuf::from(graphfname);
    let graphfileres = OpenOptions::new().read(true).open(&graphpath);
    if graphfileres.is_err() {
        println!("test_dump_reload: could not open file {:?}", graphpath.as_os_str());
        return Err(std::io::Error::new(std::io::ErrorKind::Other, "no previous Hnsw dump"));
    }
    let graphfile = graphfileres.unwrap();
    //  
    let datafname = String::from("Higgs.hnsw.data");
    let datapath = PathBuf::from(datafname);
    let datafileres = OpenOptions::new().read(true).open(&datapath);
    if datafileres.is_err() {
        println!("test_dump_reload : could not open file {:?}", datapath.as_os_str());
        std::panic::panic_any("test_dump_reload : could not open file".to_string());            
    }
    let datafile = datafileres.unwrap();
    //
    let mut graph_in = BufReader::new(graphfile);
    let mut data_in = BufReader::new(datafile);
    let sys_now = SystemTime::now();
    let cpu_time = ProcessTime::now();
    let hnsw_description = load_description(&mut graph_in).unwrap();
    let hnsw_loaded : std::io::Result<Hnsw<f32, D>> = load_hnsw::<f32, D>(&mut graph_in, &hnsw_description, &mut data_in);
    println!(" higgs ann reload sys time(s) {:?} cpu time {:?}", sys_now.elapsed().unwrap().as_secs(), cpu_time.elapsed().as_secs());
    //
    return hnsw_loaded;
}  // end of reload_higgshnsw



pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let mut fname = String::from(HIGGS_DIR);
    fname.push_str("HIGGS.csv");
    let hnsw : Hnsw::<f32, DistL2>;
    // read csv 8Gb data label is in first field as a float. We need to read it to get labels even if we reload a Hnsw
    let res = read_higgs_csv(fname);
    if res.is_err() {
        log::error!("error reading Higgs.csv {:?}", &res.as_ref().err().as_ref());
        std::process::exit(1);
    }
    let mut res = res.unwrap();
    let labels = res.0;
    let data = rescale(&mut res.1);
    drop(res.1);     // we do not need res.1 anymore
    assert_eq!(data.len(), labels.len());
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    // DO we have a dump ?
    let res_reload = reload_higgshnsw();
    if res_reload.is_ok() {
        hnsw = res_reload.unwrap();
        hnsw.dump_layer_info();
        let cpu_time: Duration = cpu_start.elapsed();
        println!(" higgs ann reload sys time(s) {:?} cpu time {:?}", sys_now.elapsed().unwrap().as_secs(), cpu_time.as_secs());
        drop(data);
    }
    else  { // need to construct hnsw
        log::info!("no Hnsw dump found in directory, reconstructing Hnsw structure");
        //
        let cpu_start_hnsw = ProcessTime::now();
        let sys_start_hnsw = SystemTime::now();
        //
        let ef_c = 400;
        let max_nb_connection = 24;
        let nbdata = data.len();
        let nb_layer = 16.min((nbdata as f32).ln().trunc() as usize);
        //
        hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nbdata, nb_layer, ef_c, DistL2{});
//        hnsw.set_keeping_pruned(true);
        // we insert by block of 1_000_000
        let block_size = 1_000_000;
        let mut inserted = 0;
        let mut numblock = 0;
        while inserted < data.len() {
            let block_length = ((numblock+1)*block_size).min(nbdata) - numblock*block_size;
            let mut data_with_id = Vec::<(&[f32], usize)>::with_capacity(block_length);
            for _ in 0..block_length {
                data_with_id.push((&data[inserted], inserted));
                inserted += 1;
            }
            hnsw.parallel_insert_slice(&data_with_id);
            numblock += 1;
        }
        // images as vectors of f32 and send to hnsw
        println!(" higgs ann construction sys time(s) {:?} cpu time {:?}", sys_start_hnsw.elapsed().unwrap().as_secs(), cpu_start_hnsw.elapsed().as_secs());
        hnsw.dump_layer_info();
        // save hnsw to avoid reconstruction runs in 1 hour...on my laptop
        let fname = String::from("Higgs");
        let _res = hnsw.file_dump(&fname);
    }
    //
    // now we embed
    //
    let mut embed_params = EmbedderParams::default();
    embed_params.nb_grad_batch = 25;
    embed_params.scale_rho = 1.;
    embed_params.beta = 1.;
    embed_params.grad_step = 1.;
    embed_params.nb_sampling_by_edge = 10;
    embed_params.dmap_init = true;
    let hierarchical = true;

    let mut embedder;
    let kgraph;
    let graphprojection;
    if !hierarchical {
        let knbn = 6;
        kgraph = kgraph_from_hnsw_all(&hnsw, knbn).unwrap();
        embedder = Embedder::new(&kgraph, embed_params);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok()); 
        assert!(embedder.get_embedded().is_some());
    } 
    else {
        let knbn = 10;
        log::debug!("trying graph projection");
        embed_params.nb_grad_batch = 25;
        graphprojection =  KGraphProjection::<f32>::new(&hnsw, knbn, 2);
        embedder = Embedder::from_hkgraph(&graphprojection, embed_params);
        let embed_res = embedder.embed();        
        assert!(embed_res.is_ok()); 
        assert!(embedder.get_embedded().is_some());
    }
    println!(" ann embed total sys time(s) {:.2e}  cpu time {:.2e}", sys_now.elapsed().unwrap().as_secs(), cpu_start.elapsed().as_secs());
    // dump
    log::info!("dumping initial embedding in csv file");
    let mut csv_w = Writer::from_path("higgs_initial_embedded.csv").unwrap();
    // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
    let _res = write_csv_labeled_array2(&mut csv_w, labels.as_slice(), &embedder.get_initial_embedding_reindexed());
    csv_w.flush().unwrap();

    log::info!("dumping in csv file");
    let mut csv_w = Writer::from_path("higgs_embedded.csv").unwrap();
    // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
    let _res = write_csv_labeled_array2(&mut csv_w, labels.as_slice(), &embedder.get_embedded_reindexed());
    csv_w.flush().unwrap();
    //
    // quality too memory consuming. must subsample
    // log::info!("estimating quality");
//    let _quality = embedder.get_quality_estimate_from_edge_length(200);
}  // end of main