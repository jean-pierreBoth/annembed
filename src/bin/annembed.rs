//! annembed binary.  
//! 
//! This module provides just access to floating point data embedding.  
//! Command syntax is embed input --csv csvfile  [--outfile | -o  output_name] [--delim u8] [hnsw params] [embed params].  
//! 
//!  --outfile or -o to specify the name of csv file containing embedded vectors. By default the name is "embedded.csv"
//! 
//! hnsw is an optional subcommand to change default parameters of the Hnsw structure. See [hnsw_rs](https://crates.io/crates/hnsw_rs).  
//! embed is an optional subcommand to change default parameters related to the embedding: gradient, edge sampling etc. See [EmbedderParams]
//! 
//! - Parameters for embed subcommand. The options give access to some fields of the [EmbedderParams] structure.  
//!  --stepg    : a float value , initial gradient step, default is 2.  
//!  --scale    : a float value, scale modification factor, default is 1.  
//!  --nbsample : number of edge sampling , default is 10   
//!  --layer    : in case of hierarchical embedding num of the lower layer we consider to run preliminary step.  
//! 
//! - Parameters for the hnsw subcommand. For more details see [hnsw_rs](https://crates.io/crates/hnsw_rs).   
//! --nbconn  : defines the number of connections by node in a layer.   Can range from 4 to 64 or more if necessary and enough memory
//! --dist    : name of distance to use: "DistL1", "DistL2", "DistCosine", "DistJeyffreys" 
//! --ef      : controls the with of the search, a good guess is between 24 and 64 or more if necessay
//! --knbn    : the number of nodes to use in retrieval requests.  
//!     
//! The csv file must have one record by vector to embed. The default delimiter is ','.  
//! The output is a csv file with embedded vectors.  
//! The Julia directory provides helpers to get Persistence diagrams and barcodes and vizualize them using Ripserer.jl



use log;

use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use anyhow::anyhow;
use clap::{Arg, ArgMatches, ArgAction, Command};



use hnsw_rs::prelude::*;

use annembed::fromhnsw::kgraph::{KGraph,kgraph_from_hnsw_all};
use annembed::fromhnsw::kgproj::KGraphProjection;
use annembed::prelude::*;
use annembed::fromhnsw::hubness::Hubness;

/// Defines parameters to drive ann computations. See the crate [hnsw_rs](https://crates.io/crates/hnsw_rs)
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// maximum number of connections within a layer
    max_conn : usize,
    /// width of search in hnsw
    ef_c : usize,
    /// number of neighbours asked for
    knbn : usize,
    /// distance to use in Hnsw. Default is "DistL2". Other choices are "DistL1", "DistCosine", DistJeffreys
    distance : String,
} // end of struct HnswParams


impl HnswParams {
    pub fn default() -> Self {
        HnswParams{max_conn : 48, ef_c : 400, knbn : 10, distance : String::from("DistL2")}
    }

    #[allow(unused)]
    pub fn new(max_conn : usize, ef_c : usize, knbn : usize, distance : String) -> Self {
        HnswParams{max_conn, ef_c, knbn, distance}
    }
} // end impl block



//==========================================================



fn parse_hnsw_cmd(matches : &ArgMatches) ->  Result<HnswParams, anyhow::Error> {
    log::debug!("in parse_hnsw_cmd");

    let mut hnswparams = HnswParams::default();

    hnswparams.max_conn = *matches.get_one::<usize>("nbconn").unwrap();

    hnswparams.ef_c = *matches.get_one::<usize>("ef").unwrap();

    hnswparams.knbn = *matches.get_one::<usize>("knbn").unwrap();


    match matches.get_one::<String>("dist") {
        Some(str) =>  { 
            match str.as_str() {
                "DistL2"          => { hnswparams.distance = String::from("DistL2");}
                "DistL1"          => { hnswparams.distance = String::from("DistL1");}
                "DistCosine"      => { hnswparams.distance = String::from("DistCosine");}
                "DistJeffreys"    => { hnswparams.distance = String::from("DistJeffreys");}
                _                 => { return Err(anyhow!("not a valid distance"));}
            }
        },
        _  => { return Err(anyhow!("could not parse distance"));}
    }; // end of match distance

    Ok(hnswparams)
}  // end of parse_hnsw_cmd




#[doc(hidden)]
fn parse_embed_cmd(matches : &ArgMatches) ->  Result<EmbedderParams, anyhow::Error> {
    log::debug!("in parse_embed_cmd");
    //
    let mut embedparams  = EmbedderParams::default();

    embedparams.scale_rho = *matches.get_one::<f64>("scale").unwrap();

    embedparams.nb_sampling_by_edge = *matches.get_one::<usize>("nbsample").unwrap();
 
    embedparams.hierarchy_layer = *matches.get_one::<usize>("hierarchy").unwrap();

    return Ok(embedparams);
} // end of parse_embed_cmd



// construct kgraph case not hierarchical
fn get_kgraph<Dist>(data_with_id : &Vec<(&Vec<f64>, usize)>, hnswparams : &HnswParams, nb_layer : usize) ->  KGraph<f64>
                    where  Dist : Distance<f64> + Default + Send + Sync {
    //
    let nb_data = data_with_id.len();
    let hnsw = Hnsw::<f64, Dist>::new(hnswparams.max_conn, nb_data, nb_layer, 
                    hnswparams.ef_c, Dist::default());
    hnsw.parallel_insert(&data_with_id);
    hnsw.dump_layer_info();
    let kgraph = kgraph_from_hnsw_all(&hnsw, hnswparams.knbn).unwrap();

    // Local Intrinsic dimension and hubness
    // Get some statistics on induced graph. This is not related to the embedding process
    let knbn = 25;
    let kgraph_new : KGraph<f32>;
    kgraph_new = kgraph_from_hnsw_all(&hnsw, knbn).unwrap();
    log::info!("minimum number of neighbours {}", kgraph_new.get_max_nbng());
    log::info!("dimension estimation...");
    let sampling_size = 10000;
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let dim_stat = kgraph_new.estimate_intrinsic_dim(sampling_size);
    let cpu_time: Duration = cpu_start.elapsed();
    println!("\n dimension estimation sys time(ms) : {:.3e},  cpu time(ms) {:.3e}\n", sys_now.elapsed().unwrap().as_millis(), cpu_time.as_millis());
    if dim_stat.is_ok() {
        let dim_stat = dim_stat.unwrap();
        log::info!("\n dimension estimation with nbpoints : {}, dim : {:.3e}, sigma = {:.3e} \n", 
            sampling_size, dim_stat.0, dim_stat.1);
        println!(" dimension estimation with nbpoints : {}, dim : {:.3e}, sigma = {:.3e}", 
            sampling_size, dim_stat.0, dim_stat.1); 
    }
    // hubness estimation
    //let hubness = hubness::Hubness::new(&kgraph);
    let hubness = Hubness::new(&kgraph_new);
    let s3_hubness = hubness.get_standard3m();
    log::info!("\n graph hubness estimation : {:.3e}", s3_hubness);
    println!("\n graph hubness estimation : {:.3e} \n", s3_hubness);
    let _histo = hubness.get_hubness_histogram();
    //
    let _kgraph_stats = kgraph_new.get_kraph_stats();

    kgraph
}  // end of get_kgraph




// construct kgraph case not hierarchical
fn get_kgraph_projection<Dist>(data_with_id : &Vec<(&Vec<f64>, usize)>, hnswparams : &HnswParams, nb_layer : usize, layer_proj : usize) ->  KGraphProjection<f64>
                    where  Dist : Distance<f64> + Default + Send + Sync {
    //
    let nb_data = data_with_id.len();
    let hnsw = Hnsw::<f64, Dist>::new(hnswparams.max_conn, nb_data, nb_layer, 
                    hnswparams.ef_c, Dist::default());
    hnsw.parallel_insert(&data_with_id);
    hnsw.dump_layer_info();
    let graphprojection =  KGraphProjection::<f64>::new(&hnsw, hnswparams.knbn, layer_proj);
    graphprojection
}  // end of get_kgraph_projection




// dispatching according to distance ... use a macro
fn get_kgraph_with_distname(data_with_id : &Vec<(&Vec<f64>, usize)>, hnswparams : &HnswParams, nb_layer : usize) -> KGraph<f64> {
    let kgraph = match hnswparams.distance.as_str() {
        "DistL2" => {
            let kgraph = get_kgraph::<DistL2>(&data_with_id, &hnswparams, nb_layer);
            kgraph            
        },
        "DistL1" => {
            let kgraph = get_kgraph::<DistL1>(&data_with_id, &hnswparams, nb_layer);
            kgraph                
        }
        "DistJeffreys" => {
            let kgraph = get_kgraph::<DistJeffreys>(&data_with_id, &hnswparams, nb_layer);
            kgraph                
        }
        "DistCosine" => {
            let kgraph = get_kgraph::<DistCosine>(&data_with_id, &hnswparams, nb_layer);
            kgraph                
        }
        _         => {
            log::error!("unknown distance : {}", hnswparams.distance);
            std::process::exit(1);           
        }
    };
    kgraph
}  // end of get_kgraph_with_distname



fn get_kgraphproj_with_distname(data_with_id : &Vec<(&Vec<f64>, usize)>, hnswparams : &HnswParams, 
                        nb_layer : usize, layer_proj : usize) ->  KGraphProjection::<f64> {
    //
    let kgraph_projection = match hnswparams.distance.as_str() {
        "DistL2" => {
            let kgraph = get_kgraph_projection::<DistL2>(&data_with_id, &hnswparams, nb_layer, layer_proj);
            kgraph            
        },
        "DistL1" => {
            let kgraph = get_kgraph_projection::<DistL1>(&data_with_id, &hnswparams, nb_layer, layer_proj);
            kgraph                
        }
        "DistJeffreys" => {
            let kgraph = get_kgraph_projection::<DistJeffreys>(&data_with_id, &hnswparams, nb_layer, layer_proj);
            kgraph                
        }
        "DistCosine" => {
            let kgraph = get_kgraph_projection::<DistCosine>(&data_with_id, &hnswparams, nb_layer, layer_proj);
            kgraph                
        }
        _         => {
            log::error!("unknown distance : {}", hnswparams.distance);
            std::process::exit(1);           
        }        
    };
    kgraph_projection
} // end of get_kgraphproj_with_distname




pub fn main() {
    println!("initializing default logger from environment ...");
    let _ = env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");
    //
    let hnswparams : HnswParams;
    let embedparams : EmbedderParams;
    //
    let embedcmd = Command::new("embed")
        .arg(Arg::new("step_grap")
            .required(false)
            .long("stepg")
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(f64))
            .help("gradient step")
        )
        .arg(Arg::new("nbsample")
            .required(false)
            .long("nbsample")
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("number of edge sampling")
        )
        .arg(Arg::new("hierarchy")
            .required(false)
            .long("layer")
            .short('l')
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("expecting a layer num")
        )
        .arg(Arg::new("scale")
            .required(false)
            .long("scale")
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(f64))
            .default_value("1.0 f64")
            .help("spatial scale factor")
        );

    let hnswcmd = Command::new("hnsw")
        .arg(Arg::new("dist")
            .long("dist")
            .short('d')
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(String))
            .help("distance is required   \"DistL1\" , \"DistL2\", \"DistCosine\", \"DistJeyffreys\"  "))
        .arg(Arg::new("nb_conn")
            .long("nbconn")
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("number of neighbours by layer"))
        .arg(Arg::new("knbn")
            .long("knbn")
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("number of neighbours to use"))
        .arg(Arg::new("ef")
            .long("ef")
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("search factor"));
        

    //
    // Now the command line
    // ===================
    //
    let matches = Command::new("annembed")
//        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(Arg::new("csvfile")
            .long("csv")    
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(String))
            .required(true)
            .help("expecting a csv file"))
        .arg(Arg::new("outfile")
            .long("out")
            .short('o')
            .required(false)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(String))
            .help("expecting output file name"))
        .arg(Arg::new("delim")
            .long("delim")
            .short('d')
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(char))
            .help("delimiter can be ' ', ','"))
        .subcommand(embedcmd)
        .subcommand(hnswcmd)
    .get_matches();

    // parse hnsw parameters
    if let Some(hnsw_m) = matches.subcommand_matches("hnsw") {
        log::debug!("subcommand_matches got hnsw");
        let res = parse_hnsw_cmd(hnsw_m);        
        match res {
            Ok(params) => { hnswparams = params; },
            _                      => { 
                                        log::error!("parsing hnsw command failed");
                                        println!("exiting with error {}", res.err().as_ref().unwrap());
                                        //  log::error!("exiting with error {}", res.err().unwrap());
                                        std::process::exit(1);                                
            },
        }
    }
    else {
        hnswparams = HnswParams::default();
    }
    log::debug!("hnswparams : {:?}", hnswparams);

    // parse ann parameters
    if let Some(ann_m) = matches.subcommand_matches("embed") {
        log::debug!("subcommand_matches got ann");
        let res = parse_embed_cmd(ann_m);        
        match res {
            Ok(params) => { embedparams = params; },
            _                      => { 
                                        log::error!("parsing embed cmd failed");
                                        println!("exiting with error {}", res.err().as_ref().unwrap());
                                        //  log::error!("exiting with error {}", res.err().unwrap());
                                        std::process::exit(1);                                
            },
        }
    }
    else {
        embedparams = EmbedderParams::default();
    }
    embedparams.log();

    let csv_file = matches.get_one::<String>("csvfile").unwrap();
    let fname = csv_file.clone();
    //
    let delim_opt = matches.get_one::<u8>("delim");
    let delim = match delim_opt {
        Some(c)  => { *c},
        None     => { b','},
    };
    // set output filename and check if option is present in command
    let mut csv_output = String::from("embedded.csv");
    let csv_out = matches.get_one::<String>("outfile");
    if csv_out.is_some() {
        csv_output = csv_out.unwrap().clone();
    }
    log::info!("output file : {:?}", &csv_output);

    // open file
    let filepath = std::path::Path::new(&fname);
    let res = get_toembed_from_csv::<f64>(filepath, delim);
    if res.is_err() {
        log::error!("could not open file : {:?}", filepath);
        std::process::exit(1);
    }
    log::info!("csv file {} read", fname);
    //
    let data = res.unwrap();
    let data_with_id : Vec<(&Vec<f64>, usize)>= data.iter().zip(0..data.len()).collect();
    let nb_data = data.len();
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();

    log::info!("dumping in csv file {}", csv_output);
    let mut csv_w = csv::Writer::from_path(csv_output).unwrap();
    //
    if embedparams.get_hierarchy_layer() == 0 {
        let kgraph = get_kgraph_with_distname(&data_with_id, &hnswparams, nb_layer);
        let cpu_time: Duration = cpu_start.elapsed();
        println!(" graph construction sys time(s) {:?} cpu time {:?}", sys_now.elapsed().unwrap().as_secs(), cpu_time.as_secs());
        let mut embedder = Embedder::new(&kgraph, embedparams);
        let embed_res = embedder.embed();
        if embed_res.is_err() {
            log::error!("embedding failed");
            std::process::exit(1);
        }
        //
        // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();
    } // end not hierarchical
    else {
        let graphprojection =  get_kgraphproj_with_distname(&data_with_id, &hnswparams, 
                                        nb_layer, embedparams.get_hierarchy_layer());
        let mut embedder = Embedder::from_hkgraph(&graphprojection, embedparams);
        let embed_res = embedder.embed();        
        assert!(embed_res.is_ok()); 
        assert!(embedder.get_embedded().is_some());
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();
    }
} // end of main