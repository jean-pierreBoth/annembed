//! annembed binary.  
//! 
//! This module provides just access to floating point data embedding.  
//! Command syntax is embed input --csv csvfile  [--outfile | -o  output_name] [--delim u8] [hnsw params] [embed params].  
//! 
//!  --outf or -o to specify the name of csv file containing embedded vectors.  
//!     By default the name is "embedded.csv"
//! 
//! hnsw is an optional command to change default parameters of the Hnsw structure. See [hnsw_rs](https://crates.io/crates/hnsw_rs).  
//! embed is an optional command to change default parameters related to the embedding: gradient, edge sampling etc. See [EmbedderParams]
//! 
//! - Parameters for embed subcommand. The options give access to some fields of the [EmbedderParams] structure.  
//!  --stepg    : a float value , initial gradient step, default is 2.
//!  --scale    : a float value, scale modification factor, default is 1.
//!  --nbsample : number of edge sampling , default is 10   
//!  --layer    : in case of hierarchical embedding num of the lower layer we consider to run preliminary step.  
//! 
//! - Parameters for the hnsw subcommand. For more details see   
//! --nbconn  : defines the number of connections by node in a layer.   
//! --dist    : name of distance to use: DistL1, DistL2, DistCosine, DistJeyffreys 
//! --ef      : controls the with of the search. 
//! --knbn    : the number of nodes to use in retrieval requests.  
//!     
//! The csv file must have one record by vector to embed. The default delimiter is ','.  
//! The output is a csv file with embedded vectors.
//! The Julia directory provide helpers to vizualize and some Topological Data analysis tools



use log;

use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use anyhow::{anyhow};
use clap::{Arg, ArgMatches, Command};



use hnsw_rs::prelude::*;

use annembed::fromhnsw::kgraph::{KGraph,kgraph_from_hnsw_all};
use annembed::fromhnsw::kgproj::{KGraphProjection};
use annembed::prelude::*;

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

    match matches.value_of("nbconn") {
        Some(str) =>  { 
            let res = str.parse::<usize>();
            match res {
                Ok(val) => { hnswparams.max_conn = val},
                _       => { return Err(anyhow!("could not parse nbconn parameter"));
                            },
            } 
        } 
        _      => { return Err(anyhow!("could not parse nbconn"));}
    };  // end of match  nbconn


    match matches.value_of("ef") {
        Some(str) =>  { 
            let res = str.parse::<usize>();
            match res {
                Ok(val) => { hnswparams.ef_c = val},
                _       => { return Err(anyhow!("could not parse ef_c parameter"));
                            },
            } 
        } 
        _      => { return Err(anyhow!("could not parse ef"));}
    };  // end of match ef


    match matches.value_of("knbn") {
        Some(str) =>  { 
            let res = str.parse::<usize>();
            match res {
                Ok(val) => { hnswparams.knbn = val},
                _       => { return Err(anyhow!("could not parse knbn parameter"));
                            },
            } 
        } 
        _      => { return Err(anyhow!("could not parse knbn"));}
    };  // end of match knbn


    match matches.value_of("dist") {
        Some(str) =>  { 
            match str {
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

    match matches.value_of("scale") {
        Some(str) =>  { 
            let res = str.parse::<f64>();
            match res {
                Ok(val) => { embedparams.scale_rho = val},
                _       => { return Err(anyhow!("could not parse scale_rho parameter"));
                            },
            } 
        } 
        _               => {}
    };  // end of match scale_rho

    match matches.value_of("nbsample") {
        Some(str) =>  { 
            let res = str.parse::<usize>();
            match res {
                Ok(val) => { embedparams.nb_sampling_by_edge = val},
                _       => { return Err(anyhow!("could not parse nbsample parameter"));
                            },
            } 
        } 
        _               => {}
    };  // end of match nb_sampling_by_edge


    match matches.value_of("hierarchy") {
        Some(str) =>  { 
            let res = str.parse::<usize>();
            match res {
                Ok(val) => { embedparams.hierarchy_layer = val},
                _       => { return Err(anyhow!("could not parse hierarchy layer parameter"));
                            },
            } 
        } 
        _               => {}
    } // end of match hierarchy

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
            .takes_value(true)
            .help("gradient step")
        )
        .arg(Arg::new("nbsample")
            .required(false)
            .long("nbsample")
            .takes_value(true)
            .help("number of edge sampling")
        )
        .arg(Arg::new("hierarchy")
            .required(false)
            .long("layer")
            .short('l')
            .takes_value(true)
            .help("expecting a layer num")
        )
        .arg(Arg::new("scale")
            .required(false)
            .long("scale")
            .takes_value(true)
            .help("spatial scale factor")
        );

    let hnswcmd = Command::new("hnsw")
        .arg(Arg::new("dist")
            .long("dist")
            .short('d')
            .required(true)
            .help("distance is required   \"DistL1\" , \"DistL2\", \"DistCosine\", \"DistJeyffreys\"  "))
        .arg(Arg::new("nb_conn")
            .long("nbconn")
            .takes_value(true)
            .help("number of neighbours by layer"))
        .arg(Arg::new("knbn")
            .long("knbn")
            .takes_value(true)
            .help("number of neighbours to use"))
        .arg(Arg::new("ef")
            .long("ef")
            .takes_value(true)
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
            .takes_value(true)
            .required(true)
            .help("expecting a csv file"))
        .arg(Arg::new("outfile")
            .long("out")
            .short('o')
            .takes_value(true)
            .help("expecting output file name"))
        .arg(Arg::new("delim")
            .long("delim")
            .short('d')
            .takes_value(true)
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

    let mut fname = String::from("");
    if matches.is_present("csvfile") {
        let csv_file = matches.value_of("csvfile").ok_or("").unwrap().parse::<String>().unwrap();
        if csv_file == "" {
            println!("parsing of request_dir failed");
            std::process::exit(1);
        }
        else {
            log::info!("input file : {:?}", csv_file.clone());
            fname = csv_file.clone();
        }
    }
    //
    let delim_opt = matches.get_one::<u8>("delim");
    let delim = match delim_opt {
        Some(c)  => { *c},
        None     => { b','},
    };
    // set output filename and check if option is present in command
    let mut csv_output = String::from("embedded.csv");
    if matches.is_present("outfile") {
        let csv_out = matches.value_of("outfile").ok_or("").unwrap().parse::<String>().unwrap();
        if csv_out == "" {
            println!("parsing of output file name failed");
            std::process::exit(1);
        }
        else {
            log::info!("input file : {:?}", csv_out.clone());
            csv_output = csv_out.clone();
        }
    }

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