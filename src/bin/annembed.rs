//! annembed excutable.  
//! 
//! This module provide just access to the simple (not hierarchical) embedding.  
//! Cmd syntax is annembed input --file [hnsw params] [embed params]
//! hnsw is an optional command to change default parameters 
//! embed is an optional command to change default parameters




use log;

use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use anyhow::{anyhow};
use clap::{Arg, ArgMatches, Command};


use hnsw_rs::prelude::*;

use annembed::fromhnsw::kgraph::{KGraph,kgraph_from_hnsw_all};
use annembed::prelude::*;

/// Defines parameters to drive ann computations. See the crate [hnsw]
#[derive(Debug, Clone)]
struct HnswParams {
    max_conn : usize,
    ef_c : usize,
    knbn : usize,
    distance : String,
}

impl HnswParams {
    pub fn default() -> Self {
        HnswParams{max_conn : 48, ef_c : 400, knbn : 10, distance : String::from("DistL2")}
    }

    pub fn new(max_conn : usize, ef_c : usize, knbn : usize, distance : String) -> Self {
        HnswParams{max_conn, ef_c, knbn, distance}
    }
} // end impl block


#[doc(hidden)]
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
                "DistL2"  => { hnswparams.distance = String::from("DistL2");}
                "DistL1"  => { hnswparams.distance = String::from("DistL1");}
                "DistCos" => { hnswparams.distance = String::from("DistCos");}

                _         => { return Err(anyhow!("not a valid distance"));}
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
        _      => { return Err(anyhow!("could not parse scale_rho"));}
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
        _      => { return Err(anyhow!("could not parse nbsample"));}
    };  // end of match nb_sampling_by_edge



    return Ok(embedparams);
} // end of parse_embed_cmd






#[doc(hidden)]
pub fn main() {
    println!("initializing default logger from environment ...");
    let _ = env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");
    //
    let hnswparams : HnswParams;
    let embedparams : EmbedderParams;
    //
    let embedcmd = Command::new("ann")
        .arg(Arg::new("step_grap")
            .long("stepg")
            .takes_value(true)
            .help("gradient step"))
        .arg(Arg::new("nbsample")
            .long("nbsample")
            .takes_value(true)
            .help("number of edge sampling"))
        .arg(Arg::new("scale")
            .long("scale")
            .takes_value(true)
            .help("spatial scale factor"));

    let hnswcmd = Command::new("hnsw")
        .arg(Arg::new("dist")
            .long("dist")
            .short('d')
            .required(true)
            .help("distance is required"))
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
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(Arg::new("csvfile")
            .long("csv")    
            .takes_value(true)
            .required(true)
            .help("expecting a csv file"))
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
    if let Some(ann_m) = matches.subcommand_matches("ann") {
        log::debug!("subcommand_matches got ann");
        let res = parse_embed_cmd(ann_m);        
        match res {
            Ok(params) => { embedparams = params; },
            _                      => { 
                                        log::error!("parsing hnsw command failed");
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

    let delim: u8 = *matches
            .get_one("delim")
            .expect("`delim`is required");
    // open file and embed

    let filepath = std::path::Path::new(&fname);
    let res = get_toembed_from_csv::<f64>(filepath, delim);

    if res.is_err() {
        log::error!("could not open file : {:?}", filepath);
        std::process::exit(1);
    }

    let data = res.unwrap();
    let nb_data = data.len();
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    //
    let hnsw = Hnsw::<f64, DistL2>::new(hnswparams.max_conn, nb_data, nb_layer, hnswparams.ef_c, DistL2{});
    let data_with_id : Vec<(&Vec<f64>, usize)>= data.iter().zip(0..data.len()).collect();
    hnsw.parallel_insert(&data_with_id);
    //
    let cpu_time: Duration = cpu_start.elapsed();
    println!(" ann construction sys time(s) {:?} cpu time {:?}", sys_now.elapsed().unwrap().as_secs(), cpu_time.as_secs());
    hnsw.dump_layer_info();

    let kgraph: KGraph<f64> = kgraph_from_hnsw_all(&hnsw, hnswparams.knbn).unwrap();

} // end of main