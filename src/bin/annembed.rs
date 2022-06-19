//! annembed excutable.  
//! 
//! This module provide just access to the simple (not hierarchical) embedding.  
//! Cmd syntax is annembed input --file [hnsw params] [embed params]
//! hnsw is an optional command to change default parameters 
//! embed is an optional command to change default parameters




use log;

use anyhow::{anyhow};
use clap::{Arg, ArgMatches, Command};

use annembed::prelude::*;

/// Defines parameters to drive ann computations. See the crate [hnsw]
#[derive(Debug, Clone, Copy)]
struct HnswParams {
    max_conn : usize,
    ef_c : usize,
    knbn : usize,
}

impl HnswParams {
    pub fn default() -> Self {
        HnswParams{max_conn : 48, ef_c : 400, knbn : 10}
    }

    pub fn new(max_conn : usize, ef_c : usize, knbn : usize) -> Self {
        HnswParams{max_conn, ef_c, knbn}
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

    Ok(hnswparams)
}  // end of parse_hnsw_cmd


#[doc(hidden)]
pub fn main() {
    println!("initializing default logger from environment ...");
    let _ = env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");
    //
    let hnswparams : HnswParams;
    let _embedparams : EmbedderParams;
    //
    let embedcmd = Command::new("annembed")
        .arg(Arg::new("step_grap")
            .long("stepg")
            .takes_value(true)
            .help("gradient step"))
        .arg(Arg::new("nbgrag")
            .long("nbsample")
            .takes_value(true)
            .help("number of edge sampling"))
        .arg(Arg::new("scale")
            .long("scale")
            .takes_value(true)
            .help("spatial scale factor"));

    let hnswcmd = Command::new("hnsw")
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
        .subcommand(embedcmd)
        .subcommand(hnswcmd)
    .get_matches();

    if let Some(hnsw_m) = matches.subcommand_matches("hnsw") {
        log::debug!("subcommand_matches got validation");
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
} // end of main