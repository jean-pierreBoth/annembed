//! annembed binary.  
//!
//! This module provides just access to floating point diffusion map data embedding.  
//! Command syntax is embed input --csv csvfile  [--outfile | -o  output_name] [--delim u8] [various embedding parameters] [hnsw params] .  
//!
//!  --outfile or -o to specify the name of csv file containing embedded vectors. By default the name is "embedded.csv"
//!
//! hnsw is an optional subcommand to change default parameters of the Hnsw structure. See [hnsw_rs](https://crates.io/crates/hnsw_rs).  
//!
//! - Parameters for embedding.  
//!     The options are optional and give access to some fields of the [EmbedderParams] structure.  
//!
//!     --layer    : optional, in case of hierarchical embedding num of the lower layer we consider to run preliminary step.
//!               default is set to 0 meaning one pass embedding.  
//!     --dim      : optional, dimension of the embedding , default to 2.  
//!
//!     --quality  : optional, asks for quality estimation.  
//!     --sampling : optional, for large data defines the fraction of sampled data as 1./sampling
//!
//! - Parameters for the hnsw subcommand. For more details see [hnsw_rs](https://crates.io/crates/hnsw_rs).   
//!     --nbconn  : defines the number of connections by node in a layer.   Can range from 4 to 64 or more if necessary and enough memory.  
//!     --dist    : name of distance to use: "DistL1", "DistL2", "DistCosine", "DistJeyffreys".  
//!     --ef      : controls the with of the search, a good guess is between 24 and 64 or more if necessary.  
//!     --knbn    : the number of nodes to use in retrieval requests.  
//!     
//! The csv file must have one record by vector to embed. The default delimiter is ','.  
//! The output is a csv file with embedded vectors.  
//! The Julia directory provides helpers to get Persistence diagrams and barcodes and vizualize them using Ripserer.jl

use annembed::diffmaps::{DiffusionMaps, DiffusionParams};
use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use anyhow::anyhow;
use clap::{Arg, ArgAction, ArgMatches, Command};

use hnsw_rs::prelude::*;

use annembed::fromhnsw::kgproj::KGraphProjection;
use annembed::fromhnsw::kgraph::{kgraph_from_hnsw_all, KGraph};
use annembed::prelude::*;

/// Defines parameters to drive ann computations. See the crate [hnsw_rs](https://crates.io/crates/hnsw_rs)
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// maximum number of connections within a layer
    max_conn: usize,
    /// width of search in hnsw
    ef_c: usize,
    /// number of neighbours asked for
    knbn: usize,
    /// distance to use in Hnsw. Default is "DistL2". Other choices are "DistL1", "DistCosine", DistJeffreys
    distance: String,
} // end of struct HnswParams

impl HnswParams {
    pub fn my_default() -> Self {
        HnswParams {
            max_conn: 48,
            ef_c: 400,
            knbn: 10,
            distance: String::from("DistL2"),
        }
    }

    #[allow(unused)]
    pub fn new(max_conn: usize, ef_c: usize, knbn: usize, distance: String) -> Self {
        HnswParams {
            max_conn,
            ef_c,
            knbn,
            distance,
        }
    }
} // end impl block

//
/// For large data volume quality estimation can use sampling of data with acceptance probability
#[derive(Copy, Clone)]
pub struct QualityParams {
    sampling_fraction: f64,
}

impl Default for QualityParams {
    fn default() -> Self {
        QualityParams {
            sampling_fraction: 1.0,
        }
    }
}

//==========================================================

fn parse_hnsw_cmd(matches: &ArgMatches) -> Result<HnswParams, anyhow::Error> {
    log::debug!("in parse_hnsw_cmd");

    let mut hnswparams = HnswParams::my_default();
    hnswparams.max_conn = *matches.get_one::<usize>("nbconn").unwrap();
    hnswparams.ef_c = *matches.get_one::<usize>("ef").unwrap();
    hnswparams.knbn = *matches.get_one::<usize>("knbn").unwrap();

    match matches.get_one::<String>("dist") {
        Some(str) => match str.as_str() {
            "DistL2" => {
                hnswparams.distance = String::from("DistL2");
            }
            "DistL1" => {
                hnswparams.distance = String::from("DistL1");
            }
            "DistCosine" => {
                hnswparams.distance = String::from("DistCosine");
            }
            "DistJeffreys" => {
                hnswparams.distance = String::from("DistJeffreys");
            }
            _ => {
                return Err(anyhow!("not a valid distance"));
            }
        },
        _ => {
            return Err(anyhow!("could not parse distance"));
        }
    }; // end of match distance

    Ok(hnswparams)
} // end of parse_hnsw_cmd

#[doc(hidden)]
fn parse_dmap_group(
    matches: &ArgMatches,
) -> Result<(DiffusionParams, Option<QualityParams>), anyhow::Error> {
    log::debug!("in parse_dmap_group");
    //
    let mut dmap_params = DiffusionParams::default();
    //
    dmap_params.set_alfa(*matches.get_one::<f32>("alfa").unwrap());
    dmap_params.set_beta(*matches.get_one::<f32>("beta").unwrap());
    dmap_params.set_hlayer(*matches.get_one::<usize>("hierarchy").unwrap());
    // for quality
    let quality: Option<QualityParams>;
    if let Some(fraction) = matches.get_one::<f64>("quality") {
        log::debug!("quality is asked, sampling fraction : {:.2e}", fraction);
        let qualityparmas = QualityParams {
            sampling_fraction: *fraction,
        };
        quality = Some(qualityparmas);
    } else {
        quality = None;
    }
    //
    Ok((dmap_params, quality))
} // end of parse_embed_cmd

//

#[allow(clippy::range_zip_with_len)]
pub fn main() {
    println!("initializing default logger from environment ...");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");
    //
    let hnswparams: HnswParams;
    let dmapparams: DiffusionParams;
    //
    let hnswcmd = Command::new("hnsw")
        .arg(Arg::new("dist")
            .long("dist")
            .short('d')
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(String))
            .help("distance is required   \"DistL1\" , \"DistL2\", \"DistCosine\", \"DistJeyffreys\"  "))
        .arg(Arg::new("nbconn")
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
        .arg(
            Arg::new("csvfile")
                .long("csv")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("expecting a csv file"),
        )
        .arg(
            Arg::new("outfile")
                .long("out")
                .short('o')
                .required(false)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .help("expecting output file name"),
        )
        .arg(
            Arg::new("delim")
                .long("delim")
                .short('d')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(char))
                .help("delimiter can be ' ', ','"),
        )
        // ann group flags
        .arg(
            Arg::new("batch")
                .required(false)
                .long("batch")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("20")
                .help("number of batches to run"),
        )
        .arg(
            Arg::new("grap_step")
                .required(false)
                .long("stepg")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .help("gradient step"),
        )
        .arg(
            Arg::new("nbsample")
                .required(false)
                .long("nbsample")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("10")
                .help("number of edge sampling"),
        )
        .arg(
            Arg::new("hierarchy")
                .required(false)
                .long("layer")
                .short('l')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("0")
                .help("expecting a layer num"),
        )
        .arg(
            Arg::new("scale")
                .required(false)
                .long("scale")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .default_value("1.0")
                .help("spatial scale factor"),
        )
        .arg(
            Arg::new("dimension")
                .required(false)
                .long("dim")
                .short('d')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("2")
                .help("dimension of embedding"),
        )
        .arg(
            Arg::new("quality")
                .required(false)
                .long("quality")
                .short('q')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .help("specify sampling fraction, should <= 1."),
        )
        .subcommand(hnswcmd)
        .get_matches();

    // parse hnsw parameters
    if let Some(hnsw_m) = matches.subcommand_matches("hnsw") {
        log::debug!("subcommand_matches got hnsw");
        let res = parse_hnsw_cmd(hnsw_m);
        match res {
            Ok(params) => {
                hnswparams = params;
            }
            _ => {
                log::error!("parsing hnsw command failed");
                println!("exiting with error {}", res.err().as_ref().unwrap());
                //  log::error!("exiting with error {}", res.err().unwrap());
                std::process::exit(1);
            }
        }
    } else {
        hnswparams = HnswParams::my_default();
    }
    log::debug!("hnswparams : {:?}", hnswparams);

    // parse ann parameters
    let quality: Option<QualityParams>;
    log::debug!("parsing specific embed params");
    let res = parse_dmap_group(&matches);
    match res {
        Ok((d_params, asked_quality)) => {
            dmapparams = d_params;
            quality = asked_quality;
        }
        _ => {
            log::error!("parsing embed cmd failed");
            println!("exiting with error {}", res.err().as_ref().unwrap());
            //  log::error!("exiting with error {}", res.err().unwrap());
            std::process::exit(1);
        }
    }

    // TODO: dmapparams.log();

    let csv_file = matches.get_one::<String>("csvfile").unwrap();
    let fname = csv_file.clone();
    //
    let delim_opt = matches.get_one::<u8>("delim");
    let delim = match delim_opt {
        Some(c) => *c,
        None => b',',
    };
    // set output filename and check if option is present in command
    let mut csv_output = String::from("embedded.csv");
    let csv_out = matches.get_one::<String>("outfile");
    if let Some(out) = csv_out {
        csv_output.clone_from(out);
    }
    log::info!("output file : {:?}", &csv_output);

    // open file
    let fraction = match quality {
        Some(q_params) => q_params.sampling_fraction,
        None => 1.,
    };
    let filepath = std::path::Path::new(&fname);
    let res = get_toembed_from_csv::<f64>(filepath, delim, fraction);
    if res.is_err() {
        log::error!("could not open file : {:?}", filepath);
        std::process::exit(1);
    }
    log::info!("csv file {} read", fname);
    //
    let data = res.unwrap();
    let data_with_id: Vec<(&Vec<f64>, usize)> = data.iter().zip(0..data.len()).collect();
    let nb_data = data.len();
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();

    log::info!("dumping in csv file {}", csv_output);
    let mut csv_w = csv::Writer::from_path(csv_output).unwrap();
    //
    if dmapparams.get_hlayer() == 0 {
        let kgraph = get_kgraph_with_distname(&data_with_id, &hnswparams, nb_layer);
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " graph construction sys time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_time.as_secs()
        );
        let mut dmapembedder = DiffusionMaps::new(dmapparams);
        let embed_res = dmapembedder.embed_from_kgraph(&kgraph, &dmapparams);
        if embed_res.is_err() {
            log::error!("diffusion map embedding failed");
            std::process::exit(1);
        }
        let embedded_data = embed_res.unwrap();
        //
        // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
        let _res = write_csv_array2(&mut csv_w, &embedded_data);
        csv_w.flush().unwrap();
        //
        /*         if quality.is_some() {
            let _quality = embedder.get_quality_estimate_from_edge_length(100);
        } */
    }
    // end not hierarchical
    else {
        let graphprojection = get_kgraphproj_with_distname(
            &data_with_id,
            &hnswparams,
            nb_layer,
            dmapparams.get_hlayer(),
        );
        panic!("not yet implemented");
        /*         let mut embedder = Embedder::from_hkgraph(&graphprojection, dmapparams);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
        assert!(embedder.get_embedded().is_some());
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();
        //
        if quality.is_some() {
            let _quality = embedder.get_quality_estimate_from_edge_length(100);
        } */
    }
} // end of main

//==========================================================================

// construct kgraph case not hierarchical
fn get_kgraph<Dist>(
    data_with_id: &[(&Vec<f64>, usize)],
    hnswparams: &HnswParams,
    nb_layer: usize,
) -> KGraph<f64>
where
    Dist: Distance<f64> + Default + Send + Sync,
{
    //
    let nb_data = data_with_id.len();
    let hnsw = Hnsw::<f64, Dist>::new(
        hnswparams.max_conn,
        nb_data,
        nb_layer,
        hnswparams.ef_c,
        Dist::default(),
    );
    hnsw.parallel_insert(data_with_id);
    hnsw.dump_layer_info();
    let kgraph_res = kgraph_from_hnsw_all(&hnsw, hnswparams.knbn);
    if kgraph_res.is_err() {
        panic!("kgraph_from_hnsw_all could not construct connected graph");
    }
    let kgraph = kgraph_res.unwrap();
    //
    kgraph
} // end of get_kgraph

//

// construct kgraph case not hierarchical
fn get_kgraph_projection<Dist>(
    data_with_id: &[(&Vec<f64>, usize)],
    hnswparams: &HnswParams,
    nb_layer: usize,
    layer_proj: usize,
) -> KGraphProjection<f64>
where
    Dist: Distance<f64> + Default + Send + Sync,
{
    //
    let nb_data = data_with_id.len();
    let hnsw = Hnsw::<f64, Dist>::new(
        hnswparams.max_conn,
        nb_data,
        nb_layer,
        hnswparams.ef_c,
        Dist::default(),
    );
    hnsw.parallel_insert(data_with_id);
    hnsw.dump_layer_info();
    KGraphProjection::<f64>::new(&hnsw, hnswparams.knbn, layer_proj)
} // end of get_kgraph_projection

//

// dispatching according to distance ... use a macro
fn get_kgraph_with_distname(
    data_with_id: &[(&Vec<f64>, usize)],
    hnswparams: &HnswParams,
    nb_layer: usize,
) -> KGraph<f64> {
    let kgraph = match hnswparams.distance.as_str() {
        "DistL2" => get_kgraph::<DistL2>(data_with_id, hnswparams, nb_layer),
        "DistL1" => get_kgraph::<DistL1>(data_with_id, hnswparams, nb_layer),
        "DistJeffreys" => get_kgraph::<DistJeffreys>(data_with_id, hnswparams, nb_layer),
        "DistCosine" => get_kgraph::<DistCosine>(data_with_id, hnswparams, nb_layer),
        "DistJensenShannon" => get_kgraph::<DistJensenShannon>(data_with_id, hnswparams, nb_layer),
        _ => {
            log::error!("unknown distance : {}", hnswparams.distance);
            std::process::exit(1);
        }
    };
    kgraph
} // end of get_kgraph_with_distname

fn get_kgraphproj_with_distname(
    data_with_id: &[(&Vec<f64>, usize)],
    hnswparams: &HnswParams,
    nb_layer: usize,
    layer_proj: usize,
) -> KGraphProjection<f64> {
    //
    let kgraph_projection = match hnswparams.distance.as_str() {
        "DistL2" => get_kgraph_projection::<DistL2>(data_with_id, hnswparams, nb_layer, layer_proj),
        "DistL1" => get_kgraph_projection::<DistL1>(data_with_id, hnswparams, nb_layer, layer_proj),
        "DistJeffreys" => {
            get_kgraph_projection::<DistJeffreys>(data_with_id, hnswparams, nb_layer, layer_proj)
        }

        "DistCosine" => {
            get_kgraph_projection::<DistCosine>(data_with_id, hnswparams, nb_layer, layer_proj)
        }
        _ => {
            log::error!("unknown distance : {}", hnswparams.distance);
            std::process::exit(1);
        }
    };
    kgraph_projection
} // end of get_kgraphproj_with_distname
