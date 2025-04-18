//! annembed binary.  
//!
//! This module provides just access to floating point data embedding.  
//! Command syntax is embed input --csv csvfile  [--outfile | -o  output_name] [--delim u8] [various embedding parameters] [hnsw params] .  
//!
//!  --outfile or -o to specify the name of csv file containing embedded vectors. By default the name is "embedded.csv"
//!
//! hnsw is an optional subcommand to change default parameters of the Hnsw structure. See [hnsw_rs](https://crates.io/crates/hnsw_rs).  
//!
//! - Parameters for embedding.  
//!     The options are optional and give access to some fields of the [EmbedderParams] structure.  
//!
//!     --batch    : optional, a integer giving the number of batch to run. Default to 15.  
//!     --stepg    : optional, a float value , initial gradient step, default is 2.  
//!     --scale    : optional, a float value, scale modification factor, default is 1.  
//!     --nbsample : optional, a number of edge sampling , default is 10   
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
//!     --scale_modification_f : scale factor to control Hierarchy of HNSW for high dimensional datasets (e.g., d > 32).
//!
//! The csv file must have one record by vector to embed. The default delimiter is ','.  
//! The output is a csv file with embedded vectors.  
//! The Julia directory provides helpers to get Persistence diagrams and barcodes and vizualize them using Ripserer.jl

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use anyhow::anyhow;
use clap::{Arg, ArgAction, ArgMatches, Command};

use hnsw_rs::prelude::*;

use annembed::fromhnsw::hubness;
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
    //scale_modification factor, must be [0.2, 1]
    scale_modification : f64,
} // end of struct HnswParams

impl HnswParams {
    pub fn my_default() -> Self {
        HnswParams {
            max_conn: 64,
            ef_c: 512,
            knbn: 10,
            distance: String::from("DistL2"),
            scale_modification: 1.0,
        }
    }

    #[allow(unused)]
    pub fn new(max_conn: usize, ef_c: usize, knbn: usize, distance: String, scale_modification: f64) -> Self {
        HnswParams {
            max_conn,
            ef_c,
            knbn,
            distance,
            scale_modification,
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
    hnswparams.scale_modification = *matches.get_one::<f64>("scale_modification").unwrap();

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
fn parse_embed_group(
    matches: &ArgMatches,
) -> Result<(EmbedderParams, Option<QualityParams>), anyhow::Error> {
    log::debug!("in parse_embed_group");
    //
    let mut embedparams = EmbedderParams::default();
    //
    embedparams.nb_grad_batch = *matches.get_one::<usize>("batch").unwrap();
    embedparams.asked_dim = *matches.get_one::<usize>("dimension").unwrap();
    embedparams.scale_rho = *matches.get_one::<f64>("scale").unwrap();
    embedparams.nb_sampling_by_edge = *matches.get_one::<usize>("nbsample").unwrap();
    embedparams.hierarchy_layer = *matches.get_one::<usize>("hierarchy").unwrap();
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
    Ok((embedparams, quality))
} // end of parse_embed_cmd

//

#[allow(clippy::range_zip_with_len)]
pub fn main() {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");
    //
    let hnswparams: HnswParams;
    let embedparams: EmbedderParams;
    //
    let hnswcmd = Command::new("hnsw")
        .about("Build HNSW graph")
        .arg(Arg::new("dist")
            .long("dist")
            .short('d')
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(String))
            .help("Distance type is required, must be one of   \"DistL1\" , \"DistL2\", \"DistCosine\" and \"DistJeyffreys\"  "))
        .arg(Arg::new("nbconn")
            .long("nbconn")
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("Maximum number of build connections allowed (M in HNSW)"))
        .arg(Arg::new("ef")
            .long("ef")
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("Build factor ef_construct in HNSW"))
        .arg(Arg::new("scale_modification")
            .long("scale_modify_f")
            .help("Hierarchy scale modification factor in HNSW/HubNSW or FlatNav, must be in [0.2,1]")
            .value_name("scale_modify")
            .default_value("1.0")
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(f64)))
        .arg(Arg::new("knbn")
            .long("knbn")
            .required(true)
            .action(ArgAction::Set)
            .value_parser(clap::value_parser!(usize))
            .help("Number of k-nearest neighbours to be retrieved for embedding"));

    //
    // Now the command line
    // ===================
    //
    let matches = Command::new("annembed")
        //        .subcommand_required(true)
        .arg_required_else_help(true)
        .about("Non-linear Dimension Reduction/Embedding via Approximate Nearest Neighbor Graph")
        .arg(
            Arg::new("csvfile")
                .long("csv")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("Expecting a csv file"),
        )
        .arg(
            Arg::new("outfile")
                .long("out")
                .short('o')
                .required(false)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .help("Output file name"),
        )
        .arg(
            Arg::new("delim")
                .long("delim")
                .short('d')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(char))
                .help("Delimiter can be ' ', ','"),
        )
        // ann group flags
        .arg(
            Arg::new("batch")
                .required(false)
                .long("batch")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("20")
                .help("Number of batches to run"),
        )
        .arg(
            Arg::new("grap_step")
                .required(false)
                .long("stepg")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .help("Number of gradient descent steps"),
        )
        .arg(
            Arg::new("nbsample")
                .required(false)
                .long("nbsample")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("10")
                .help("Number of edge sampling"),
        )
        .arg(
            Arg::new("hierarchy")
                .required(false)
                .long("layer")
                .short('l')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("0")
                .help("A layer num"),
        )
        .arg(
            Arg::new("scale")
                .required(false)
                .long("scale")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .default_value("1.0")
                .help("Spatial scale factor"),
        )
        .arg(
            Arg::new("dimension")
                .required(false)
                .long("dim")
                .short('d')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .default_value("2")
                .help("Dimension of embedding"),
        )
        .arg(
            Arg::new("quality")
                .required(false)
                .long("quality")
                .short('q')
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .help("Sampling fraction, should <= 1."),
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
    let res = parse_embed_group(&matches);
    match res {
        Ok((e_params, asked_quality)) => {
            embedparams = e_params;
            quality = asked_quality;
        }
        _ => {
            log::error!("parsing embed cmd failed");
            println!("exiting with error {}", res.err().as_ref().unwrap());
            //  log::error!("exiting with error {}", res.err().unwrap());
            std::process::exit(1);
        }
    }

    embedparams.log();

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
    if embedparams.get_hierarchy_layer() == 0 {
        let hubdim = true; // to get hubness and intrinsic dimension info
        let kgraph = get_kgraph_with_distname(&data_with_id, &hnswparams, nb_layer, hubdim);
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " graph construction sys time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_time.as_secs()
        );
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
        //
        if quality.is_some() {
            let _quality = embedder.get_quality_estimate_from_edge_length(100);
        }
    }
    // end not hierarchical
    else {
        let graphprojection = get_kgraphproj_with_distname(
            &data_with_id,
            &hnswparams,
            nb_layer,
            embedparams.get_hierarchy_layer(),
        );
        let mut embedder = Embedder::from_hkgraph(&graphprojection, embedparams);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
        assert!(embedder.get_embedded().is_some());
        let _res = write_csv_array2(&mut csv_w, &embedder.get_embedded_reindexed());
        csv_w.flush().unwrap();
        //
        if quality.is_some() {
            let _quality = embedder.get_quality_estimate_from_edge_length(100);
        }
    }
} // end of main

//==========================================================================

// construct kgraph case not hierarchical
fn get_kgraph<Dist>(
    data_with_id: &[(&Vec<f64>, usize)],
    hnswparams: &HnswParams,
    nb_layer: usize,
    hubdim_asked: bool,
) -> KGraph<f64>
where
    Dist: Distance<f64> + Default + Send + Sync,
{
    //
    let nb_data = data_with_id.len();
    let mut hnsw = Hnsw::<f64, Dist>::new(
        hnswparams.max_conn,
        nb_data,
        nb_layer,
        hnswparams.ef_c,
        Dist::default(),
    );
    hnsw.modify_level_scale(hnswparams.scale_modification);
    hnsw.parallel_insert(data_with_id);
    hnsw.dump_layer_info();
    let kgraph_res = kgraph_from_hnsw_all(&hnsw, hnswparams.knbn);
    if kgraph_res.is_err() {
        panic!("kgraph_from_hnsw_all could not construct connected graph");
    }
    let kgraph = kgraph_res.unwrap();
    if hubdim_asked {
        // hubness and intrinsic dimension.
        log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
        log::info!("dimension estimation...");
        let sampling_size = 10000;
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        let dim_stat = kgraph.estimate_intrinsic_dim(sampling_size);
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            "\n dimension estimation sys time(ms) : {:.3e},  cpu time(ms) {:.3e}\n",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_time.as_millis()
        );
        if dim_stat.is_ok() {
            let dim_stat = dim_stat.unwrap();
            log::info!(
                "\n dimension estimation with nbpoints : {}, dim : {:.3e}, sigma = {:.3e} \n",
                sampling_size,
                dim_stat.0,
                dim_stat.1
            );
            println!(
                " dimension estimation with nbpoints : {}, dim : {:.3e}, sigma = {:.3e}",
                sampling_size, dim_stat.0, dim_stat.1
            );
        }
        // hubness estimation
        let hubness = hubness::Hubness::new(&kgraph);
        let s3_hubness = hubness.get_standard3m();
        log::info!("\n graph hubness estimation : {:.3e}", s3_hubness);
        println!("\n graph hubness estimation : {:.3e} \n", s3_hubness);
        let _histo = hubness.get_hubness_histogram();
        let _kgraph_stats = kgraph.get_kraph_stats();
    }
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
    let mut hnsw = Hnsw::<f64, Dist>::new(
        hnswparams.max_conn,
        nb_data,
        nb_layer,
        hnswparams.ef_c,
        Dist::default(),
    );
    hnsw.modify_level_scale(hnswparams.scale_modification);
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
    hubdim: bool,
) -> KGraph<f64> {
    let kgraph = match hnswparams.distance.as_str() {
        "DistL2" => get_kgraph::<DistL2>(data_with_id, hnswparams, nb_layer, hubdim),
        "DistL1" => get_kgraph::<DistL1>(data_with_id, hnswparams, nb_layer, hubdim),
        "DistJeffreys" => get_kgraph::<DistJeffreys>(data_with_id, hnswparams, nb_layer, hubdim),
        "DistCosine" => get_kgraph::<DistCosine>(data_with_id, hnswparams, nb_layer, hubdim),
        "DistJensenShannon" => {
            get_kgraph::<DistJensenShannon>(data_with_id, hnswparams, nb_layer, hubdim)
        }
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
