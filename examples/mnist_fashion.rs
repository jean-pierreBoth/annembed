//! Structure and functions to read MNIST fashion database
//! To run the examples change the line :  
//!
//! const MNIST_FASHION_DIR : &'static str = "/home.1/jpboth/Data/Fashion-MNIST/";
//!
//! The data can be downloaded in the same format as the FASHION database from:  
//!
//! <https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion>
//!

use ndarray::s;

use hnsw_rs::prelude::*;

use annembed::prelude::*;

use annembed::utils::mnistio::*;

//============================================================================================

use csv::*;

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use annembed::fromhnsw::hubness;
use annembed::fromhnsw::kgproj::KGraphProjection;
use annembed::fromhnsw::kgraph::{KGraph, kgraph_from_hnsw_all};

const MNIST_FASHION_DIR: &str = "/home/jpboth/Data/ANN/Fashion-MNIST/";

#[allow(clippy::range_zip_with_len)]
pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let fashion_data = load_mnist_train_data(MNIST_FASHION_DIR).unwrap();
    let mut labels = fashion_data.get_labels().to_vec();
    let images = fashion_data.get_images();
    // convert images as vectors
    let (_, _, nbimages) = images.dim();
    let mut images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
    //
    for k in 0..nbimages {
        let v: Vec<f32> = images
            .slice(s![.., .., k])
            .iter()
            .map(|v| *v as f32)
            .collect();
        images_as_v.push(v);
    }
    // load test data
    // ===============
    let fashion_test_data = load_mnist_test_data(MNIST_FASHION_DIR).unwrap();
    labels.append(&mut fashion_test_data.get_labels().to_vec());
    let images = fashion_test_data.get_images();
    // convert images as vectors
    let (_, _, nbimages) = images.dim();
    //
    for k in 0..nbimages {
        let v: Vec<f32> = images
            .slice(s![.., .., k])
            .iter()
            .map(|v| *v as f32)
            .collect();
        images_as_v.push(v);
    }
    //
    let ef_c = 200;
    let max_nb_connection = 16;
    let nbimages = images_as_v.len();
    let nb_layer = 16.min((nbimages as f32).ln().trunc() as usize);
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let mut hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nbimages, nb_layer, ef_c, DistL2 {});
    hnsw.set_keeping_pruned(true);
    // we must pay fortran indexation once!. transform image to a vector
    let data_with_id: Vec<(&Vec<f32>, usize)> =
        images_as_v.iter().zip(0..images_as_v.len()).collect();
    hnsw.parallel_insert(&data_with_id);
    // images as vectors of f32 and send to hnsw
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        " ann construction sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    hnsw.dump_layer_info();
    //
    let mut embed_params = EmbedderParams::default();
    embed_params.nb_grad_batch = 25;
    embed_params.scale_rho = 1.;
    embed_params.beta = 1.;
    embed_params.grad_step = 1.;
    embed_params.nb_sampling_by_edge = 10;
    embed_params.dmap_init = true;
    embed_params.hubness_weighting = false;
    // to check the impact of dimension
    //    embed_params.asked_dim = 15;

    let mut embedder;
    let kgraph;
    let graphprojection;
    let hierarchical = false;
    if !hierarchical {
        let knbn = 12;
        kgraph = kgraph_from_hnsw_all(&hnsw, knbn).unwrap();
        embedder = Embedder::new(&kgraph, embed_params);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
        assert!(embedder.get_embedded().is_some());
    } else {
        let knbn = 6;
        embed_params.nb_grad_batch = 20;
        log::debug!("trying graph projection");
        embed_params.grad_factor = 4;
        graphprojection = KGraphProjection::<f32>::new(&hnsw, knbn, 1);
        embedder = Embedder::from_hkgraph(&graphprojection, embed_params);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
        assert!(embedder.get_embedded().is_some());
    }
    println!(
        " ann embed time time {:.2e} s, cpu time : {}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_start.elapsed().as_secs()
    );
    // dump
    if embed_params.dmap_init {
        log::info!("dumping initial embedding in csv file");
        let mut csv_w = Writer::from_path("mnist_init_fashion.csv").unwrap();
        // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
        let _res = write_csv_labeled_array2(
            &mut csv_w,
            labels.as_slice(),
            &embedder.get_initial_embedding_reindexed(),
        );
        csv_w.flush().unwrap();
    }
    log::info!("dumping in csv file");
    let mut csv_w = Writer::from_path("mnist_fashion.csv").unwrap();
    // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
    let _res = write_csv_labeled_array2(
        &mut csv_w,
        labels.as_slice(),
        &embedder.get_embedded_reindexed(),
    );
    csv_w.flush().unwrap();
    //
    let _quality = embedder.get_quality_estimate_from_edge_length(50);
    // Get some statistics on induced graph. This is not related to the embedding process
    let knbn = 25;
    let kgraph: KGraph<f32> = kgraph_from_hnsw_all(&hnsw, knbn).unwrap();
    log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
    //
    //
    //
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
    //
    let sampling_size = 10000;
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let dim_stat = kgraph.estimate_intrinsic_dim_2nn(sampling_size);
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        "\n dimension estimation sys time(ms) : {:.5e},  cpu time(ms) {:.5e}\n",
        sys_now.elapsed().unwrap().as_millis(),
        cpu_time.as_millis()
    );
    if dim_stat.is_ok() {
        let dim_stat = dim_stat.unwrap();
        log::info!(
            "\n estimate_intrinsic_dim_2nn dimension estimation with nbpoints : {}, dim : {:.3e} \n",
            sampling_size,
            dim_stat,
        );
        println!(
            " estimate_intrinsic_dim_2nn dimension estimation with nbpoints : {}, dim : {:.3e}",
            sampling_size, dim_stat
        );
    }
    //
    // hubness estimation. We can get hubness from embedder if we asked for hubness weighting of nodes in embedding
    //
    let hubness = match embedder.get_hubness() {
        Some(hubness) => hubness,
        None => &hubness::Hubness::new(&kgraph),
    };
    let _histo = hubness.get_hubness_histogram();
    // get the DataId of the first points largest hubness in deacresing order
    let largest = hubness.get_largest_hubs_by_dataid(20);
    // get labels of largest hubs
    println!(" largest hubs, id, count and labels");
    println!("   id      count    label");
    for (id, count) in largest {
        println!("  {:>5}    {:>5}    {:>5} ", id, count, labels[id]);
    }
    // hubness asymetry.
    let s3_hubness = hubness.get_standard3m();
    log::info!(
        "\n graph hubness asymetry estimation (Radovanic 2010) : {:.3e}",
        s3_hubness
    );
    println!(
        "\n graph hubness asymetry estimation  (Radovanic 2010): {:.3e} \n",
        s3_hubness
    );
    //
    let _kgraph_stats = kgraph.get_kraph_stats();
} // end of main

//============================================================================================

#[cfg(test)]

mod tests {

    use super::*;

    // test and compare some values obtained with Julia loading

    #[test]
    fn test_load_mnist_fashion() {
        let mut image_fname = String::from(MNIST_FASHION_DIR);
        image_fname.push_str("train-images-idx3-ubyte");
        let image_path = PathBuf::from(image_fname.clone());
        let image_file_res = OpenOptions::new().read(true).open(&image_path);
        if image_file_res.is_err() {
            println!("could not open image file : {:?}", image_fname);
            return;
        }

        let mut label_fname = String::from(MNIST_FASHION_DIR);
        label_fname.push_str("train-labels-idx1-ubyte");
        let label_path = PathBuf::from(label_fname.clone());
        let label_file_res = OpenOptions::new().read(true).open(&label_path);
        if label_file_res.is_err() {
            println!("could not open label file : {:?}", label_fname);
            return;
        }

        let _mnist_data = MnistData::new(image_fname, label_fname).unwrap();
        // check some value of the tenth images
    } // end test_load
} // end module tests
