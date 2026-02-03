//! Run Carre Du Champ computations on Mnist Digits data

use indexmap::IndexMap;
use ndarray::{Array1, Array2, s};

use hnsw_rs::prelude::*;

use annembed::prelude::*;
use annembed::utils::mnistio::*;

//============================================================================================

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

const MNIST_DIGITS_DIR: &str = "/home/jpboth/Data/ANN/MNIST/";

#[allow(clippy::range_zip_with_len)]
pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let digits_data = load_mnist_train_data(MNIST_DIGITS_DIR).unwrap();
    let mut labels = digits_data.get_labels().to_vec();
    let images = digits_data.get_images();
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
    //
    // load test data
    // ===============
    let digits_test_data = load_mnist_test_data(MNIST_DIGITS_DIR).unwrap();
    labels.append(&mut digits_test_data.get_labels().to_vec());
    let images = digits_test_data.get_images();
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
    let ef_c = 50;
    let max_nb_connection = 70;
    let nbimages = images_as_v.len();
    let nb_layer = 16.min((nbimages as f32).ln().trunc() as usize);
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nbimages, nb_layer, ef_c, DistL2 {});
    // we must pay fortran indexation once!. transform image to a vector
    let data_with_id: Vec<(&Vec<f32>, usize)> =
        images_as_v.iter().zip(0..images_as_v.len()).collect();
    hnsw.parallel_insert(&data_with_id);
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        " ann construction sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    //
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    //
    let cdc = CarreDuChamp::from_hnsw_ref(&hnsw);
    // choose some images at different labels.
    let points = choose_points(&labels);
    let mut cdc_points = Vec::<(Array1<f32>, Array2<f32>)>::with_capacity(points.len());
    let distl2 = DistL2 {};
    for (l, p) in points {
        log::info!("\n\n label : {}, point : {}", l, p);
        let (mean, cdc_at_point) = cdc.get_cdc_at_point(p);
        // get dist between mean and data point
        let dist_to_mean = distl2.eval(&mean.as_slice().unwrap(), &images_as_v[p]);
        log::info!(
            "dist between point and mean of kernel at point : {:.3e}",
            dist_to_mean
        );
        cdc_points.push((mean, cdc_at_point));
    }
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        "cdc work sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    //
}

/// choose index corresponding to different digits
fn choose_points(labels: &Vec<u8>) -> IndexMap<u8, usize> {
    // choose one point for each label
    let nb_digits: usize = 10;
    let mut index = IndexMap::<u8, usize>::with_capacity(nb_digits);
    //
    let mut labels_it = labels.into_iter().enumerate();
    loop {
        if let Some((rank, key)) = labels_it.next() {
            if !index.contains_key(key) {
                index.insert(*key, rank);
                log::info!("inserting digits : {}, rank : {}", *key, rank);
                if index.len() >= nb_digits {
                    break;
                }
            }
        };
    }
    //
    index
}
