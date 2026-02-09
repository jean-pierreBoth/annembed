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
    for (l, p) in points.as_slice() {
        log::info!("\n\n label : {}, point : {}", l, p);
        let (mean, cdc_at_point) = cdc.get_cdc_at_point(*p);
        // get dist between mean and data point
        let dist_to_mean = distl2.eval(mean.as_slice().unwrap(), &images_as_v[*p]);

        //
        let neighbours = hnsw.search(&images_as_v[*p], 12, 50);
        let first_dist = neighbours[1].distance;
        let last_dist = neighbours.last().unwrap().distance;
        log::info!(
            "dist between kernel at point and mean: {:.3e} and first neighbour {:.3e}, last neighbour {:.3e}",
            dist_to_mean,
            first_dist,
            last_dist
        );
        cdc_points.push((mean, cdc_at_point));
    }
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        "cdc work sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    // distances between points of differents labels compared with distances between cdc operaror
    //
    let p_slice = points.as_slice();
    let mut p_dist_vec = Vec::<f32>::new();
    p_dist_vec.reserve(p_slice.len());

    let mut cdc_dist_vec = Vec::<f32>::with_capacity(p_slice.len());
    //
    for i in 0..p_slice.len() {
        let (l_i, idx_i) = p_slice.get_index(i).unwrap();
        for j in 0..i {
            let (l_j, idx_j) = p_slice.get_index(j).unwrap();
            let dist_p = distl2.eval(&images_as_v[*idx_i], &images_as_v[*idx_j]);
            let dist_cdc = psd_dist(&cdc_points[i].1, &cdc_points[j].1);
            p_dist_vec.push(dist_p);
            cdc_dist_vec.push(dist_cdc);
            log::info!(
                "dist between lables ({}, {}), point dist : {:.3e}, cdc dist {:.3e}",
                *l_i,
                *l_j,
                dist_p,
                dist_cdc
            );
        }
    }
    log::info!(
        "correlation between dists : {:.3e}",
        correlation(&p_dist_vec, &cdc_dist_vec)
    );
}

/// choose index corresponding to different digits
fn choose_points(labels: &[u8]) -> IndexMap<u8, usize> {
    // choose one point for each label
    let nb_digits: usize = 10;
    let mut index = IndexMap::<u8, usize>::with_capacity(nb_digits);
    //
    let mut labels_it = labels.iter().enumerate();
    loop {
        if let Some((rank, key)) = labels_it.next()
            && !index.contains_key(key)
        {
            index.insert(*key, rank);
            log::info!("inserting digits : {}, rank : {}", *key, rank);
            if index.len() >= nb_digits {
                break;
            }
        };
    }
    //
    index
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.len() == 1 {
        return 0.0;
    }
    //
    let mean_a: f32 = a.iter().sum::<f32>() / a.len() as f32;
    let mean_b = b.iter().sum::<f32>() / b.len() as f32;
    let mut cov_ab = 0.0f32;
    let mut std_dev_a = 0.0f32;
    let mut std_dev_b = 0.0f32;
    for i in 0..a.len() {
        cov_ab += (a[i] - mean_a) * (b[i] - mean_b);
        std_dev_a += (a[i] - mean_a) * (a[i] - mean_a);
        std_dev_b += (b[i] - mean_b) * (b[i] - mean_b);
    }
    cov_ab / (std_dev_a * std_dev_b).sqrt()
}
