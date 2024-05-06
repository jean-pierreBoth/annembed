//! Structure and functions to read MNIST digits database
//! To run the examples change the line :  
//!
//! const MNIST_DIGITS_DIR : &'static str = "/home/jpboth/Data/MNIST/";
//!
//! to whatever directory you downloaded the [MNIST digits data](http://yann.lecun.com/exdb/mnist/)

use ndarray::{s, Array1, Array3};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;

//use anndists::dist::*;
use hnsw_rs::prelude::*;

use annembed::prelude::*;

/// A struct to load/store [MNIST data](http://yann.lecun.com/exdb/mnist/)  
/// stores labels (i.e : digits between 0 and 9) coming from file train-labels-idx1-ubyte      
/// and hand written characters as 28*28 images with values between 0 and 255 coming from train-images-idx3-ubyte
pub struct MnistData {
    _image_filename: String,
    _label_filename: String,
    images: Array3<u8>,
    labels: Array1<u8>,
}

impl MnistData {
    pub fn new(image_filename: String, label_filename: String) -> std::io::Result<MnistData> {
        let image_path = PathBuf::from(image_filename.clone());
        let image_file = OpenOptions::new().read(true).open(&image_path)?;
        let mut image_io = BufReader::new(image_file);
        let images = read_image_file(&mut image_io);
        // labels
        let label_path = PathBuf::from(label_filename.clone());
        let labels_file = OpenOptions::new().read(true).open(&label_path)?;
        let mut labels_io = BufReader::new(labels_file);
        let labels = read_label_file(&mut labels_io);
        Ok(MnistData {
            _image_filename: image_filename,
            _label_filename: label_filename,
            images,
            labels,
        })
    } // end of new for MnistData

    /// returns labels of images. lables[k] is the label of the k th image.
    pub fn get_labels(&self) -> &Array1<u8> {
        &self.labels
    }

    /// returns images. images are stored in Array3 with Array3[[.., .., k]] being the k images!
    /// Each image is stored as it is in the Mnist files, Array3[[i, .., k]] is the i row of the k image
    pub fn get_images(&self) -> &Array3<u8> {
        &self.images
    }
} // end of impl MnistData

pub fn read_image_file(io_in: &mut dyn Read) -> Array3<u8> {
    // read 4 bytes magic
    let magic: u32;
    // to read 32 bits in network order!
    let toread: u32 = 0;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    magic = u32::from_be(toread);
    assert_eq!(magic, 2051);
    // read nbitems
    let nbitem: u32;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    nbitem = u32::from_be(toread);
    assert!(nbitem == 60000 || nbitem == 10000);
    //  read nbrow
    let nbrow: u32;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    nbrow = u32::from_be(toread);
    assert_eq!(nbrow, 28);
    // read nbcolumns
    let nbcolumn: u32;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    nbcolumn = u32::from_be(toread);
    assert_eq!(nbcolumn, 28);
    // for each item, read a row of nbcolumns u8
    let mut images = Array3::<u8>::zeros((nbrow as usize, nbcolumn as usize, nbitem as usize));
    let mut datarow = Vec::<u8>::new();
    datarow.resize(nbcolumn as usize, 0);
    for k in 0..nbitem as usize {
        for i in 0..nbrow as usize {
            let it_slice;
            it_slice = datarow.as_mut_slice();
            io_in.read_exact(it_slice).unwrap();
            let mut smut_ik = images.slice_mut(s![i, .., k]);
            assert_eq!(nbcolumn as usize, it_slice.len());
            assert_eq!(nbcolumn as usize, smut_ik.len());
            for j in 0..smut_ik.len() {
                smut_ik[j] = it_slice[j];
            }
            //    for j in 0..nbcolumn as usize {
            //        *(images.get_mut([i,j,k]).unwrap()) = it_slice[j];
            //   }
            // how do a block copy from read slice to view of images.
            // images.slice_mut(s![i as usize, .. , k as usize]).assign(&Array::from(it_slice)) ;
        }
    }
    images
} // end of readImageFile

pub fn read_label_file(io_in: &mut dyn Read) -> Array1<u8> {
    let magic: u32;
    // to read 32 bits in network order!
    let toread: u32 = 0;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    magic = u32::from_be(toread);
    assert_eq!(magic, 2049);
    // read nbitems
    let nbitem: u32;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    nbitem = u32::from_be(toread);
    assert!(nbitem == 60000 || nbitem == 10000);
    let mut labels_vec = Vec::<u8>::new();
    labels_vec.resize(nbitem as usize, 0);
    io_in.read_exact(&mut labels_vec).unwrap();
    let labels = Array1::from(labels_vec);
    labels
} // end of fn read_label

//============================================================================================

use csv::*;

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use annembed::fromhnsw::hubness;
use annembed::fromhnsw::kgproj::KGraphProjection;
use annembed::fromhnsw::kgraph::{kgraph_from_hnsw_all, KGraph};

const MNIST_DIGITS_DIR: &'static str = "/home/jpboth/Data/ANN/MNIST/";

pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let mut image_fname = String::from(MNIST_DIGITS_DIR);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_DIGITS_DIR);
    label_fname.push_str("train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    let mut images_as_v: Vec<Vec<f32>>;
    let mut labels: Vec<u8>;
    {
        let mnist_train_data = MnistData::new(image_fname, label_fname).unwrap();
        let images = mnist_train_data.get_images();
        labels = mnist_train_data.get_labels().to_vec();
        let (_, _, nbimages) = images.dim();
        //
        images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        for k in 0..nbimages {
            let v: Vec<f32> = images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32)
                .collect();
            images_as_v.push(v);
        }
    } // drop mnist_train_data
      // now read test data
    let mut image_fname = String::from(MNIST_DIGITS_DIR);
    image_fname.push_str("t10k-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_DIGITS_DIR);
    label_fname.push_str("t10k-labels-idx1-ubyte");
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    {
        let mnist_test_data = MnistData::new(image_fname, label_fname).unwrap();
        let test_images = mnist_test_data.get_images();
        let mut test_labels = mnist_test_data.get_labels().to_vec();
        let (_, _, nbimages) = test_images.dim();
        let mut test_images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        //
        for k in 0..nbimages {
            let v: Vec<f32> = test_images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32)
                .collect();
            test_images_as_v.push(v);
        }
        labels.append(&mut test_labels);
        images_as_v.append(&mut test_images_as_v);
    } // drop mnist_test_data

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
    // for i in 0..2000 {
    //     hnsw.insert(data_with_id[i]);
    // }
    // images as vectors of f32 and send to hnsw
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        " ann construction sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    hnsw.dump_layer_info();
    //
    let kgraph: KGraph<f32>;
    let graphprojection: KGraphProjection<f32>;
    //
    let mut embed_params = EmbedderParams::default();
    embed_params.nb_grad_batch = 30;
    embed_params.scale_rho = 1.;
    embed_params.beta = 1.;
    embed_params.b = 1.;
    embed_params.grad_step = 1.;
    embed_params.nb_sampling_by_edge = 10;
    embed_params.dmap_init = true;
    //
    let mut embedder;
    let hierarchical = false;
    if !hierarchical {
        let knbn = 6;
        log::info!("calling kgraph.init_from_hnsw_all");
        kgraph = kgraph_from_hnsw_all(&hnsw, knbn).unwrap();
        log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
        embedder = Embedder::new(&kgraph, embed_params);
        let embed_res = embedder.embed();
        assert!(embed_res.is_ok());
        assert!(embedder.get_embedded().is_some());
    } else {
        log::info!("graph projection");
        embed_params.nb_grad_batch = 20;
        let knbn = 6;
        embed_params.grad_factor = 4; // default in fact
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
    log::info!("dumping initial embedding in csv file");
    let mut csv_w = Writer::from_path("mnist_init_digits.csv").unwrap();
    // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
    let _res = write_csv_labeled_array2(
        &mut csv_w,
        labels.as_slice(),
        &embedder.get_initial_embedding_reindexed(),
    );
    csv_w.flush().unwrap();

    log::info!("dumping in csv file");
    let mut csv_w = Writer::from_path("mnist_digits.csv").unwrap();
    // we can use get_embedded_reindexed as we indexed DataId contiguously in hnsw!
    let _res = write_csv_labeled_array2(
        &mut csv_w,
        labels.as_slice(),
        &embedder.get_embedded_reindexed(),
    );
    csv_w.flush().unwrap();
    //
    let _quality = embedder.get_quality_estimate_from_edge_length(100);
    //
    // Get some statistics on induced graph. This is not related to the embedding process
    //
    let knbn = 25;
    let kgraph: KGraph<f32>;
    kgraph = kgraph_from_hnsw_all(&hnsw, knbn).unwrap();
    log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
    log::info!("dimension estimation...");
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let sampling_size = 10000;
    let dim_stat = kgraph.estimate_intrinsic_dim(sampling_size);
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        " dimension estimation sys time(ms) : {:.3e},  cpu time(ms) {:?}",
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
    // hubness estimation
    //
    let hubness = hubness::Hubness::new(&kgraph);
    let s3_hubness = hubness.get_standard3m();
    log::info!("\n graph hubness asymetry estimation : {:.3e}", s3_hubness);
    println!(
        "\n graph hubness asymetry estimation : {:.3e} \n",
        s3_hubness
    );
    let _histo = hubness.get_hubness_histogram();
    // get the DataId of the first points largest hubness in deacresing order
    let _largest = hubness.get_largest_hubs_by_dataid(10);
} // end of main digits

//============================================================================================

#[cfg(test)]

mod tests {

    use super::*;

    // test and compare some values obtained with Julia loading

    #[test]

    fn test_load_mnist() {
        let mut image_fname = String::from(MNIST_DIGITS_DIR);
        image_fname.push_str("train-images-idx3-ubyte");
        let image_path = PathBuf::from(image_fname.clone());
        let image_file_res = OpenOptions::new().read(true).open(&image_path);
        if image_file_res.is_err() {
            println!("could not open image file : {:?}", image_fname);
            return;
        }

        let mut label_fname = String::from(MNIST_DIGITS_DIR);
        label_fname.push_str("train-labels-idx1-ubyte");
        let label_path = PathBuf::from(label_fname.clone());
        let label_file_res = OpenOptions::new().read(true).open(&label_path);
        if label_file_res.is_err() {
            println!("could not open label file : {:?}", label_fname);
            return;
        }

        let mnist_data = MnistData::new(image_fname, label_fname).unwrap();
        assert_eq!(0x3c, *mnist_data.images.get([9, 14, 9]).unwrap());
        assert_eq!(0xfd, mnist_data.images[(14, 9, 9)]);
        // check some value of the tenth images

        // check first and last labels
        assert_eq!(5, mnist_data.labels[0]);
        assert_eq!(8, mnist_data.labels[mnist_data.labels.len() - 1]);
        assert_eq!(1, 1);
    } // end test_load
} // end module tests
