//! Some tests at dumping sparse matrix distances between points using the Hnsw structure.
//! Hnsw gives us
//!      - a graph projection enabling dump of sparse distance matrix between sampled points from the data
//!      - the possibility to extract the neighbourhood of a point and extract a small matrix of distance between points around a node.
//!
//! The small Julia module in the crate, using the **Ripserer module** associated can reload these matrices
//! and computes homology on these extracted data and dumps persistence graphics
//!
//! Change the variable locating mnist data files to your convenience.
//!

use ndarray::{s, Array1, Array3};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;

use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

//use anndists::dist::*;
use hnsw_rs::prelude::*;

// The directory where the files t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte
// and train-labels-idx1-ubyte reside
// one directory for mnist_digits and a directory for fashion_mnist
const MNIST_DATA_DIR: &'static str = "/home/jpboth/Data/Fashion-MNIST/";

/// A struct to load/store for Fashion Mnist in the same format as [MNIST data](http://yann.lecun.com/exdb/mnist/)  
/// stores labels (i.e : FASHION between 0 and 9) coming from file train-labels-idx1-ubyte      
/// and objects as 28*28 images with values between 0 and 255 coming from train-images-idx3-ubyte
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
    // to read 32 bits in network order!
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let magic = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 2051);
    // read nbitems
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbitem = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert!(nbitem == 60000 || nbitem == 10000);
    //  read nbrow
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbrow = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert_eq!(nbrow, 28);
    // read nbcolumns
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbcolumn = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
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
    // to read 32 bits in network order!
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let magic = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert_eq!(magic, 2049);
    // read nbitems
    let mut it_slice = vec![0; ::std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice).unwrap();
    let nbitem = Cursor::new(it_slice).read_u32::<BigEndian>().unwrap();
    assert!(nbitem == 60000 || nbitem == 10000);
    let mut labels_vec = Vec::<u8>::new();
    labels_vec.resize(nbitem as usize, 0);
    io_in.read_exact(&mut labels_vec).unwrap();
    let labels = Array1::from(labels_vec);
    labels
} // end of fn read_label

//  end of reading Mnist

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use annembed::fromhnsw::toripserer::ToRipserer;

pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let mut image_fname = String::from(MNIST_DATA_DIR);
    log::info!(" treating data from dir : {}", MNIST_DATA_DIR);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_DATA_DIR);
    label_fname.push_str("train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    let mut images_as_v: Vec<Vec<f32>>;
    let mut _labels: Vec<u8>;
    {
        let mnist_train_data = MnistData::new(image_fname, label_fname).unwrap();
        let images = mnist_train_data.get_images();
        _labels = mnist_train_data.get_labels().to_vec();
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

    let ef_c = 400;
    let max_nb_connection = 48;
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
    hnsw.dump_layer_info();
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        " ann construction sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    ); //
    log::info!("calling kgraph.init_from_hnsw_layer");
    //
    // extract projection data for persistence
    //
    let knbn = 20;
    let layer = 1;
    //
    let toripserer = ToRipserer::new(&hnsw);
    let outfile = String::from("fashionproj.ripser");
    let res = toripserer.extract_projection_to_ripserer(knbn, layer, &outfile);
    // define a projection on some layer
    if res.is_err() {
        log::info!("graph_projection dump_sparse_mat_for_ripser failed");
    }
    //
    // try to get local persistent data around first images (which is automatically in layer 0)
    //
    log::debug!("extracting matrix of distances around first point");
    let center = data_with_id[0].0;
    let outbson = String::from("fashionlocal.bson");
    let res = toripserer.extract_neighbourhood(&center, 1000, ef_c, &outbson);
    if res.is_err() {
        panic!("ToRipserer.extract_neighbourhood{}", res.err().unwrap());
    }
} // end of main
