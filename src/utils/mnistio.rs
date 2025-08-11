//! utility for Mnist
//!
//!
//!

use anyhow::anyhow;
use ndarray::{s, Array1, Array3};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;

/// A struct to load/store [MNIST data](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
/// stores labels (i.e : digits between 0 and 9) coming from file train-labels-idx1-ubyte      
/// and hand written characters as 28*28 images with values between 0 and 255 coming from train-images-idx3-ubyte
pub struct MnistData {
    _image_filename: String,
    _label_filename: String,
    pub(crate) images: Array3<u8>,
    pub(crate) labels: Array1<u8>,
}

impl MnistData {
    pub fn new(image_filename: String, label_filename: String) -> std::io::Result<MnistData> {
        let image_path = PathBuf::from(image_filename.clone());
        let image_file = OpenOptions::new().read(true).open(image_path)?;
        let mut image_io = BufReader::new(image_file);
        let images = read_image_file(&mut image_io);
        // labels
        let label_path = PathBuf::from(label_filename.clone());
        let labels_file = OpenOptions::new().read(true).open(label_path)?;
        let mut labels_io = BufReader::new(labels_file);
        let labels = read_label_file(&mut labels_io);
        Ok(MnistData {
            _image_filename: image_filename,
            _label_filename: label_filename,
            images,
            labels,
        })
    } // end of new for MnistData

    /// returns labels of images. lables\[k\] is the label of the k th image.
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
    let toread: u32 = 0;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let magic = u32::from_be(toread);
    assert_eq!(magic, 2051);
    // read nbitems
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbitem = u32::from_be(toread);
    assert!(nbitem == 60000 || nbitem == 10000);
    //  read nbrow
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbrow = u32::from_be(toread);
    assert_eq!(nbrow, 28);
    // read nbcolumns
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbcolumn = u32::from_be(toread);
    assert_eq!(nbcolumn, 28);
    // for each item, read a row of nbcolumns u8
    let mut images = Array3::<u8>::zeros((nbrow as usize, nbcolumn as usize, nbitem as usize));
    let mut datarow = vec![0u8; nbcolumn as usize];
    for k in 0..nbitem as usize {
        for i in 0..nbrow as usize {
            let it_slice = datarow.as_mut_slice();
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
    let toread: u32 = 0;
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let magic = u32::from_be(toread);
    assert_eq!(magic, 2049);
    // read nbitems
    let it_slice = unsafe {
        ::std::slice::from_raw_parts_mut(
            (&toread as *const u32) as *mut u8,
            ::std::mem::size_of::<u32>(),
        )
    };
    io_in.read_exact(it_slice).unwrap();
    let nbitem = u32::from_be(toread);
    assert!(nbitem == 60000 || nbitem == 10000);
    let mut labels_vec = vec![0u8; nbitem as usize];
    io_in.read_exact(&mut labels_vec).unwrap();
    Array1::from(labels_vec)
} // end of fn read_label

/// load mnist data from Directory
pub fn load_mnist_train_data(dname: &str) -> anyhow::Result<MnistData> {
    let mut image_fname = String::from(dname);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        log::error!("could not open image file : {:?}", image_fname);
        return Err(anyhow!("io error"));
    }

    let mut label_fname = String::from(dname);
    label_fname.push_str("train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        log::error!("could not open label file : {:?}", label_fname);
        return Err(anyhow!("io error"));
    }
    //
    Ok(MnistData::new(image_fname, label_fname).unwrap())
}

pub fn load_mnist_test_data(dname: &str) -> anyhow::Result<MnistData> {
    let mut image_fname = String::from(dname);
    image_fname.push_str("t10k-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        log::error!("could not open image file : {:?}", image_fname);
        return Err(anyhow!("io error"));
    }

    let mut label_fname = String::from(dname);
    label_fname.push_str("t10k-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        log::error!("could not open label file : {:?}", label_fname);
        return Err(anyhow!("io error"));
    }
    //
    Ok(MnistData::new(image_fname, label_fname).unwrap())
}
