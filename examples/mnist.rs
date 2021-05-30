//! Structure and functions to read MNIST database

use std::io::prelude::*;
use std::io::{BufReader};
use ndarray::{Array3, Array1, s};
use std::fs::{OpenOptions};
use std::path::{PathBuf};



use hnsw_rs::prelude::*;

use annembed::prelude::*;

/// A struct to load/store [MNIST data](http://yann.lecun.com/exdb/mnist/)  
/// stores labels (i.e : digits between 0 and 9) coming from file train-labels-idx1-ubyte      
/// and hand written characters as 28*28 images with values between 0 and 255 coming from train-images-idx3-ubyte
pub struct MnistData {
    _image_filename : String,
    _label_filename : String,
    images : Array3::<u8>,
    labels : Array1::<u8>,
}


impl MnistData {
    pub fn new(image_filename : String, label_filename : String) -> std::io::Result<MnistData> {
        let image_path = PathBuf::from(image_filename.clone());
        let image_file = OpenOptions::new().read(true).open(&image_path)?;
        let mut image_io = BufReader::new(image_file);
        let images = read_image_file(&mut image_io);
        // labels
        let label_path = PathBuf::from(label_filename.clone());
        let labels_file = OpenOptions::new().read(true).open(&label_path)?;
        let mut labels_io = BufReader::new(labels_file);
        let labels = read_label_file(&mut labels_io);
        Ok(MnistData{
            _image_filename : image_filename,
            _label_filename : label_filename,
            images,
            labels
        } )
    } // end of new for MnistData

    /// returns labels of images. lables[k] is the label of the k th image.
    pub fn get_labels(&self) -> &Array1::<u8> {
        &self.labels
    }

    /// returns images. images are stored in Array3 with Array3[[.., .., k]] being the k images!
    /// Each image is stored as it is in the Mnist files, Array3[[i, .., k]] is the i row of the k image
    pub fn get_images(&self) -> &Array3::<u8> {
        &self.images
    }
} // end of impl MnistData


pub fn read_image_file(io_in: &mut dyn Read) -> Array3::<u8> {
    // read 4 bytes magic
    let magic : u32;
    // to read 32 bits in network order!
    let toread : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    magic = u32::from_be(toread);
    assert_eq!(magic, 2051);
    // read nbitems
    let nbitem : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbitem = u32::from_be(toread);
    assert_eq!(nbitem, 60000);
    //  read nbrow
    let nbrow : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbrow = u32::from_be(toread); 
    assert_eq!(nbrow, 28);   
    // read nbcolumns
    let nbcolumn : u32;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    nbcolumn = u32::from_be(toread);     
    assert_eq!(nbcolumn,28);   
    // for each item, read a row of nbcolumns u8
    let mut images = Array3::<u8>::zeros((nbrow as usize , nbcolumn as usize, nbitem as usize));
    let mut datarow = Vec::<u8>::new();
    datarow.resize(nbcolumn as usize, 0);
    for k in 0..nbitem as usize {
        for i in 0..nbrow as usize {
            let it_slice ;
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



pub fn read_label_file(io_in: &mut dyn Read) -> Array1<u8>{
    let magic : u32;
    // to read 32 bits in network order!
    let toread : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice).unwrap();
    magic = u32::from_be(toread);
    assert_eq!(magic, 2049);
     // read nbitems
     let nbitem : u32;
     let it_slice = unsafe {::std::slice::from_raw_parts_mut((&toread as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
     io_in.read_exact(it_slice).unwrap();
     nbitem = u32::from_be(toread);   
     assert_eq!(nbitem, 60000);
     let mut labels_vec = Vec::<u8>::new();
     labels_vec.resize(nbitem as usize, 0);
     io_in.read_exact(&mut labels_vec).unwrap();
     let labels = Array1::from(labels_vec);
     labels
    }  // end of fn read_label

//============================================================================================

use csv::*;


pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let image_fname = String::from("/home.1/jpboth/Data/MNIST/train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }

    let label_fname = String::from("/home.1/jpboth/Data/MNIST/train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    let mut images_as_v:  Vec::<Vec<f32>>;
    let labels :  Array1::<u8>;
    {
        let mnist_data  = MnistData::new(image_fname, label_fname).unwrap();
        let images = mnist_data.get_images();
        labels = mnist_data.get_labels().clone();
        let( _, _, nbimages) = images.dim();
        //
        images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        for k in 0..nbimages {
            let v : Vec<f32> = images.slice(s![.., .., k]).iter().map(|v| *v as f32).collect();
            images_as_v.push(v);
        }
    }  // drop mnist_data
    //
    let ef_c = 50;
    let max_nb_connection = 50;
    let nbimages = images_as_v.len();
    let nb_layer = 16.min((nbimages as f32).ln().trunc() as usize);
    let hnsw = Hnsw::<f32, DistL1>::new(max_nb_connection, nbimages, nb_layer, ef_c, DistL1{});
    // we must pay fortran indexation once!. transform image to a vector
    let data_with_id = images_as_v.iter().zip(0..images_as_v.len()).collect();
    hnsw.parallel_insert(&data_with_id);
    // images as vectors of f32 and send to hnsw
    hnsw.dump_layer_info();
    //
    let mut kgraph = KGraph::<f32>::new();
    log::info!("calling kgraph.init_from_hnsw_all");
    let knbn = 20;
    let res = kgraph.init_from_hnsw_all(&hnsw, knbn);
    if res.is_err() {
        panic!("init_from_hnsw_all  failed");
    }
    //
    let mut kgraph = KGraph::<f32>::new();
    log::info!("calling kgraph.init_from_hnsw_all");
    let res = kgraph.init_from_hnsw_all(&hnsw, knbn);
    if res.is_err() {
        panic!("init_from_hnsw_all  failed");
    }
    log::info!("minimum number of neighbours {}", kgraph.get_max_nbng());
    // 
    let _kgraph_stats = kgraph.get_kraph_stats();
    let embed_dim = 5;
    let mut embedder = Embedder::new(&kgraph, embed_dim);
    let embed_res = embedder.embed();
    assert!(embed_res.is_ok());  
    // dump
    let mut csv_w = Writer::from_path("mnist_csv").unwrap();
    let _res = write_csv_labeled_array2(&mut csv_w, labels.as_slice().unwrap(), embedder.get_emmbedded().unwrap());
    csv_w.flush().unwrap();
}


//============================================================================================



#[cfg(test)]

mod tests {


use super::*;

// test and compare some values obtained with Julia loading

#[test]

fn test_load_mnist() {
    let image_fname = String::from("/home.1/jpboth/Data/MNIST/train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(&image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }

    let label_fname = String::from("/home.1/jpboth/Data/MNIST/train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }

    let mnist_data  = MnistData::new(image_fname, label_fname).unwrap();
    assert_eq!(0x3c, *mnist_data.images.get([9,14,9]).unwrap());
    assert_eq!(0xfd, mnist_data.images[(14 , 9, 9)]);
    // check some value of the tenth images

    // check first and last labels
    assert_eq!(5, mnist_data.labels[0]);
    assert_eq!(8, mnist_data.labels[mnist_data.labels.len()-1]);
    assert_eq!(1,1);
} // end test_load


}  // end module tests