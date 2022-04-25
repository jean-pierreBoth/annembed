
// for logging (debug mostly, switched at compile time in cargo.toml)




#[macro_use]
extern crate lazy_static;



pub mod tools;
pub mod fromhnsw;
pub mod hdbscan;
pub mod embedder;
pub mod embedparams;
pub mod graphlaplace;
pub mod diffmaps;
pub mod prelude;



lazy_static! {
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
fn init_log() -> u64 {
    let _res = env_logger::try_init();
    println!("\n ************** initializing logger *****************\n");    
    return 1;
}

#[cfg(test)]
mod tests {
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        let _res = env_logger::try_init();
    }
}  // end of tests
