extern crate rand;

// for logging (debug mostly, switched at compile time in cargo.toml)


extern crate log;

extern crate env_logger;

#[macro_use]
extern crate lazy_static;



pub mod tools;

pub mod fromhnsw;

pub mod hdbscan;

pub mod embedder;


lazy_static! {
    #[allow(dead_code)]
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
    use super::*;
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        let _res = env_logger::try_init().unwrap();
    }
}  // end of tests
