// for logging (debug mostly, switched at compile time in cargo.toml)

use lazy_static::lazy_static;

pub mod diffmaps;
pub mod embedder;
pub mod embedparams;
pub mod fromhnsw;
pub mod graphlaplace;
pub mod hdbscan;
pub mod prelude;
pub mod tools;
pub mod utils;


#[cfg(feature = "python")]
mod python;


lazy_static! {
    static ref LOG: u64 = init_log();
}

// install a logger facility
fn init_log() -> u64 {
    let _res = env_logger::try_init();
    log::info!("\n ************** initializing logger *****************\n");
    1
}

#[cfg(test)]
mod tests {
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        let _res = env_logger::try_init();
    }
} // end of tests
