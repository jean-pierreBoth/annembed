extern crate rand;

// for logging (debug mostly, switched at compile time in cargo.toml)


extern crate log;

extern crate env_logger;

#[macro_use]
extern crate lazy_static;

pub mod entropy;
pub mod truncsvd;


lazy_static! {
    #[allow(dead_code)]
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
fn init_log() -> u64 {
    env_logger::try_init().unwrap();
    println!("\n ************** initializing logger *****************\n");    
    return 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        env_logger::try_init().unwrap();
    }
}  // end of tests
