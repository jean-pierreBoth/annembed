#[cfg(feature = "openblas-system")]
fn main() {
    println!("cargo:rustc-link-lib=lapacke");
    println!("cargo:rustc-link-lib=openblas64");
}

#[cfg(not(any(feature = "openblas-system", feature = "netlib-system")))]
fn main() {}