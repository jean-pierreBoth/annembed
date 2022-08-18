#[cfg(feature = "openblas-system")]
fn main() {
}

#[cfg(not(any(feature = "openblas-system", feature = "netlib-system")))]
fn main() {}