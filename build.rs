
#[cfg(feature = "macos-accelerate")]
fn main() {
	println!("cargo:rustc-link-lib=framework=Accelerate");
}

#[cfg(not(any(feature = "macos-accelerate")))]
fn main() {}