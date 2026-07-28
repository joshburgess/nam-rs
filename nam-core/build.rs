#![allow(clippy::print_stdout)]

fn main() {
    println!("cargo:rerun-if-changed=csrc/fast_kernels.c");
    if std::env::var("CARGO_FEATURE_FAST_KERNELS").is_ok() {
        cc::Build::new()
            .file("csrc/fast_kernels.c")
            .opt_level(3)
            .compile("fast_kernels");
    }
}
