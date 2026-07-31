#![allow(clippy::print_stdout)]

fn main() {
    println!("cargo:rerun-if-changed=csrc/fast_kernels.c");
    println!("cargo:rerun-if-env-changed=NAM_DISABLE_PORTABLE_VECTOR_TANH");
    if std::env::var("CARGO_FEATURE_FAST_KERNELS").is_ok() {
        if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
            && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu")
            && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
        {
            println!("cargo:rustc-link-lib=dl");
        }
        let mut build = cc::Build::new();
        build.file("csrc/fast_kernels.c").opt_level(3);
        if std::env::var("NAM_DISABLE_PORTABLE_VECTOR_TANH").is_ok() {
            build.define("NAM_DISABLE_PORTABLE_VECTOR_TANH", None);
        }
        build.compile("fast_kernels");
    }
}
