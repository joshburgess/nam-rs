fn main() {
    if std::env::var("CARGO_FEATURE_FAST_KERNELS").is_ok() {
        cc::Build::new()
            .file("csrc/fast_kernels.c")
            .opt_level(3)
            .flag("-ffast-math")
            .compile("fast_kernels");
    }
}
