#![allow(clippy::unwrap_used)]

use iai_callgrind::{
    client_requests::callgrind, library_benchmark, library_benchmark_group, main, EntryPoint,
    LibraryBenchmarkConfig,
};
use std::hint::black_box;
use std::path::Path;

type InferenceCase = (
    Box<dyn nam_core::Dsp>,
    Vec<nam_core::Sample>,
    Vec<nam_core::Sample>,
);

fn setup_model(buffer_size: usize) -> InferenceCase {
    #[cfg(target_arch = "x86_64")]
    assert!(
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
        "native A1 profiling requires AVX2 and FMA"
    );

    let path = Path::new("test_fixtures/models").join("wavenet_a1_standard.nam");
    let mut model = nam_core::get_dsp(&path).unwrap();
    let sample_rate = model.metadata().expected_sample_rate.unwrap_or(48_000.0);
    model.reset(sample_rate, buffer_size);
    model.prewarm();
    (
        model,
        vec![nam_core::Sample::default(); buffer_size],
        vec![nam_core::Sample::default(); buffer_size],
    )
}

#[library_benchmark]
#[bench::a1_native_32(setup_model(32))]
#[bench::a1_native_64(setup_model(64))]
#[bench::a1_native_128(setup_model(128))]
#[bench::a1_native_256(setup_model(256))]
fn inference((mut model, input, mut output): InferenceCase) {
    callgrind::toggle_collect();
    model.process(black_box(&input), black_box(&mut output));
    callgrind::toggle_collect();
    black_box(output);
}

library_benchmark_group!(name = native_a1_group; benchmarks = inference);
main!(
    config = LibraryBenchmarkConfig::default()
        .callgrind_args(["collect-atstart=no"])
        .entry_point(EntryPoint::None);
    library_benchmark_groups = native_a1_group
);
