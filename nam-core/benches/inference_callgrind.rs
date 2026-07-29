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

fn setup_model(filename: &str, buffer_size: usize) -> InferenceCase {
    let path = Path::new("test_fixtures/models").join(filename);
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
#[bench::a1_standard_16(setup_model("wavenet_a1_standard.nam", 16))]
#[bench::a1_standard(setup_model("wavenet_a1_standard.nam", 64))]
#[bench::a1_standard_256(setup_model("wavenet_a1_standard.nam", 256))]
#[bench::a2_max_16(setup_model("wavenet_a2_max.nam", 16))]
#[bench::a2_max(setup_model("wavenet_a2_max.nam", 64))]
#[bench::a2_max_256(setup_model("wavenet_a2_max.nam", 256))]
#[bench::lstm_64(setup_model("lstm.nam", 64))]
fn inference((mut model, input, mut output): InferenceCase) {
    callgrind::toggle_collect();
    model.process(black_box(&input), black_box(&mut output));
    callgrind::toggle_collect();
    black_box(output);
}

library_benchmark_group!(name = inference_group; benchmarks = inference);
main!(
    config = LibraryBenchmarkConfig::default()
        .callgrind_args(["collect-atstart=no"])
        .env(
            "GLIBC_TUNABLES",
            "glibc.cpu.hwcaps=-SSSE3,-SSE4_1,-SSE4_2,-AVX,-AVX2,-FMA,-AVX512,-AVX512F,-AVX512CD,-AVX512BW,-AVX512DQ,-AVX512VL",
        )
        .env("LD_HWCAP_MASK", "0")
        .entry_point(EntryPoint::None);
    library_benchmark_groups = inference_group
);
