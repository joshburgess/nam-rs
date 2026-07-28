#![allow(clippy::unwrap_used)]

use iai_callgrind::{
    black_box, client_requests::callgrind, library_benchmark, library_benchmark_group, main,
    EntryPoint, LibraryBenchmarkConfig,
};
use nam_plugin::benchmark::CallbackCase;

fn setup(host_rate: usize, model_rate: usize, buffer_size: usize) -> CallbackCase {
    CallbackCase::new(host_rate, model_rate, buffer_size).unwrap()
}

#[library_benchmark]
#[bench::native_64(setup(48_000, 48_000, 64))]
#[bench::resample_44100_to_48000_64(setup(44_100, 48_000, 64))]
#[bench::resample_48000_to_44100_256(setup(48_000, 44_100, 256))]
fn callback(mut case: CallbackCase) {
    callgrind::toggle_collect();
    case.process();
    callgrind::toggle_collect();
    black_box(case);
}

library_benchmark_group!(name = callback_group; benchmarks = callback);
main!(
    config = LibraryBenchmarkConfig::default()
        .callgrind_args(["collect-atstart=no"])
        .entry_point(EntryPoint::None);
    library_benchmark_groups = callback_group
);
