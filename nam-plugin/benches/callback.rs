#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nam_plugin::benchmark::CallbackCase;

fn benchmark_model_callback(
    criterion: &mut Criterion,
    group_name: &str,
    make_case: fn(usize) -> Result<CallbackCase, nam_core::NamError>,
) {
    let mut group = criterion.benchmark_group(group_name);
    for buffer_size in [16, 32, 64, 128, 256] {
        let mut case = make_case(buffer_size).unwrap();
        group.throughput(Throughput::Elements(buffer_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |bencher, _| bencher.iter(|| black_box(&mut case).process()),
        );
    }
    group.finish();
}

fn benchmark_callbacks(criterion: &mut Criterion) {
    benchmark_model_callback(
        criterion,
        "a1_accurate_complete_plugin_callback",
        CallbackCase::new_a1,
    );
    benchmark_model_callback(
        criterion,
        "a2_complete_plugin_callback",
        CallbackCase::new_a2,
    );
}

criterion_group!(benches, benchmark_callbacks);
criterion_main!(benches);
