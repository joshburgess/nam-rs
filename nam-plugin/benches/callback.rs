#![allow(clippy::unwrap_used)]

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use nam_plugin::benchmark::{CallbackCase, LifecycleCase};

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
    benchmark_model_callback(
        criterion,
        "packed_a2_small_complete_plugin_callback",
        CallbackCase::new_packed_a2_small,
    );
    benchmark_model_callback(
        criterion,
        "packed_a2_full_complete_plugin_callback",
        CallbackCase::new_packed_a2_full,
    );
}

fn benchmark_packed_a2_lifecycle(criterion: &mut Criterion) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../nam-core/test_fixtures/models/upstream_packed_a2_export.nam");
    let mut group = criterion.benchmark_group("packed_a2_lifecycle");
    group.sample_size(20);
    group.bench_function("load_48000_256", |bencher| {
        bencher.iter_batched(
            LifecycleCase::default,
            |mut lifecycle| {
                black_box(lifecycle.restore_model(&path, 48_000.0, 256));
            },
            BatchSize::LargeInput,
        );
    });

    let mut callback = CallbackCase::new_packed_a2_full(128).unwrap();
    let mut select_small = true;
    group.bench_function("switch_model_size", |bencher| {
        bencher.iter(|| {
            callback.set_model_size(if select_small { 0.25 } else { 1.0 });
            select_small = !select_small;
            black_box(&mut callback);
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_callbacks, benchmark_packed_a2_lifecycle);
criterion_main!(benches);
