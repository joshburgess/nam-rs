#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nam_plugin::benchmark::CallbackCase;

fn benchmark_a2_callback(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("a2_complete_plugin_callback");
    for buffer_size in [16, 32, 64, 128, 256] {
        let mut case = CallbackCase::new_a2(buffer_size).unwrap();
        group.throughput(Throughput::Elements(buffer_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |bencher, _| bencher.iter(|| black_box(&mut case).process()),
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_a2_callback);
criterion_main!(benches);
