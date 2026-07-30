#![allow(clippy::print_stderr, clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn benchmark_shapes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("a2_grouped_conv1x1_shapes");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);
    for (name, out_channels, in_channels, groups) in [
        ("3x6_g3", 3, 6, 3),
        ("6x6_g3", 6, 6, 3),
        ("4x2_g2", 4, 2, 2),
        ("4x4_g2", 4, 4, 2),
        ("4x8_g4", 4, 8, 4),
        ("8x8_g2", 8, 8, 2),
        ("8x8_g4", 8, 8, 4),
        ("8x8_g8", 8, 8, 8),
        ("16x8_g4", 16, 8, 4),
    ] {
        for num_frames in [16, 32, 64, 128, 256] {
            let mut case = nam_core::benchmark::GroupedConv1x1Benchmark::new(
                out_channels,
                in_channels,
                groups,
                num_frames,
            )
            .unwrap();
            group.throughput(Throughput::Elements(num_frames as u64));
            group.bench_with_input(
                BenchmarkId::new(name, num_frames),
                &num_frames,
                |bencher, _| bencher.iter(|| black_box(&mut case).process()),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_shapes);
criterion_main!(benches);
