#![allow(clippy::print_stderr)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

struct Conv1dCase {
    output: Vec<f32>,
    input: Vec<f32>,
    tap_offsets: Vec<usize>,
    weights: Vec<f32>,
    bias: Vec<f32>,
    out_channels: usize,
    in_channels: usize,
    num_frames: usize,
}

impl Conv1dCase {
    fn new(out_channels: usize, in_channels: usize, kernel_size: usize, num_frames: usize) -> Self {
        let tap_elements = in_channels * num_frames;
        let input = (0..tap_elements * kernel_size)
            .map(|index| ((index + 1) as f32 * 0.017).sin())
            .collect();
        let tap_offsets = (0..kernel_size).map(|tap| tap * tap_elements).collect();
        let weights = (0..out_channels * in_channels * kernel_size)
            .map(|index| ((index + 1) as f32 * 0.031).cos() * 0.25)
            .collect();
        let bias = (0..out_channels)
            .map(|index| index as f32 * 0.01 - 0.03)
            .collect();
        Self {
            output: vec![0.0; out_channels * num_frames],
            input,
            tap_offsets,
            weights,
            bias,
            out_channels,
            in_channels,
            num_frames,
        }
    }

    fn process(&mut self) {
        nam_core::benchmark::conv1d_small_gemv(
            &mut self.output,
            &self.input,
            &self.tap_offsets,
            &self.weights,
            &self.bias,
            self.out_channels,
            self.in_channels,
            self.num_frames,
        );
    }
}

fn benchmark_shapes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("a2_conv1d_shapes");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);
    for (name, out_channels, in_channels, kernel_size) in [
        ("8x4_k4", 8, 4, 4),
        ("4x4_k4", 4, 4, 4),
        ("4x4_k3", 4, 4, 3),
        ("12x3_k2", 12, 3, 2),
    ] {
        for num_frames in [16, 32, 64, 128, 256] {
            let mut case = Conv1dCase::new(out_channels, in_channels, kernel_size, num_frames);
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
