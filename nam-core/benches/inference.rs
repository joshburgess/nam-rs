use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::Path;

fn bench_model(c: &mut Criterion, name: &str, filename: &str) {
    let path = Path::new("test_fixtures/models").join(filename);
    if !path.exists() {
        eprintln!("Skipping benchmark: {:?} not found", path);
        return;
    }

    let mut model = nam_core::get_dsp(&path).unwrap();
    let sample_rate = model.metadata().expected_sample_rate.unwrap_or(48000.0);

    let buffer_size = 64;
    model.reset(sample_rate, buffer_size);
    model.prewarm();

    let input = vec![0.0 as nam_core::Sample; buffer_size];
    let mut output = vec![0.0 as nam_core::Sample; buffer_size];

    c.bench_function(name, |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        })
    });
}

fn bench_model_buffer_sizes(c: &mut Criterion, name: &str, filename: &str) {
    let path = Path::new("test_fixtures/models").join(filename);
    if !path.exists() {
        return;
    }

    let mut group = c.benchmark_group(name);
    for &buffer_size in &[16, 32, 64, 128, 256, 512, 1024, 2048] {
        let mut model = nam_core::get_dsp(&path).unwrap();
        let sample_rate = model.metadata().expected_sample_rate.unwrap_or(48000.0);
        model.reset(sample_rate, buffer_size);
        model.prewarm();

        let input = vec![0.0 as nam_core::Sample; buffer_size];
        let mut output = vec![0.0 as nam_core::Sample; buffer_size];

        group.throughput(criterion::Throughput::Elements(buffer_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |b, _| {
                b.iter(|| {
                    model.process(&input, &mut output);
                })
            },
        );
    }
    group.finish();
}

fn bench_wavenet(c: &mut Criterion) {
    bench_model(c, "wavenet_small_64", "wavenet.nam");
}

fn bench_wavenet_standard(c: &mut Criterion) {
    bench_model(c, "wavenet_standard_64", "wavenet_a1_standard.nam");
}

fn bench_lstm(c: &mut Criterion) {
    bench_model(c, "lstm_64", "lstm.nam");
}

fn bench_a2_max(c: &mut Criterion) {
    bench_model(c, "wavenet_a2_max_64", "wavenet_a2_max.nam");
}

fn bench_wavenet_standard_bufsizes(c: &mut Criterion) {
    bench_model_buffer_sizes(c, "wavenet_standard_bufsizes", "wavenet_a1_standard.nam");
}

criterion_group!(
    benches,
    bench_wavenet,
    bench_wavenet_standard,
    bench_lstm,
    bench_a2_max,
    bench_wavenet_standard_bufsizes,
);
criterion_main!(benches);
