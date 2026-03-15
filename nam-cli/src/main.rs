use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nam", about = "Neural Amp Modeler Rust tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a WAV file through a NAM model
    Render {
        /// Path to .nam model file
        model: PathBuf,
        /// Path to input WAV file
        input: PathBuf,
        /// Path to output WAV file
        output: PathBuf,
    },
    /// Benchmark a model's inference speed
    Bench {
        /// Path to .nam model file
        model: PathBuf,
        /// Buffer size in samples
        #[arg(default_value = "2048")]
        buffer_size: usize,
        /// Number of iterations
        #[arg(default_value = "1000")]
        iterations: usize,
        /// Use fast tanh approximation (faster but less accurate)
        #[arg(long)]
        fast: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Render {
            model,
            input,
            output,
        } => {
            render(&model, &input, &output);
        }
        Commands::Bench {
            model,
            buffer_size,
            iterations,
            fast,
        } => {
            if fast {
                nam_core::enable_fast_tanh();
                eprintln!("Fast tanh: enabled");
            }
            bench(&model, buffer_size, iterations);
        }
    }
}

fn render(model_path: &Path, input_path: &Path, output_path: &Path) {
    // Load model
    let mut model = nam_core::get_dsp(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });

    // Read input WAV
    let mut reader = hound::WavReader::open(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to open input WAV: {}", e);
        std::process::exit(1);
    });

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let num_samples = reader.len() as usize;

    // Read samples as f64
    let input_samples: Vec<nam_core::Sample> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.unwrap() as nam_core::Sample)
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f64 / max_val)
                .map(|s| s as nam_core::Sample)
                .collect()
        }
    };

    // Reset and prewarm
    model.reset(sample_rate, num_samples);
    model.prewarm();

    // Process
    let mut output_samples = vec![0.0 as nam_core::Sample; input_samples.len()];
    let chunk_size = 2048;

    for i in (0..input_samples.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(input_samples.len());
        model.process(&input_samples[i..end], &mut output_samples[i..end]);
    }

    // Write output WAV
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: spec.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(output_path, out_spec).unwrap_or_else(|e| {
        eprintln!("Failed to create output WAV: {}", e);
        std::process::exit(1);
    });

    for &sample in &output_samples {
        writer.write_sample(sample as f32).unwrap();
    }
    writer.finalize().unwrap();

    eprintln!(
        "Rendered {} samples at {} Hz",
        output_samples.len(),
        spec.sample_rate
    );
}

fn bench(model_path: &Path, buffer_size: usize, iterations: usize) {
    let mut model = nam_core::get_dsp(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });

    let sample_rate = model.metadata().expected_sample_rate.unwrap_or(48000.0);
    model.reset(sample_rate, buffer_size);
    model.prewarm();

    let input = vec![0.0 as nam_core::Sample; buffer_size];
    let mut output = vec![0.0 as nam_core::Sample; buffer_size];

    // Warmup
    for _ in 0..10 {
        model.process(&input, &mut output);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        model.process(&input, &mut output);
    }
    let elapsed = start.elapsed();

    let total_samples = buffer_size * iterations;
    let real_time_secs = total_samples as f64 / sample_rate;
    let process_secs = elapsed.as_secs_f64();
    let rtf = process_secs / real_time_secs;

    eprintln!("Model: {:?}", model_path);
    eprintln!("Buffer size: {} samples", buffer_size);
    eprintln!("Iterations: {}", iterations);
    eprintln!("Total samples: {}", total_samples);
    eprintln!("Processing time: {:.3}s", process_secs);
    eprintln!("Real-time audio: {:.3}s", real_time_secs);
    eprintln!("RTF (Real-Time Factor): {:.4}x", rtf);
    if rtf < 1.0 {
        eprintln!("Status: FASTER than real-time ({:.1}x headroom)", 1.0 / rtf);
    } else {
        eprintln!("Status: SLOWER than real-time");
    }
}
