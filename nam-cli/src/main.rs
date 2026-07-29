#![allow(clippy::print_stderr)]

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use thiserror::Error;

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

#[derive(Debug, Error)]
enum CliError {
    #[error("failed to load model {path}: {source}")]
    LoadModel {
        path: String,
        #[source]
        source: nam_core::NamError,
    },

    #[error("failed to open input WAV {path}: {source}")]
    OpenInput {
        path: String,
        #[source]
        source: hound::Error,
    },

    #[error("input WAV {path} is not supported: {reason}")]
    UnsupportedInput { path: String, reason: &'static str },

    #[error("failed to read sample {index} from input WAV: {source}")]
    ReadSample {
        index: usize,
        #[source]
        source: hound::Error,
    },

    #[error("failed to create output WAV {path}: {source}")]
    CreateOutput {
        path: String,
        #[source]
        source: hound::Error,
    },

    #[error("failed to write sample {index} to output WAV: {source}")]
    WriteSample {
        index: usize,
        #[source]
        source: hound::Error,
    },

    #[error("failed to finalize output WAV: {0}")]
    FinalizeOutput(#[source] hound::Error),

    #[error("{name} must be greater than zero")]
    ZeroArgument { name: &'static str },

    #[error("benchmark sample count overflowed")]
    BenchmarkSampleCountOverflow,

    #[error("model sample rate must be finite and greater than zero, got {0}")]
    InvalidModelSampleRate(f64),
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Commands::Render {
            model,
            input,
            output,
        } => render(&model, &input, &output),
        Commands::Bench {
            model,
            buffer_size,
            iterations,
            fast,
        } => bench(&model, buffer_size, iterations, fast),
    }
}

fn load_model(path: &Path) -> Result<Box<dyn nam_core::Dsp>, CliError> {
    nam_core::get_dsp(path).map_err(|source| CliError::LoadModel {
        path: path.display().to_string(),
        source,
    })
}

fn validate_input_spec(path: &Path, spec: hound::WavSpec) -> Result<(), CliError> {
    let reason = if spec.channels != 1 {
        Some("audio must be mono")
    } else if spec.sample_rate == 0 {
        Some("sample rate must be greater than zero")
    } else {
        match spec.sample_format {
            hound::SampleFormat::Float if spec.bits_per_sample != 32 => {
                Some("floating-point audio must use 32-bit samples")
            }
            hound::SampleFormat::Int if !(1..=32).contains(&spec.bits_per_sample) => {
                Some("integer audio must use between 1 and 32 bits per sample")
            }
            _ => None,
        }
    };

    match reason {
        Some(reason) => Err(CliError::UnsupportedInput {
            path: path.display().to_string(),
            reason,
        }),
        None => Ok(()),
    }
}

fn read_input_samples(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    spec: hound::WavSpec,
) -> Result<Vec<nam_core::Sample>, CliError> {
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .enumerate()
            .map(|(index, sample)| {
                sample
                    .map(|value| value as nam_core::Sample)
                    .map_err(|source| CliError::ReadSample { index, source })
            })
            .collect(),
        hound::SampleFormat::Int => {
            let max_value = (1_i64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .enumerate()
                .map(|(index, sample)| {
                    sample
                        .map(|value| (f64::from(value) / max_value) as nam_core::Sample)
                        .map_err(|source| CliError::ReadSample { index, source })
                })
                .collect()
        }
    }
}

fn render(model_path: &Path, input_path: &Path, output_path: &Path) -> Result<(), CliError> {
    let mut model = load_model(model_path)?;
    let mut reader = hound::WavReader::open(input_path).map_err(|source| CliError::OpenInput {
        path: input_path.display().to_string(),
        source,
    })?;

    let spec = reader.spec();
    validate_input_spec(input_path, spec)?;
    let sample_rate = f64::from(spec.sample_rate);
    let input_samples = read_input_samples(&mut reader, spec)?;

    model.reset(sample_rate, input_samples.len());
    model.prewarm();

    let mut output_samples = vec![nam_core::Sample::default(); input_samples.len()];
    const CHUNK_SIZE: usize = 2048;
    for start in (0..input_samples.len()).step_by(CHUNK_SIZE) {
        let end = start.saturating_add(CHUNK_SIZE).min(input_samples.len());
        model.process(&input_samples[start..end], &mut output_samples[start..end]);
    }

    let output_spec = hound::WavSpec {
        channels: 1,
        sample_rate: spec.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(output_path, output_spec).map_err(|source| {
        CliError::CreateOutput {
            path: output_path.display().to_string(),
            source,
        }
    })?;

    for (index, &sample) in output_samples.iter().enumerate() {
        writer
            .write_sample(nam_core::dsp::sample_to_f32(sample))
            .map_err(|source| CliError::WriteSample { index, source })?;
    }
    writer.finalize().map_err(CliError::FinalizeOutput)?;

    eprintln!(
        "Rendered {} samples at {} Hz",
        output_samples.len(),
        spec.sample_rate
    );
    Ok(())
}

fn positive_argument(value: usize, name: &'static str) -> Result<usize, CliError> {
    if value == 0 {
        Err(CliError::ZeroArgument { name })
    } else {
        Ok(value)
    }
}

fn bench(
    model_path: &Path,
    buffer_size: usize,
    iterations: usize,
    fast: bool,
) -> Result<(), CliError> {
    let buffer_size = positive_argument(buffer_size, "buffer size")?;
    let iterations = positive_argument(iterations, "iteration count")?;
    let total_samples = buffer_size
        .checked_mul(iterations)
        .ok_or(CliError::BenchmarkSampleCountOverflow)?;

    let mut model = load_model(model_path)?;
    model.set_activation_mode(if fast {
        nam_core::ActivationMode::Fast
    } else {
        nam_core::ActivationMode::Accurate
    });

    let sample_rate = model.metadata().expected_sample_rate.unwrap_or(48_000.0);
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(CliError::InvalidModelSampleRate(sample_rate));
    }

    model.reset(sample_rate, buffer_size);
    model.prewarm();

    let input = vec![nam_core::Sample::default(); buffer_size];
    let mut output = vec![nam_core::Sample::default(); buffer_size];

    for _ in 0..10 {
        model.process(&input, &mut output);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        model.process(&input, &mut output);
    }
    let elapsed = start.elapsed();

    let real_time_seconds = total_samples as f64 / sample_rate;
    let process_seconds = elapsed.as_secs_f64();
    let real_time_factor = process_seconds / real_time_seconds;

    eprintln!("Model: {:?}", model_path);
    eprintln!("Buffer size: {buffer_size} samples");
    eprintln!("Iterations: {iterations}");
    eprintln!("Total samples: {total_samples}");
    eprintln!("Processing time: {process_seconds:.3}s");
    eprintln!("Real-time audio: {real_time_seconds:.3}s");
    eprintln!("RTF (Real-Time Factor): {real_time_factor:.4}x");
    if real_time_factor < 1.0 {
        eprintln!(
            "Status: FASTER than real-time ({:.1}x headroom)",
            1.0 / real_time_factor
        );
    } else {
        eprintln!("Status: SLOWER than real-time");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_spec_requires_mono_audio() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        assert!(matches!(
            validate_input_spec(Path::new("stereo.wav"), spec),
            Err(CliError::UnsupportedInput {
                reason: "audio must be mono",
                ..
            })
        ));
    }

    #[test]
    fn input_spec_requires_nonzero_sample_rate() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 0,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        assert!(matches!(
            validate_input_spec(Path::new("zero-rate.wav"), spec),
            Err(CliError::UnsupportedInput {
                reason: "sample rate must be greater than zero",
                ..
            })
        ));
    }

    #[test]
    fn input_spec_rejects_invalid_sample_widths() {
        let invalid_float = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 64,
            sample_format: hound::SampleFormat::Float,
        };
        let invalid_integer = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 0,
            sample_format: hound::SampleFormat::Int,
        };

        assert!(validate_input_spec(Path::new("float.wav"), invalid_float).is_err());
        assert!(validate_input_spec(Path::new("integer.wav"), invalid_integer).is_err());
    }

    #[test]
    fn benchmark_arguments_must_be_positive() {
        assert!(matches!(
            positive_argument(0, "buffer size"),
            Err(CliError::ZeroArgument {
                name: "buffer size"
            })
        ));
        assert_eq!(positive_argument(1, "buffer size").unwrap(), 1);
    }
}
