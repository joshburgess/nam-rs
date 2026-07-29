use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const IDENTITY_MODEL: &str = r#"{
  "version": "0.7.0",
  "architecture": "Linear",
  "config": {
    "receptive_field": 1,
    "bias": false
  },
  "weights": [1.0]
}"#;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_nam-cli"))
}

fn run(args: &[&str]) -> std::io::Result<Output> {
    command().args(args).output()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn write_model(directory: &Path) -> std::io::Result<PathBuf> {
    let path = directory.join("identity.nam");
    std::fs::write(&path, IDENTITY_MODEL)?;
    Ok(path)
}

fn write_integer_wav(path: &Path, channels: u16, samples: &[i16]) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()
}

#[test]
fn help_succeeds_and_describes_commands() {
    let output = run(&["--help"]).unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(stdout.contains("Neural Amp Modeler Rust tools"));
    assert!(stdout.contains("render"));
    assert!(stdout.contains("bench"));
}

#[test]
fn malformed_model_fails_with_a_diagnostic() {
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("invalid.nam");
    std::fs::write(&model, "{not json").unwrap();

    let output = run(&["bench", model.to_str().unwrap(), "64", "1"]).unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("failed to load model"));
    assert!(stderr(&output).contains("JSON parse error"));
}

#[test]
fn malformed_wav_sample_is_not_replaced_with_silence() {
    let directory = tempfile::tempdir().unwrap();
    let model = write_model(directory.path()).unwrap();
    let input = directory.path().join("truncated.wav");
    let output_path = directory.path().join("output.wav");
    write_integer_wav(&input, 1, &[12_345]).unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&input)
        .unwrap();
    file.set_len(std::fs::metadata(&input).unwrap().len() - 1)
        .unwrap();

    let output = run(&[
        "render",
        model.to_str().unwrap(),
        input.to_str().unwrap(),
        output_path.to_str().unwrap(),
    ])
    .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("failed to read sample 0"));
    assert!(!output_path.exists());
}

#[test]
fn stereo_input_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let model = write_model(directory.path()).unwrap();
    let input = directory.path().join("stereo.wav");
    let output_path = directory.path().join("output.wav");
    write_integer_wav(&input, 2, &[1_000, -1_000]).unwrap();

    let output = run(&[
        "render",
        model.to_str().unwrap(),
        input.to_str().unwrap(),
        output_path.to_str().unwrap(),
    ])
    .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("audio must be mono"));
    assert!(!output_path.exists());
}

#[test]
fn zero_benchmark_arguments_are_rejected_before_loading_the_model() {
    let zero_buffer = run(&["bench", "missing.nam", "0", "1"]).unwrap();
    let zero_iterations = run(&["bench", "missing.nam", "1", "0"]).unwrap();

    assert_eq!(zero_buffer.status.code(), Some(1));
    assert!(stderr(&zero_buffer).contains("buffer size must be greater than zero"));
    assert_eq!(zero_iterations.status.code(), Some(1));
    assert!(stderr(&zero_iterations).contains("iteration count must be greater than zero"));
}

#[test]
fn benchmark_sample_count_overflow_is_reported() {
    let maximum = usize::MAX.to_string();
    let output = run(&["bench", "missing.nam", maximum.as_str(), "2"]).unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("benchmark sample count overflowed"));
}

#[test]
fn render_processes_a_mono_wav_and_reports_success() {
    let directory = tempfile::tempdir().unwrap();
    let model = write_model(directory.path()).unwrap();
    let input = directory.path().join("input.wav");
    let output_path = directory.path().join("output.wav");
    write_integer_wav(&input, 1, &[-16_384, 0, 16_384]).unwrap();

    let output = run(&[
        "render",
        model.to_str().unwrap(),
        input.to_str().unwrap(),
        output_path.to_str().unwrap(),
    ])
    .unwrap();

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stderr(&output).contains("Rendered 3 samples at 48000 Hz"));

    let mut reader = hound::WavReader::open(output_path).unwrap();
    let samples = reader
        .samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(samples.len(), 3);
    assert!((samples[0] + 0.5).abs() < 1.0e-6);
    assert!(samples[1].abs() < 1.0e-6);
    assert!((samples[2] - 0.5).abs() < 1.0e-6);
}
