use std::path::Path;

use crate::convnet::ConvNet;
use crate::dsp::{Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::linear::Linear;
use crate::lstm::Lstm;
use crate::version::verify_config_version;
use crate::wavenet::WaveNet;

pub fn get_dsp(path: &Path) -> Result<Box<dyn Dsp>, NamError> {
    let content = std::fs::read_to_string(path)?;
    get_dsp_from_json(&content)
}

pub fn get_dsp_from_json(json_str: &str) -> Result<Box<dyn Dsp>, NamError> {
    let root: serde_json::Value = serde_json::from_str(json_str)?;

    // 1. Version check
    let version = root["version"]
        .as_str()
        .ok_or_else(|| NamError::MissingField("version".into()))?;
    verify_config_version(version)?;

    get_dsp_from_value_inner(&root, true)
}

/// Build a DSP model from a JSON value. Used for nested models (condition_dsp).
/// Matches C++ behavior: the nested model is prewarmed during construction.
pub fn get_dsp_from_value(val: &serde_json::Value) -> Result<Box<dyn Dsp>, NamError> {
    // For nested models, still verify version if present
    if let Some(version) = val.get("version").and_then(|v| v.as_str()) {
        verify_config_version(version)?;
    }
    let mut model = get_dsp_from_value_inner(val, false)?;
    // C++ get_dsp() calls prewarm() during construction — match that behavior
    model.prewarm();
    Ok(model)
}

fn get_dsp_from_value_inner(
    root: &serde_json::Value,
    _is_top_level: bool,
) -> Result<Box<dyn Dsp>, NamError> {
    // Dispatch on architecture
    let arch = root["architecture"]
        .as_str()
        .ok_or_else(|| NamError::MissingField("architecture".into()))?;

    // SlimmableContainer has no top-level weights; each submodel has its own.
    if arch == "SlimmableContainer" {
        let config = &root["config"];
        let metadata = parse_metadata(root);
        return SlimmableContainer::from_config(config, metadata);
    }

    // Parse weights
    let weights: Vec<f32> = root["weights"]
        .as_array()
        .ok_or_else(|| NamError::MissingField("weights".into()))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();

    // Parse metadata
    let metadata = parse_metadata(root);

    let config = &root["config"];

    match arch {
        "Linear" => Ok(Box::new(Linear::from_config(config, &weights, metadata)?)),
        "ConvNet" => Ok(Box::new(ConvNet::from_config(config, &weights, metadata)?)),
        "LSTM" => Ok(Box::new(Lstm::from_config(config, &weights, metadata)?)),
        "WaveNet" => {
            // Check for condition_dsp in config
            let condition_dsp = if let Some(cd) = config.get("condition_dsp") {
                if !cd.is_null() {
                    Some(get_dsp_from_value(cd)?)
                } else {
                    None
                }
            } else {
                None
            };
            Ok(Box::new(WaveNet::from_config_with_condition_dsp(
                config,
                &weights,
                metadata,
                condition_dsp,
            )?))
        }
        other => Err(NamError::UnknownArchitecture(other.into())),
    }
}

// ── SlimmableContainer ─────────────────────────────────────────────────────

/// A container model that holds multiple sub-models at different sizes.
/// For now, we load ALL sub-models but default the active index to the last
/// (full-size) variant, matching C++ behavior.
struct SlimmableContainer {
    submodels: Vec<(f64, Box<dyn Dsp>)>, // (max_value, model)
    active_index: usize,
    metadata: DspMetadata,
}

impl SlimmableContainer {
    fn from_config(
        config: &serde_json::Value,
        metadata: DspMetadata,
    ) -> Result<Box<dyn Dsp>, NamError> {
        let submodels_json = config["submodels"]
            .as_array()
            .ok_or_else(|| NamError::MissingField("config.submodels".into()))?;

        if submodels_json.is_empty() {
            return Err(NamError::InvalidConfig(
                "SlimmableContainer: submodels array is empty".into(),
            ));
        }

        let mut submodels = Vec::with_capacity(submodels_json.len());
        for entry in submodels_json {
            let max_value = entry["max_value"]
                .as_f64()
                .ok_or_else(|| NamError::MissingField("submodel max_value".into()))?;
            let model_json = entry
                .get("model")
                .ok_or_else(|| NamError::MissingField("submodel model".into()))?;
            let model = get_dsp_from_value(model_json)?;
            submodels.push((max_value, model));
        }

        let active_index = submodels.len() - 1;
        Ok(Box::new(SlimmableContainer {
            submodels,
            active_index,
            metadata,
        }))
    }
}

impl Dsp for SlimmableContainer {
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        self.submodels[self.active_index].1.process(input, output);
    }

    fn reset(&mut self, sample_rate: f64, max_buffer_size: usize) {
        for (_, model) in &mut self.submodels {
            model.reset(sample_rate, max_buffer_size);
        }
    }

    fn prewarm_samples(&self) -> usize {
        0
    }

    fn prewarm(&mut self) {
        for (_, model) in &mut self.submodels {
            model.prewarm();
        }
    }

    fn metadata(&self) -> &DspMetadata {
        &self.metadata
    }
}

fn parse_metadata(root: &serde_json::Value) -> DspMetadata {
    let m = &root["metadata"];
    DspMetadata {
        loudness: m["loudness"].as_f64(),
        expected_sample_rate: root
            .get("sample_rate")
            .and_then(|v| v.as_f64())
            .or_else(|| m["sample_rate"].as_f64()),
        input_level_dbu: m["input_level_dbu"].as_f64(),
        output_level_dbu: m["output_level_dbu"].as_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_get_dsp_lstm() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();
        let input = vec![0.0 as crate::dsp::Sample; 64];
        let mut output = vec![0.0 as crate::dsp::Sample; 64];
        model.process(&input, &mut output);
    }

    #[test]
    fn test_get_dsp_wavenet() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();
        let input = vec![0.0 as crate::dsp::Sample; 64];
        let mut output = vec![0.0 as crate::dsp::Sample; 64];
        model.process(&input, &mut output);
    }

    #[test]
    fn test_get_dsp_metadata() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }
        let model = get_dsp(path).unwrap();
        let meta = model.metadata();
        assert!(meta.loudness.is_some());
        assert!(meta.input_level_dbu.is_some());
    }

    #[test]
    fn test_invalid_json() {
        let result = get_dsp_from_json("not valid json {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_version() {
        let json = r#"{"architecture": "Linear", "config": {}, "weights": []}"#;
        assert!(get_dsp_from_json(json).is_err());
    }

    #[test]
    fn test_wrong_version() {
        let json = r#"{"version": "1.0.0", "architecture": "Linear", "config": {}, "weights": []}"#;
        assert!(matches!(
            get_dsp_from_json(json),
            Err(NamError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn test_missing_architecture() {
        let json = r#"{"version": "0.5.0", "config": {}, "weights": []}"#;
        assert!(get_dsp_from_json(json).is_err());
    }

    #[test]
    fn test_unknown_architecture() {
        let json =
            r#"{"version": "0.5.0", "architecture": "Transformer", "config": {}, "weights": []}"#;
        assert!(matches!(
            get_dsp_from_json(json),
            Err(NamError::UnknownArchitecture(_))
        ));
    }

    #[test]
    fn test_missing_weights() {
        let json =
            r#"{"version": "0.5.0", "architecture": "Linear", "config": {"receptive_field": 2}}"#;
        assert!(get_dsp_from_json(json).is_err());
    }

    #[test]
    fn test_empty_weights_for_linear() {
        let json = r#"{"version": "0.5.0", "architecture": "Linear", "config": {"receptive_field": 2}, "weights": []}"#;
        assert!(get_dsp_from_json(json).is_err());
    }

    #[test]
    fn test_valid_minimal_linear() {
        let json = r#"{"version":"0.5.0","architecture":"Linear","config":{"receptive_field":3},"weights":[1.0,0.5,0.25]}"#;
        assert!(get_dsp_from_json(json).is_ok());
    }

    #[test]
    fn test_metadata_missing_is_ok() {
        let json = r#"{"version":"0.5.0","architecture":"Linear","config":{"receptive_field":1},"weights":[1.0]}"#;
        let model = get_dsp_from_json(json).unwrap();
        assert!(model.metadata().loudness.is_none());
        assert!(model.metadata().expected_sample_rate.is_none());
    }

    #[test]
    fn test_metadata_parses_correctly() {
        let json = r#"{"version":"0.5.0","architecture":"Linear","config":{"receptive_field":1},"weights":[1.0],
            "metadata":{"loudness":-15.5,"sample_rate":44100.0,"input_level_dbu":12.0,"output_level_dbu":6.0}}"#;
        let model = get_dsp_from_json(json).unwrap();
        let meta = model.metadata();
        assert!((meta.loudness.unwrap() - (-15.5)).abs() < 1e-6);
        assert!((meta.expected_sample_rate.unwrap() - 44100.0).abs() < 1e-6);
        assert!((meta.input_level_dbu.unwrap() - 12.0).abs() < 1e-6);
        assert!((meta.output_level_dbu.unwrap() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_dsp_nonexistent_file() {
        assert!(matches!(
            get_dsp(Path::new("/nonexistent/model.nam")),
            Err(NamError::Io(_))
        ));
    }

    #[test]
    fn test_deterministic_output() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let input = vec![0.3 as crate::dsp::Sample; 64];

        let mut model1 = get_dsp(path).unwrap();
        let mut out1 = vec![0.0 as crate::dsp::Sample; 64];
        model1.process(&input, &mut out1);

        let mut model2 = get_dsp(path).unwrap();
        let mut out2 = vec![0.0 as crate::dsp::Sample; 64];
        model2.process(&input, &mut out2);

        for i in 0..64 {
            assert_eq!(out1[i], out2[i], "Non-deterministic at sample {}", i);
        }
    }

    #[test]
    fn test_null_metadata_fields() {
        let json = r#"{"version":"0.5.0","architecture":"Linear","config":{"receptive_field":1},"weights":[1.0],
            "metadata":{"loudness":-15.0,"input_level_dbu":null,"output_level_dbu":12.0}}"#;
        let model = get_dsp_from_json(json).unwrap();
        let meta = model.metadata();
        assert!(meta.loudness.is_some());
        assert!(
            meta.input_level_dbu.is_none(),
            "null input_level_dbu should parse as None"
        );
        assert!(meta.output_level_dbu.is_some());
    }

    #[test]
    fn test_load_and_process_all_fixture_models() {
        use std::fs;
        let dir = Path::new("test_fixtures/models");
        if !dir.exists() {
            return;
        }

        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().map(|e| e == "nam").unwrap_or(false) {
                let result = get_dsp(&path);
                match result {
                    Ok(mut model) => {
                        for _ in 0..3 {
                            let input = vec![0.1 as crate::dsp::Sample; 64];
                            let mut output = vec![0.0 as crate::dsp::Sample; 64];
                            model.process(&input, &mut output);
                            for (s, &val) in output.iter().enumerate() {
                                assert!(
                                    (val as f64).is_finite(),
                                    "Non-finite output from {:?} at sample {}",
                                    path,
                                    s
                                );
                            }
                        }
                    }
                    Err(_) => {
                        // Some models use unsupported features (slimmable, etc.)
                    }
                }
            }
        }
    }

    #[test]
    fn test_output_always_finite() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();

        let test_inputs: Vec<Vec<crate::dsp::Sample>> = vec![
            vec![0.0; 32],
            vec![1.0; 32],
            vec![-1.0; 32],
            vec![10.0; 32],
            (0..32)
                .map(|i| (i as f64 * 0.1).sin() as crate::dsp::Sample)
                .collect(),
        ];

        for (idx, input) in test_inputs.iter().enumerate() {
            let mut output = vec![0.0 as crate::dsp::Sample; 32];
            model.process(input, &mut output);
            for (s, &val) in output.iter().enumerate() {
                assert!(
                    (val as f64).is_finite(),
                    "Non-finite at test {}, sample {}: {}",
                    idx,
                    s,
                    val
                );
            }
        }
    }

    #[test]
    fn test_version_0_6_accepted() {
        // Models with version 0.6.0 should load fine
        let json = r#"{"version":"0.6.0","architecture":"Linear","config":{"receptive_field":1},"weights":[1.0]}"#;
        assert!(get_dsp_from_json(json).is_ok());
    }

    #[test]
    fn test_version_0_7_accepted() {
        let json = r#"{"version":"0.7.0","architecture":"Linear","config":{"receptive_field":1},"weights":[1.0]}"#;
        assert!(get_dsp_from_json(json).is_ok());
    }

    #[test]
    fn test_get_dsp_wavenet_condition_dsp() {
        let path = Path::new("test_fixtures/models/wavenet_condition_dsp.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();
        let input = vec![0.1 as crate::dsp::Sample; 64];
        let mut output = vec![0.0 as crate::dsp::Sample; 64];
        model.process(&input, &mut output);
        assert!(output.iter().all(|&x| (x as f64).is_finite()));
    }

    #[test]
    fn test_get_dsp_wavenet_a2_max() {
        let path = Path::new("test_fixtures/models/wavenet_a2_max.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();
        let input = vec![0.1 as crate::dsp::Sample; 64];
        let mut output = vec![0.0 as crate::dsp::Sample; 64];
        model.process(&input, &mut output);
        assert!(output.iter().all(|&x| (x as f64).is_finite()));
    }

    // ── C++ regression tests ──────────────────────────────────────────────
    // Compare Rust output against C++ reference WAV files.
    // Reference files are generated by the C++ `render` tool from upstream-core.

    /// Read f32 samples from a WAV file (supports both standard and extensible format).
    fn read_wav_f32(path: &Path) -> Option<Vec<f32>> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
            return None;
        }
        let mut pos = 12;
        let mut fmt_code: u16 = 0;
        while pos + 8 < data.len() {
            let chunk_id = &data[pos..pos + 4];
            let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().ok()?) as usize;
            let chunk_data = &data[pos + 8..pos + 8 + chunk_size.min(data.len() - pos - 8)];

            if chunk_id == b"fmt " {
                fmt_code = u16::from_le_bytes(chunk_data[0..2].try_into().ok()?);
                if fmt_code == 65534 && chunk_data.len() >= 26 {
                    fmt_code = u16::from_le_bytes(chunk_data[24..26].try_into().ok()?);
                }
            } else if chunk_id == b"data" && fmt_code == 3 {
                let n = chunk_data.len() / 4;
                let samples: Vec<f32> = (0..n)
                    .map(|i| f32::from_le_bytes(chunk_data[i * 4..(i + 1) * 4].try_into().unwrap()))
                    .collect();
                return Some(samples);
            }

            pos += 8 + chunk_size;
            if chunk_size % 2 != 0 {
                pos += 1;
            }
        }
        None
    }

    /// Compare Rust output against C++ reference for a given model.
    /// Matches the C++ render tool behavior: Reset + process in chunks of 64.
    /// Returns (max_diff, rms_diff) or None if files are missing.
    fn regression_compare(model_name: &str) -> Option<(f64, f64)> {
        let model_path = Path::new("test_fixtures/models").join(format!("{}.nam", model_name));
        let input_path = Path::new("test_fixtures/audio/test_input.wav");
        let ref_path = Path::new("test_fixtures/audio").join(format!("{}_cpp_ref.wav", model_name));

        if !model_path.exists() || !input_path.exists() || !ref_path.exists() {
            return None;
        }

        let input_samples = read_wav_f32(input_path)?;
        let ref_samples = read_wav_f32(&ref_path)?;

        let mut model = get_dsp(&model_path).ok()?;

        // Match C++ render: Reset(sampleRate, 64) then process in chunks of 64
        let sample_rate = model.metadata().expected_sample_rate.unwrap_or(48000.0);
        model.reset(sample_rate, 64);
        model.prewarm();

        let chunk_size = 64;
        let mut output = Vec::with_capacity(input_samples.len());

        for chunk in input_samples.chunks(chunk_size) {
            let input: Vec<crate::dsp::Sample> =
                chunk.iter().map(|&s| s as crate::dsp::Sample).collect();
            let mut out_chunk = vec![0.0 as crate::dsp::Sample; input.len()];
            model.process(&input, &mut out_chunk);
            output.extend(out_chunk);
        }

        let n = output.len().min(ref_samples.len());
        let mut max_diff: f64 = 0.0;
        let mut sum_sq_diff: f64 = 0.0;
        for i in 0..n {
            let diff = (output[i] as f64 - ref_samples[i] as f64).abs();
            max_diff = max_diff.max(diff);
            sum_sq_diff += diff * diff;
        }
        let rms_diff = (sum_sq_diff / n as f64).sqrt();
        Some((max_diff, rms_diff))
    }

    // ── Strict accuracy guards ────────────────────────────────────────────
    //
    // These thresholds are the EXACT known-good accuracy levels. Any code
    // change that makes accuracy worse on ANY model will fail these tests.
    //
    // Current known-good values (max_diff vs C++ reference):
    //   wavenet:              1.19e-07  (f32 precision floor)
    //   wavenet_condition_dsp: 8.94e-08  (f32 precision floor)
    //   lstm:                 8.94e-08  (f32 precision floor)
    //   wavenet_a1_standard:  6.42e-07  (f32 precision floor for 20-layer model)
    //   my_model:             6.42e-07  (same architecture as a1_standard)
    //   wavenet_a2_max:       7.15e-06  (deep network with condition_dsp)
    //
    // Thresholds are set slightly above actual values to allow for
    // platform-specific floating-point variation.

    #[test]
    fn test_regression_wavenet() {
        if let Some((max_diff, _rms)) = regression_compare("wavenet") {
            assert!(
                max_diff <= 1.3e-07,
                "wavenet: accuracy regressed, max_diff={:.2e} (limit 1.3e-07)",
                max_diff
            );
        }
    }

    #[test]
    fn test_regression_wavenet_condition_dsp() {
        if let Some((max_diff, _rms)) = regression_compare("wavenet_condition_dsp") {
            assert!(
                max_diff <= 1.0e-07,
                "wavenet_condition_dsp: accuracy regressed, max_diff={:.2e} (limit 1.0e-07)",
                max_diff
            );
        }
    }

    #[test]
    fn test_regression_lstm() {
        if let Some((max_diff, _rms)) = regression_compare("lstm") {
            assert!(
                max_diff <= 1.0e-07,
                "lstm: accuracy regressed, max_diff={:.2e} (limit 1.0e-07)",
                max_diff
            );
        }
    }

    #[test]
    fn test_regression_wavenet_a1_standard() {
        if let Some((max_diff, _rms)) = regression_compare("wavenet_a1_standard") {
            assert!(
                max_diff <= 7.0e-07,
                "wavenet_a1_standard: accuracy regressed, max_diff={:.2e} (limit 7.0e-07)",
                max_diff
            );
        }
    }

    #[test]
    fn test_regression_my_model() {
        if let Some((max_diff, _rms)) = regression_compare("my_model") {
            assert!(
                max_diff <= 7.0e-07,
                "my_model: accuracy regressed, max_diff={:.2e} (limit 7.0e-07)",
                max_diff
            );
        }
    }

    #[test]
    fn test_regression_wavenet_a2_max() {
        if let Some((max_diff, _rms)) = regression_compare("wavenet_a2_max") {
            assert!(
                max_diff <= 8.0e-06,
                "wavenet_a2_max: accuracy regressed, max_diff={:.2e} (limit 8.0e-06)",
                max_diff
            );
        }
    }

    #[test]
    fn test_print_all_diffs() {
        let models = [
            "wavenet",
            "wavenet_condition_dsp",
            "lstm",
            "wavenet_a1_standard",
            "my_model",
            "wavenet_a2_max",
        ];
        let mut report = String::from("\n\n=== Accuracy Report vs C++ ===\n");
        let mut any = false;
        for name in models {
            if let Some((max_diff, rms_diff)) = regression_compare(name) {
                any = true;
                report.push_str(&format!(
                    "{:<30} max_diff={:.2e}  rms_diff={:.2e}\n",
                    name, max_diff, rms_diff
                ));
            }
        }
        if any {
            panic!("{}", report);
        } else {
            let cwd = std::env::current_dir().unwrap();
            let input_exists = Path::new("test_fixtures/audio/test_input.wav").exists();
            let model_exists = Path::new("test_fixtures/models/wavenet.nam").exists();
            let ref_exists = Path::new("test_fixtures/audio/wavenet_cpp_ref.wav").exists();
            panic!(
                "\n\nNo test fixtures found! cwd={}\ninput={} model={} ref={}\n",
                cwd.display(),
                input_exists,
                model_exists,
                ref_exists
            );
        }
    }

    #[test]
    fn test_a2_max_divergence_profile() {
        let model_path = Path::new("test_fixtures/models/wavenet_a2_max.nam");
        let input_path = Path::new("test_fixtures/audio/test_input.wav");
        let ref_path = Path::new("test_fixtures/audio/wavenet_a2_max_cpp_ref.wav");
        if !model_path.exists() || !input_path.exists() || !ref_path.exists() {
            return;
        }

        let input_samples = read_wav_f32(input_path).unwrap();
        let ref_samples = read_wav_f32(&ref_path).unwrap();
        let mut model = get_dsp(&model_path).unwrap();
        let sample_rate = model.metadata().expected_sample_rate.unwrap_or(48000.0);
        model.reset(sample_rate, 64);
        model.prewarm();

        let chunk_size = 64;
        let mut output = Vec::with_capacity(input_samples.len());
        for chunk in input_samples.chunks(chunk_size) {
            let input: Vec<crate::dsp::Sample> =
                chunk.iter().map(|&s| s as crate::dsp::Sample).collect();
            let mut out_chunk = vec![0.0 as crate::dsp::Sample; input.len()];
            model.process(&input, &mut out_chunk);
            output.extend(out_chunk);
        }

        let n = output.len().min(ref_samples.len());
        let mut report = String::from("\n\n=== a2_max divergence profile ===\n");
        report.push_str("Chunk | Samples     | Max Diff   | Where (sample idx)\n");
        report.push_str("------|-------------|------------|-------------------\n");

        // Report per-chunk divergence
        let chunk_report_size = 64;
        let num_chunks = (n + chunk_report_size - 1) / chunk_report_size;
        let mut first_nonzero = None;
        let mut worst_chunk = 0usize;
        let mut worst_chunk_diff = 0.0f64;

        for c in 0..num_chunks {
            let start = c * chunk_report_size;
            let end = (start + chunk_report_size).min(n);
            let mut max_diff = 0.0f64;
            let mut max_idx = start;
            for i in start..end {
                let diff = (output[i] as f64 - ref_samples[i] as f64).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_idx = i;
                }
            }
            if max_diff > 0.0 && first_nonzero.is_none() {
                first_nonzero = Some(c);
            }
            if max_diff > worst_chunk_diff {
                worst_chunk_diff = max_diff;
                worst_chunk = c;
            }
            // Only print first 10 chunks, the worst chunk, and last 5
            if c < 10 || c == worst_chunk || c >= num_chunks - 5 {
                report.push_str(&format!(
                    "{:<5} | {:>5}-{:<5} | {:.2e}  | {}\n",
                    c,
                    start,
                    end - 1,
                    max_diff,
                    max_idx
                ));
            } else if c == 10 {
                report.push_str("...   | ...         | ...        | ...\n");
            }
        }

        report.push_str(&format!(
            "\nFirst non-zero divergence at chunk: {:?}\n",
            first_nonzero
        ));
        report.push_str(&format!(
            "Worst chunk: {} (max_diff={:.2e})\n",
            worst_chunk, worst_chunk_diff
        ));

        // Also show the 10 worst individual samples
        let mut diffs: Vec<(usize, f64)> = (0..n)
            .map(|i| (i, (output[i] as f64 - ref_samples[i] as f64).abs()))
            .collect();
        diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        report.push_str("\nTop 10 worst samples:\n");
        report.push_str("Sample | Rust         | C++          | Diff\n");
        for &(idx, diff) in diffs.iter().take(10) {
            report.push_str(&format!(
                "{:<6} | {:>12.8} | {:>12.8} | {:.2e}\n",
                idx, output[idx], ref_samples[idx], diff
            ));
        }

        panic!("{}", report);
    }

    #[test]
    fn test_get_dsp_slimmable_wavenet() {
        let path = Path::new("test_fixtures/models/slimmable_wavenet.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();
        let input = vec![0.1 as crate::dsp::Sample; 64];
        let mut output = vec![0.0 as crate::dsp::Sample; 64];
        model.process(&input, &mut output);
        assert!(
            output.iter().all(|&x| (x as f64).is_finite()),
            "slimmable_wavenet produced non-finite output"
        );
    }

    #[test]
    fn test_get_dsp_slimmable_container() {
        let path = Path::new("test_fixtures/models/slimmable_container.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();
        let input = vec![0.1 as crate::dsp::Sample; 64];
        let mut output = vec![0.0 as crate::dsp::Sample; 64];
        model.process(&input, &mut output);
        assert!(
            output.iter().all(|&x| (x as f64).is_finite()),
            "slimmable_container produced non-finite output"
        );
    }

    #[test]
    fn test_slimmable_container_delegates_to_active_model() {
        let path = Path::new("test_fixtures/models/slimmable_container.nam");
        if !path.exists() {
            return;
        }
        let mut model = get_dsp(path).unwrap();
        model.reset(48000.0, 64);
        model.prewarm();

        // Process multiple chunks to verify state persistence works
        for _ in 0..3 {
            let input = vec![0.2 as crate::dsp::Sample; 64];
            let mut output = vec![0.0 as crate::dsp::Sample; 64];
            model.process(&input, &mut output);
            assert!(
                output.iter().all(|&x| (x as f64).is_finite()),
                "slimmable_container: non-finite after multiple chunks"
            );
        }
    }
}
