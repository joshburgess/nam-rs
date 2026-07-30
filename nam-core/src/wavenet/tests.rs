use super::*;
use proptest::prelude::*;
use std::path::Path;

fn load_wavenet(filename: &str) -> Option<WaveNet> {
    let path = Path::new("test_fixtures/models").join(filename);
    if !path.exists() {
        eprintln!("Skipping test: {:?} not found", path);
        return None;
    }
    let content = std::fs::read_to_string(&path).unwrap();
    let root: serde_json::Value = serde_json::from_str(&content).unwrap();
    let weights: Vec<f32> = root["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let metadata = DspMetadata::default();
    let config = &root["config"];

    // Check for condition_dsp
    let condition_dsp = if let Some(cd) = config.get("condition_dsp") {
        if !cd.is_null() {
            match crate::get_dsp::get_dsp_from_value(cd) {
                Ok(dsp) => Some(dsp),
                Err(e) => {
                    eprintln!("Failed to load condition_dsp for {}: {}", filename, e);
                    return None;
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    match WaveNet::from_config_with_condition_dsp(config, &weights, metadata, condition_dsp) {
        Ok(wn) => Some(wn),
        Err(e) => {
            eprintln!("Failed to load {}: {}", filename, e);
            None
        }
    }
}

#[test]
fn test_wavenet_loads() {
    let model = load_wavenet("wavenet.nam");
    assert!(model.is_some(), "wavenet.nam should load");
}

#[test]
fn test_wavenet_processes() {
    let mut model = match load_wavenet("wavenet.nam") {
        Some(m) => m,
        None => return,
    };

    let input = vec![0.0 as Sample; 128];
    let mut output = vec![0.0 as Sample; 128];
    model.process(&input, &mut output);

    let mut impulse = vec![0.0 as Sample; 128];
    impulse[0] = 1.0 as Sample;
    let mut out2 = vec![0.0 as Sample; 128];
    model.process(&impulse, &mut out2);

    let has_nonzero = out2.iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "WaveNet output was all zeros after impulse");
}

#[test]
fn test_slimmable_wavenet_set_slimming() {
    let mut model = match load_wavenet("slimmable_wavenet.nam") {
        Some(m) => m,
        None => return,
    };
    let input = vec![0.1 as Sample; 64];
    let mut output = vec![0.0 as Sample; 64];

    model.set_slimming(0.0).unwrap();
    model.reset(48_000.0, 64);
    model.process(&input, &mut output);
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "SlimmableWaveNet smallest width produced non-finite output"
    );

    model.set_slimming(1.0).unwrap();
    model.reset(48_000.0, 64);
    model.process(&input, &mut output);
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "SlimmableWaveNet largest width produced non-finite output"
    );
}

#[test]
fn test_slimmable_wavenet_set_slimming_changes_output() {
    let input = vec![0.1 as Sample; 64];

    let mut smallest = match load_wavenet("slimmable_wavenet.nam") {
        Some(m) => m,
        None => return,
    };
    smallest.set_slimming(0.0).unwrap();
    smallest.reset(48_000.0, 64);
    let mut smallest_output = vec![0.0 as Sample; 64];
    smallest.process(&input, &mut smallest_output);

    let mut largest = match load_wavenet("slimmable_wavenet.nam") {
        Some(m) => m,
        None => return,
    };
    largest.set_slimming(1.0).unwrap();
    largest.reset(48_000.0, 64);
    let mut largest_output = vec![0.0 as Sample; 64];
    largest.process(&input, &mut largest_output);

    assert_ne!(
        smallest_output, largest_output,
        "SlimmableWaveNet width selection should affect the rendered output"
    );
}

fn slimmable_wavenet_with_channels(allowed_channels: &[usize]) -> Result<WaveNet, NamError> {
    let path = Path::new("test_fixtures/models/slimmable_wavenet.nam");
    let content = std::fs::read_to_string(path).unwrap();
    let mut root: serde_json::Value = serde_json::from_str(&content).unwrap();
    root["config"]["layers"][0]["slimmable"]["kwargs"]["allowed_channels"] =
        serde_json::json!(allowed_channels);
    let weights = root["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    WaveNet::from_config(&root["config"], &weights, DspMetadata::default())
}

#[test]
fn test_slimmable_wavenet_breakpoints() {
    let model = slimmable_wavenet_with_channels(&[1, 2, 3]).unwrap();

    assert_eq!(model.slimming_breakpoints(), vec![1.0 / 3.0, 2.0 / 3.0]);
}

#[test]
fn test_slimmable_wavenet_validates_allowed_channel_contract() {
    for allowed_channels in [&[1, 1, 3][..], &[2, 1, 3], &[1, 2]] {
        let result = slimmable_wavenet_with_channels(allowed_channels);
        assert!(
            matches!(result, Err(NamError::InvalidConfigField { .. })),
            "invalid allowed_channels were accepted: {allowed_channels:?}"
        );
    }
}

#[test]
fn test_all_example_models_load() {
    let models = [
        "wavenet.nam",
        "wavenet_a1_standard.nam",
        "my_model.nam",
        "wavenet_a2_max.nam",
        "wavenet_condition_dsp.nam",
    ];
    for name in &models {
        let path = Path::new("test_fixtures/models").join(name);
        if !path.exists() {
            eprintln!("Skipping: {:?}", path);
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&content).unwrap();
        if root["architecture"].as_str() != Some("WaveNet") {
            continue;
        }
        let model = load_wavenet(name);
        assert!(model.is_some(), "Failed to load {}", name);
    }
}

#[test]
fn test_all_example_models_process() {
    let models = [
        "wavenet.nam",
        "wavenet_a1_standard.nam",
        "my_model.nam",
        "wavenet_a2_max.nam",
        "wavenet_condition_dsp.nam",
    ];
    for name in &models {
        let path = Path::new("test_fixtures/models").join(name);
        if !path.exists() {
            eprintln!("Skipping: {:?}", path);
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&content).unwrap();
        if root["architecture"].as_str() != Some("WaveNet") {
            continue;
        }
        let mut model = match load_wavenet(name) {
            Some(m) => m,
            None => {
                panic!("Failed to load {}", name);
            }
        };

        // Process some audio
        let input = vec![0.1 as Sample; 64];
        let mut output = vec![0.0 as Sample; 64];
        model.process(&input, &mut output);

        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Non-finite output from {}",
            name
        );
    }
}

fn minimal_upstream_layer_config() -> serde_json::Value {
    serde_json::json!({
        "condition_size": 1,
        "input_size": 1,
        "channels": 1,
        "head": {"out_channels": 1, "kernel_size": 1, "bias": false},
        "kernel_size": 1,
        "dilations": [1],
        "activation": "Tanh",
        "layer_1x1_config": {"active": true, "groups": 1},
        "head_1x1_config": {"active": false, "out_channels": 1, "groups": 1}
    })
}

fn process_small_config(config: serde_json::Value, weights: Vec<f32>) -> Vec<Sample> {
    let mut model = WaveNet::from_config(&config, &weights, DspMetadata::default()).unwrap();
    model.reset(48_000.0, 8);
    let input = vec![0.1 as Sample; 8];
    let mut output = vec![0.0 as Sample; 8];
    model.process(&input, &mut output);
    output
}

fn render_small_config(
    activation: &str,
    weights: &[f32],
    input: &[Sample],
    chunk_size: usize,
) -> Vec<Sample> {
    let mut layer = minimal_upstream_layer_config();
    layer["activation"] = serde_json::Value::String(activation.to_string());
    let config = serde_json::json!({
        "layers_configs": [layer],
        "head": null,
        "head_scale": 1.0
    });
    let mut model = WaveNet::from_config(&config, weights, DspMetadata::default()).unwrap();
    model.reset(48_000.0, input.len().max(1));
    let mut output = Vec::with_capacity(input.len());
    for chunk in input.chunks(chunk_size) {
        let mut chunk_output = vec![Sample::default(); chunk.len()];
        model.process(chunk, &mut chunk_output);
        output.extend(chunk_output);
    }
    output
}

proptest! {
    #[test]
    fn backend_equivalence_matrix_matches_column_major_oracle(
        m in 1usize..12,
        k in 1usize..12,
        n in 1usize..12,
        padding in 0usize..4,
        raw_a in prop::collection::vec(-1.0f32..1.0, 1..144),
        raw_b in prop::collection::vec(-1.0f32..1.0, 1..180),
        raw_c in prop::collection::vec(-1.0f32..1.0, 1..144),
    ) {
        let stride = k + padding;
        let a = raw_a.into_iter().cycle().take(m * k).collect::<Vec<_>>();
        let b = raw_b.into_iter().cycle().take(stride * n).collect::<Vec<_>>();
        let initial = raw_c.into_iter().cycle().take(m * n).collect::<Vec<_>>();
        let mut actual = initial.clone();
        let mut expected = initial;
        let alpha = 0.75f32;
        let beta = -0.25f32;

        let mut layout = MatrixLayout::new(m, k).unwrap();
        layout.set_max_buffer_size(n);
        prop_assert!(layout.multiply(n, alpha, &a, &b, stride, beta, &mut actual));
        for column in 0..n {
            for row in 0..m {
                let mut product = 0.0f32;
                for inner in 0..k {
                    product += a[inner * m + row] * b[column * stride + inner];
                }
                let index = column * m + row;
                expected[index] = alpha * product + beta * expected[index];
            }
        }

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            prop_assert!(
                (actual - expected).abs() <= 2.0e-4,
                "matrix output {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn backend_equivalence_randomized_wavenet_is_block_partition_invariant(
        activation in prop_oneof![
            Just("Tanh"),
            Just("ReLU"),
            Just("Sigmoid"),
            Just("Softsign"),
        ],
        weights in prop::collection::vec(-0.75f32..0.75, 8),
        input in prop::collection::vec(-0.5f64..0.5, 1..96),
        chunk_size in 1usize..24,
    ) {
        let input = input
            .into_iter()
            .map(crate::dsp::sample_from_f64)
            .collect::<Vec<_>>();
        let full = render_small_config(activation, &weights, &input, input.len());
        let partitioned = render_small_config(activation, &weights, &input, chunk_size);
        for (index, (&full, &partitioned)) in full.iter().zip(&partitioned).enumerate() {
            prop_assert!(
                (full - partitioned).abs() <= 2.0e-5,
                "sample {index}: full={full}, partitioned={partitioned}"
            );
        }
    }
}

#[test]
fn test_upstream_layers_configs_and_head_object_load() {
    let config = serde_json::json!({
        "layers_configs": [minimal_upstream_layer_config()],
        "head": null,
        "head_scale": 1.0
    });
    let weights = vec![1.0, 0.5, 0.0, 0.25, 1.0, 0.0, 1.0, 1.0];
    let output = process_small_config(config, weights);
    assert!(output.iter().all(|&sample| sample.is_finite()));
}

#[test]
fn test_upstream_layer_aliases_and_film_params_load() {
    let mut layer = minimal_upstream_layer_config();
    layer["film_params"] = serde_json::json!({
        "conv_pre_film": {"active": true, "shift": true, "groups": 1}
    });
    let config = serde_json::json!({
        "layers_configs": [layer],
        "head": null,
        "head_scale": 1.0
    });
    let weights = vec![
        1.0, // rechannel
        0.5, 0.0,  // conv
        0.25, // input mixin
        1.0, 0.0, // layer1x1
        1.0, 0.0, 0.0, 0.0, // conv_pre_film
        1.0, // head rechannel
        1.0, // head_scale
    ];
    let output = process_small_config(config, weights);
    assert!(output.iter().all(|&sample| sample.is_finite()));
}

#[test]
fn test_upstream_pairmultiply_activation_config_loads() {
    let mut layer = minimal_upstream_layer_config();
    layer["activation"] = serde_json::json!({
        "name": "PairMultiply",
        "primary": "Tanh",
        "secondary": "Sigmoid"
    });
    let config = serde_json::json!({
        "layers_configs": [layer],
        "head": null,
        "head_scale": 1.0
    });
    let weights = vec![
        1.0, // rechannel
        0.5, 0.5, 0.0, 0.0, // gated conv
        0.25, 0.25, // gated input mixin
        1.0, 0.0, // layer1x1
        1.0, // head rechannel
        1.0, // head_scale
    ];
    let output = process_small_config(config, weights);
    assert!(output.iter().all(|&sample| sample.is_finite()));
}

#[test]
fn test_top_level_head_loads_and_processes() {
    let config = serde_json::json!({
        "layers_configs": [minimal_upstream_layer_config()],
        "head": {
            "channels": 1,
            "activation": {"name": "Softsigmoid"},
            "out_channels": 1,
            "kernel_sizes": [1]
        },
        "head_scale": 1.0
    });
    let weights = vec![
        1.0, // rechannel
        0.5, 0.0,  // conv
        0.25, // input mixin
        1.0, 0.0, // layer1x1
        1.0, // layer-array head rechannel
        1.0, 0.0, // top-level head conv
        1.0, // head_scale
    ];
    let output = process_small_config(config, weights);
    assert!(output.iter().all(|&sample| sample.is_finite()));
}

#[test]
fn test_state_persists_across_calls() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }

    let mut model_split = crate::get_dsp(path).unwrap();
    let input1 = vec![0.5 as Sample; 16];
    let mut out1 = vec![0.0 as Sample; 16];
    model_split.process(&input1, &mut out1);
    let input2 = vec![0.0 as Sample; 16];
    let mut out2a = vec![0.0 as Sample; 16];
    model_split.process(&input2, &mut out2a);

    let mut model_full = crate::get_dsp(path).unwrap();
    let mut full_input = vec![0.5 as Sample; 16];
    full_input.extend(vec![0.0 as Sample; 16]);
    let mut full_output = vec![0.0 as Sample; 32];
    model_full.process(&full_input, &mut full_output);

    for i in 0..16 {
        assert!(
            (out2a[i] - full_output[16 + i]).abs() < 1e-5,
            "State mismatch at {}: split={}, full={}",
            i,
            out2a[i],
            full_output[16 + i]
        );
    }
}

#[test]
fn test_single_sample_vs_block() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }

    let mut model_single = crate::get_dsp(path).unwrap();
    let mut outputs_single = Vec::new();
    for i in 0..32 {
        let input = vec![if i == 0 { 1.0 } else { 0.0 } as Sample];
        let mut output = vec![0.0 as Sample; 1];
        model_single.process(&input, &mut output);
        outputs_single.push(output[0]);
    }

    let mut model_block = crate::get_dsp(path).unwrap();
    let mut block_input = vec![0.0 as Sample; 32];
    block_input[0] = 1.0 as Sample;
    let mut outputs_block = vec![0.0 as Sample; 32];
    model_block.process(&block_input, &mut outputs_block);

    for i in 0..32 {
        assert!(
            (outputs_single[i] - outputs_block[i]).abs() < 1e-5,
            "Sample {} mismatch: single={}, block={}",
            i,
            outputs_single[i],
            outputs_block[i]
        );
    }
}

#[test]
fn test_prewarm_changes_output() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }

    let input = vec![0.1 as Sample; 16];

    let mut model_no_pw = crate::get_dsp(path).unwrap();
    let mut out_no_pw = vec![0.0 as Sample; 16];
    model_no_pw.process(&input, &mut out_no_pw);

    let mut model_pw = crate::get_dsp(path).unwrap();
    model_pw.prewarm();
    let mut out_pw = vec![0.0 as Sample; 16];
    model_pw.process(&input, &mut out_pw);

    let any_different = out_no_pw
        .iter()
        .zip(out_pw.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-10);
    assert!(
        any_different,
        "Prewarm should change initial output behavior"
    );
}

#[test]
fn test_prewarm_samples_positive() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }
    let model = crate::get_dsp(path).unwrap();
    assert!(model.prewarm_samples() > 0);
}

#[test]
fn test_reset_clears_state() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }

    let mut model = crate::get_dsp(path).unwrap();
    let input = vec![1.0 as Sample; 64];
    let mut output = vec![0.0 as Sample; 64];
    model.process(&input, &mut output);

    model.reset(48000.0, 4096);

    let mut model_fresh = crate::get_dsp(path).unwrap();
    let mut out_reset = vec![0.0 as Sample; 64];
    let mut out_fresh = vec![0.0 as Sample; 64];
    model.process(&input, &mut out_reset);
    model_fresh.process(&input, &mut out_fresh);

    for i in 0..64 {
        assert!(
            (out_reset[i] - out_fresh[i]).abs() < 1e-5,
            "Reset mismatch at {}: reset={}, fresh={}",
            i,
            out_reset[i],
            out_fresh[i]
        );
    }
}

fn render_fixture_with_partitions(
    model_name: &str,
    input: &[Sample],
    partitions: &[usize],
) -> Vec<Sample> {
    let mut model = load_wavenet(model_name).unwrap();
    model.reset(48_000.0, input.len());
    let mut output = vec![Sample::default(); input.len()];
    let mut offset = 0usize;
    let mut partition = 0usize;
    while offset < input.len() {
        let frames = partitions[partition % partitions.len()].min(input.len() - offset);
        model.process(
            &input[offset..offset + frames],
            &mut output[offset..offset + frames],
        );
        offset += frames;
        partition += 1;
    }
    output
}

#[test]
fn standard_wavenets_preserve_streaming_and_reset_behavior() {
    let input = (0..256)
        .map(|index| ((index as Sample + 1.0) * 0.017).sin() * 0.25)
        .collect::<Vec<_>>();
    let disturbance = vec![0.5 as Sample; 128];

    for model_name in ["wavenet_a1_standard.nam", "wavenet_a2_max.nam"] {
        let whole = render_fixture_with_partitions(model_name, &input, &[input.len()]);
        let partitioned = render_fixture_with_partitions(model_name, &input, &[17, 31, 5, 64]);
        assert_eq!(
            partitioned, whole,
            "{model_name} changed output across callback partitions"
        );

        let mut reused = load_wavenet(model_name).unwrap();
        reused.reset(48_000.0, 256);
        let mut discarded = vec![Sample::default(); disturbance.len()];
        reused.process(&disturbance, &mut discarded);
        reused.reset(48_000.0, 256);
        let mut after_reset = vec![Sample::default(); input.len()];
        reused.process(&input, &mut after_reset);

        assert_eq!(
            after_reset, whole,
            "{model_name} reset did not restore fresh state"
        );
    }
}

#[test]
fn test_large_standard_model() {
    let path = Path::new("test_fixtures/models/wavenet_a1_standard.nam");
    if !path.exists() {
        return;
    }
    let mut model = crate::get_dsp(path).unwrap();
    model.prewarm();

    let input = vec![0.1 as Sample; 256];
    let mut output = vec![0.0 as Sample; 256];
    model.process(&input, &mut output);

    assert!(output.iter().all(|&x| x.is_finite()));
    assert!(output.iter().any(|&x| x != 0.0));
}

#[test]
fn test_process_empty_buffer() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }
    let mut model = crate::get_dsp(path).unwrap();
    let input: Vec<Sample> = vec![];
    let mut output: Vec<Sample> = vec![];
    model.process(&input, &mut output);
}

#[test]
fn test_receptive_field_calculation() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }
    let model = crate::get_dsp(path).unwrap();
    assert_eq!(model.prewarm_samples(), 23);
}

#[test]
fn test_a1_standard_receptive_field() {
    let path = Path::new("test_fixtures/models/wavenet_a1_standard.nam");
    if !path.exists() {
        return;
    }
    let model = crate::get_dsp(path).unwrap();
    assert_eq!(model.prewarm_samples(), 4093);
}

fn head_dilation_config(head: serde_json::Value) -> (serde_json::Value, Vec<f32>) {
    let head_kernel_size = head
        .get("kernel_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1) as usize;
    let config = serde_json::json!({
        "layers": [{
            "input_size": 1,
            "condition_size": 1,
            "head": head,
            "channels": 1,
            "kernel_sizes": [1],
            "dilations": [1],
            "activation": "ReLU",
            "bottleneck": 1,
            "head1x1": {"active": false, "out_channels": 1, "groups": 1},
            "layer1x1": {"active": true, "groups": 1}
        }],
        "head_scale": 1.0
    });
    let weights = vec![0.25; 1 + 5 + head_kernel_size + 1];
    (config, weights)
}

#[test]
fn test_head_dilation_defaults_to_one() {
    let (config, weights) = head_dilation_config(serde_json::json!({
        "out_channels": 1,
        "kernel_size": 3,
        "bias": false
    }));

    let model = WaveNet::from_config(&config, &weights, DspMetadata::default()).unwrap();

    assert_eq!(model.prewarm_samples(), 3);
}

#[test]
fn test_head_dilation_contributes_to_receptive_field() {
    let (config, weights) = head_dilation_config(serde_json::json!({
        "out_channels": 1,
        "kernel_size": 3,
        "head_dilation": 4,
        "bias": false
    }));

    let model = WaveNet::from_config(&config, &weights, DspMetadata::default()).unwrap();

    assert_eq!(model.prewarm_samples(), 9);
}

#[test]
fn test_head_dilation_must_be_positive() {
    let (config, weights) = head_dilation_config(serde_json::json!({
        "out_channels": 1,
        "kernel_size": 3,
        "head_dilation": 0,
        "bias": false
    }));

    let result = WaveNet::from_config(&config, &weights, DspMetadata::default());

    assert!(matches!(
        result,
        Err(NamError::InvalidConfigField { field, .. })
            if field == "layer_array.head_dilation"
    ));
}

#[test]
fn test_head_dilation_type_is_validated() {
    let (config, weights) = head_dilation_config(serde_json::json!({
        "out_channels": 1,
        "kernel_size": 3,
        "head_dilation": 1.5,
        "bias": false
    }));

    let result = WaveNet::from_config(&config, &weights, DspMetadata::default());

    assert!(matches!(
        result,
        Err(NamError::InvalidConfigType { field, .. })
            if field == "layer_array.head_dilation"
    ));
}

#[test]
fn test_zero_input() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }
    let mut model = crate::get_dsp(path).unwrap();

    let input = vec![0.0 as Sample; 32];
    let mut output = vec![0.0 as Sample; 32];
    model.process(&input, &mut output);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_different_buffer_sizes() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }
    let mut model = crate::get_dsp(path).unwrap();

    for &size in &[1, 7, 16, 64, 128, 256] {
        let input = vec![0.1 as Sample; size];
        let mut output = vec![0.0 as Sample; size];
        model.process(&input, &mut output);
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Non-finite output at buffer size {}",
            size
        );
    }
}

#[test]
fn test_multiple_consecutive_calls() {
    let path = Path::new("test_fixtures/models/wavenet.nam");
    if !path.exists() {
        return;
    }
    let mut model = crate::get_dsp(path).unwrap();

    for call in 0..10 {
        let input = vec![0.1 as Sample; 8];
        let mut output = vec![0.0 as Sample; 8];
        model.process(&input, &mut output);
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Non-finite at call {}",
            call
        );
    }
}

#[test]
fn test_wavenet_a2_max_loads_and_processes() {
    let mut model = match load_wavenet("wavenet_a2_max.nam") {
        Some(m) => m,
        None => return,
    };

    let input = vec![0.1 as Sample; 64];
    let mut output = vec![0.0 as Sample; 64];
    model.process(&input, &mut output);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_wavenet_condition_dsp_loads_and_processes() {
    let mut model = match load_wavenet("wavenet_condition_dsp.nam") {
        Some(m) => m,
        None => return,
    };

    let input = vec![0.1 as Sample; 64];
    let mut output = vec![0.0 as Sample; 64];
    model.process(&input, &mut output);
    assert!(output.iter().all(|&x| x.is_finite()));
}

// ── Per-layer kernel_sizes tests ────────────────────────────────────────

/// Helper: build a minimal WaveNet JSON config and matching weight vec.
/// Returns (config_json, weights) for simple 1-channel, no-gating configs.
fn make_kernel_size_config(kernel_field: &str, num_layers: usize) -> (String, Vec<f32>) {
    // Weight budget for 1-ch, 1-bottleneck, no-gating, layer1x1-active, no-head1x1:
    //   rechannel: 1, per layer with kernel K: K+4, head_rechannel: 1, head_scale: 1
    // Parse kernel sizes from the field string to compute exact weight count.
    let kernel_sizes: Vec<usize> = if kernel_field.contains('[') {
        // Array form: extract numbers from brackets
        let start = kernel_field.find('[').unwrap();
        let end = kernel_field.find(']').unwrap();
        kernel_field[start + 1..end]
            .split(',')
            .map(|s| s.trim().parse::<usize>().unwrap())
            .collect()
    } else {
        // Scalar form: extract the number
        let num: usize = kernel_field
            .split(':')
            .next_back()
            .unwrap()
            .trim()
            .trim_matches('"')
            .parse()
            .unwrap();
        vec![num; num_layers]
    };
    let num_weights = 1 + kernel_sizes.iter().map(|k| k + 4).sum::<usize>() + 1 + 1;
    let weights = vec![1.0f32; num_weights];
    let dilations: Vec<String> = (0..num_layers).map(|i| format!("{}", 1 << i)).collect();
    let config_str = format!(
        r#"{{
                "layers": [{{
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 1,
                    {},
                    "dilations": [{}],
                    "activation": "ReLU",
                    "gated": false,
                    "head_bias": false
                }}],
                "head_scale": 1.0
            }}"#,
        kernel_field,
        dilations.join(", ")
    );
    (config_str, weights)
}

#[test]
fn test_kernel_size_int_compat() {
    // Legacy single kernel_size integer should be expanded to all layers
    let (config_str, weights) = make_kernel_size_config(r#""kernel_size": 3"#, 3);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let metadata = DspMetadata::default();
    let result = WaveNet::from_config(&config, &weights, metadata);
    assert!(
        result.is_ok(),
        "Legacy kernel_size int should parse: {:?}",
        result.err()
    );
}

#[test]
fn test_kernel_sizes_per_layer_array() {
    // New per-layer kernel_sizes array
    let (config_str, weights) = make_kernel_size_config(r#""kernel_sizes": [2, 3, 5]"#, 3);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let metadata = DspMetadata::default();
    let result = WaveNet::from_config(&config, &weights, metadata);
    assert!(
        result.is_ok(),
        "Per-layer kernel_sizes should parse: {:?}",
        result.err()
    );
}

#[test]
fn test_kernel_size_mutual_exclusivity() {
    // Providing both kernel_size and kernel_sizes should error
    let (config_str, weights) =
        make_kernel_size_config(r#""kernel_size": 3, "kernel_sizes": [2, 3, 5]"#, 3);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let metadata = DspMetadata::default();
    let result = WaveNet::from_config(&config, &weights, metadata);
    assert!(matches!(
        result,
        Err(NamError::ConflictingConfigFields { .. })
    ));
}

#[test]
fn test_kernel_sizes_length_mismatch() {
    // kernel_sizes length != dilations length should error
    let (config_str, weights) = make_kernel_size_config(r#""kernel_sizes": [2, 3]"#, 3);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let metadata = DspMetadata::default();
    let result = WaveNet::from_config(&config, &weights, metadata);
    assert!(matches!(result, Err(NamError::ConfigLengthMismatch { .. })));
}

#[test]
fn test_receptive_field_overflow_is_rejected_before_reading_weights() {
    let (config_str, _) = make_kernel_size_config(r#""kernel_size": 3"#, 1);
    let mut config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    config["layers"][0]["dilations"][0] = serde_json::json!(u64::MAX);

    let result = WaveNet::from_config(&config, &[], DspMetadata::default());

    #[cfg(target_pointer_width = "64")]
    assert!(matches!(
        result,
        Err(NamError::DimensionOverflow {
            context: "convolution receptive field",
            ..
        })
    ));
    #[cfg(not(target_pointer_width = "64"))]
    assert!(matches!(
        result,
        Err(NamError::ConfigIntegerOutOfRange { .. })
    ));
}

#[test]
fn test_no_kernel_size_field_errors() {
    // Neither kernel_size nor kernel_sizes should error
    let config_str = r#"{
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "head_size": 1,
                "channels": 1,
                "dilations": [1, 2],
                "activation": "ReLU",
                "gated": false,
                "head_bias": false
            }],
            "head_scale": 1.0
        }"#;
    let config: serde_json::Value = serde_json::from_str(config_str).unwrap();
    let weights = vec![1.0f32; 500];
    let metadata = DspMetadata::default();
    let result = WaveNet::from_config(&config, &weights, metadata);
    assert!(result.is_err(), "Missing kernel_size should be rejected");
}

#[test]
fn test_kernel_sizes_per_layer_different_receptive_fields() {
    // With kernel_sizes [2, 3] and dilations [1, 2]:
    //   layer 0: RF = 1 * (2-1) = 1
    //   layer 1: RF = 2 * (3-1) = 4
    //   total RF = 5, prewarm = 1 (base) + 5 = 6
    let (config_str, weights) = make_kernel_size_config(r#""kernel_sizes": [2, 3]"#, 2);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let metadata = DspMetadata::default();
    let model = WaveNet::from_config(&config, &weights, metadata).unwrap();
    assert_eq!(model.prewarm_samples(), 6);
}

#[test]
fn test_kernel_size_as_array() {
    // kernel_size (singular key) with array value should also be accepted
    // for compatibility with trainer exports
    let (config_str, weights) = make_kernel_size_config(r#""kernel_size": [2, 3]"#, 2);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let metadata = DspMetadata::default();
    let result = WaveNet::from_config(&config, &weights, metadata);
    assert!(
        result.is_ok(),
        "kernel_size as array should parse: {:?}",
        result.err()
    );
    let model = result.unwrap();
    // dilations [1, 2], kernel_sizes [2, 3]: RF = 1*(2-1) + 2*(3-1) = 5, prewarm = 1 + 5 = 6
    assert_eq!(model.prewarm_samples(), 6);
}

#[test]
fn test_kernel_size_int_receptive_field() {
    // With kernel_size=3 and dilations [1, 2]:
    //   layer 0: RF = 1 * (3-1) = 2
    //   layer 1: RF = 2 * (3-1) = 4
    //   total RF = 6, prewarm = 1 + 6 = 7
    let (config_str, weights) = make_kernel_size_config(r#""kernel_size": 3"#, 2);
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let metadata = DspMetadata::default();
    let model = WaveNet::from_config(&config, &weights, metadata).unwrap();
    assert_eq!(model.prewarm_samples(), 7);
}

// ── Depthwise convolution tests ─────────────────────────────────────────

#[test]
fn test_conv1d_depthwise_detected() {
    // groups == in_channels == out_channels triggers depthwise path
    let weights_data = vec![1.0f32; 100];
    let mut iter = crate::util::WeightIter::new(&weights_data);
    let conv = Conv1d::from_weights(4, 4, 3, 1, 4, &mut iter).unwrap();
    assert!(matches!(conv.weights, Conv1dWeights::Depthwise(_)));
}

#[test]
fn test_conv1d_general_when_not_depthwise() {
    // groups != in_channels should use general path
    let weights_data = vec![1.0f32; 100];
    let mut iter = crate::util::WeightIter::new(&weights_data);
    let conv = Conv1d::from_weights(4, 4, 3, 1, 2, &mut iter).unwrap();
    assert!(matches!(conv.weights, Conv1dWeights::General(_)));
}

#[cfg(feature = "fast-kernels")]
#[test]
fn test_grouped_conv1d_preserves_compact_tap_major_weights() {
    let mut weights_data = (1..=24).map(|value| value as f32).collect::<Vec<_>>();
    weights_data.extend([0.0; 12]);
    let mut iter = crate::util::WeightIter::new(&weights_data);
    let conv = Conv1d::from_weights(3, 12, 2, 1, 3, &mut iter).unwrap();

    assert_eq!(
        conv.compact_grouped_weights.as_deref(),
        Some(
            &[
                1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 2.0, 4.0, 6.0,
                8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
            ][..]
        )
    );
}

#[test]
fn test_conv1d_depthwise_identity() {
    // 2-channel depthwise with kernel_size=1, weights=[1,1]
    // Should act as identity (plus bias)
    let weights_data = vec![1.0, 1.0, 0.0, 0.0]; // 2 weights + 2 bias
    let mut iter = crate::util::WeightIter::new(&weights_data);
    let mut conv = Conv1d::from_weights(2, 2, 1, 1, 2, &mut iter).unwrap();
    conv.set_max_buffer_size(4);

    let mut input = ColMajorMatrix::new(2, 4);
    // Frame 0: [3.0, 5.0], Frame 1: [7.0, 11.0]
    input.data[0] = 3.0;
    input.data[1] = 5.0;
    input.data[2] = 7.0;
    input.data[3] = 11.0;

    conv.process_block(&input, 2);
    // With weight=1 and bias=0: output should equal input
    assert!((conv.output_buf.data[0] - 3.0).abs() < 1e-6);
    assert!((conv.output_buf.data[1] - 5.0).abs() < 1e-6);
    assert!((conv.output_buf.data[2] - 7.0).abs() < 1e-6);
    assert!((conv.output_buf.data[3] - 11.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_depthwise_scaling() {
    // 2-channel depthwise with kernel_size=1, weights=[2, 3], bias=[10, 20]
    let weights_data = vec![2.0, 3.0, 10.0, 20.0];
    let mut iter = crate::util::WeightIter::new(&weights_data);
    let mut conv = Conv1d::from_weights(2, 2, 1, 1, 2, &mut iter).unwrap();
    conv.set_max_buffer_size(4);

    let mut input = ColMajorMatrix::new(2, 4);
    input.data[0] = 1.0; // ch0, frame0
    input.data[1] = 1.0; // ch1, frame0

    conv.process_block(&input, 1);
    // ch0: 2*1 + 10 = 12, ch1: 3*1 + 20 = 23
    assert!((conv.output_buf.data[0] - 12.0).abs() < 1e-6);
    assert!((conv.output_buf.data[1] - 23.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_depthwise_multi_tap() {
    // 2-channel depthwise with kernel_size=2, dilation=1
    // weights: ch0=[1, 2], ch1=[3, 4], bias=[0, 0]
    // C++ weight order for depthwise: for each channel c, for each tap k
    let weights_data = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
    let mut iter = crate::util::WeightIter::new(&weights_data);
    let mut conv = Conv1d::from_weights(2, 2, 2, 1, 2, &mut iter).unwrap();
    conv.set_max_buffer_size(4);

    // Process two calls to build up ring buffer history
    let mut input1 = ColMajorMatrix::new(2, 4);
    input1.data[0] = 1.0; // ch0
    input1.data[1] = 0.0; // ch1
    conv.process_block(&input1, 1);

    let mut input2 = ColMajorMatrix::new(2, 4);
    input2.data[0] = 0.0; // ch0
    input2.data[1] = 1.0; // ch1
    conv.process_block(&input2, 1);
    // Tap ordering: k=0 has lookback=1 (prev), k=1 has lookback=0 (current)
    // Frame 1 output:
    //   ch0: w[0]*prev[ch0] + w[1]*now[ch0] = 1*1 + 2*0 = 1
    //   ch1: w[0]*prev[ch1] + w[1]*now[ch1] = 3*0 + 4*1 = 4
    assert!(
        (conv.output_buf.data[0] - 1.0).abs() < 1e-6,
        "ch0: {}",
        conv.output_buf.data[0]
    );
    assert!(
        (conv.output_buf.data[1] - 4.0).abs() < 1e-6,
        "ch1: {}",
        conv.output_buf.data[1]
    );
}
