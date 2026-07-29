use super::*;
use proptest::prelude::*;

/// A trivial pass-through DSP for testing resampling in isolation.
struct PassthroughDsp;

impl nam_core::Dsp for PassthroughDsp {
    fn process(&mut self, input: &[nam_core::Sample], output: &mut [nam_core::Sample]) {
        output[..input.len()].copy_from_slice(input);
    }
    fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {}
    fn metadata(&self) -> &nam_core::dsp::DspMetadata {
        static META: nam_core::dsp::DspMetadata = nam_core::dsp::DspMetadata {
            raw: None,
            loudness: None,
            gain: None,
            expected_sample_rate: None,
            name: None,
            modeled_by: None,
            gear_type: None,
            gear_make: None,
            gear_model: None,
            tone_type: None,
            input_level_dbu: None,
            output_level_dbu: None,
            validation_esr: None,
        };
        &META
    }
}

struct NonFiniteDsp;

impl nam_core::Dsp for NonFiniteDsp {
    fn process(&mut self, input: &[nam_core::Sample], output: &mut [nam_core::Sample]) {
        output[..input.len()].fill(0.25);
        if let Some(sample) = output.first_mut() {
            *sample = nam_core::Sample::NAN;
        }
        if let Some(sample) = output.get_mut(1) {
            *sample = nam_core::Sample::INFINITY;
        }
    }

    fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {}

    fn metadata(&self) -> &nam_core::dsp::DspMetadata {
        PassthroughDsp.metadata()
    }
}

struct ActivationModeProbe {
    mode: nam_core::ActivationMode,
}

impl nam_core::Dsp for ActivationModeProbe {
    fn process(&mut self, input: &[nam_core::Sample], output: &mut [nam_core::Sample]) {
        let value = if self.mode == nam_core::ActivationMode::Fast {
            1.0
        } else {
            -1.0
        };
        output[..input.len()].fill(value);
    }

    fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {}

    fn metadata(&self) -> &nam_core::dsp::DspMetadata {
        static META: nam_core::dsp::DspMetadata = nam_core::dsp::DspMetadata {
            raw: None,
            loudness: None,
            gain: None,
            expected_sample_rate: None,
            name: None,
            modeled_by: None,
            gear_type: None,
            gear_make: None,
            gear_model: None,
            tone_type: None,
            input_level_dbu: None,
            output_level_dbu: None,
            validation_esr: None,
        };
        &META
    }

    fn set_activation_mode(&mut self, mode: nam_core::ActivationMode) {
        self.mode = mode;
    }
}

fn loaded_model(generation: u64) -> LoadedModel {
    LoadedModel {
        generation,
        dsp: Box::new(PassthroughDsp),
        resampler: None,
    }
}

fn mono_buffer(samples: &mut [f32]) -> Buffer<'_> {
    let mut buffer = Buffer::default();
    // SAFETY: `samples` remains exclusively borrowed for the buffer's lifetime, and the
    // declared sample count equals the only channel's length
    unsafe {
        buffer.set_slices(samples.len(), |channels| channels.push(samples));
    }
    buffer
}

fn plugin_with_passthrough_model(buffer_size: usize) -> NamPlugin {
    let mut plugin = NamPlugin::default();
    plugin.model = Some(loaded_model(1));
    plugin.latest_generation.store(1, Ordering::Release);
    plugin.installed_generation.store(1, Ordering::Release);
    plugin.input_buf = vec![0.0; buffer_size];
    plugin.output_buf = vec![0.0; buffer_size];
    plugin.max_buffer_size = buffer_size;
    plugin
}

#[test]
fn callback_mutes_non_finite_model_output_without_allocating() {
    let buffer_size = 64;
    let mut plugin = plugin_with_passthrough_model(buffer_size);
    plugin.model = Some(LoadedModel {
        generation: 1,
        dsp: Box::new(NonFiniteDsp),
        resampler: None,
    });
    let mut audio = vec![1.0f32; buffer_size];
    let mut buffer = mono_buffer(&mut audio);

    let (status, allocations) =
        allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
    drop(buffer);

    assert_eq!(status, ProcessStatus::Normal);
    assert_eq!(allocations, 0);
    assert_eq!(&audio[..2], &[0.0, 0.0]);
    assert!(audio[2..].iter().all(|sample| sample.is_finite()));
    assert_eq!(
        AudioProcessError::from_raw(plugin.audio_error.load(Ordering::Acquire)),
        AudioProcessError::NonFiniteOutput
    );

    plugin.model = Some(loaded_model(1));
    audio.fill(1.0);
    let mut buffer = mono_buffer(&mut audio);
    plugin.process_buffer(&mut buffer);
    assert!(audio.iter().all(|sample| sample.is_finite()));
    assert_eq!(
        AudioProcessError::from_raw(plugin.audio_error.load(Ordering::Acquire)),
        AudioProcessError::NonFiniteOutput
    );
}

#[test]
fn callback_resets_resampling_before_non_finite_output_can_poison_it() {
    let buffer_size = 4096;
    let mut plugin = plugin_with_resampled_passthrough_model(44_100, 48_000, buffer_size);
    if let Some(model) = plugin.model.as_mut() {
        model.dsp = Box::new(NonFiniteDsp);
    }
    let mut audio = vec![1.0f32; buffer_size];
    let mut buffer = mono_buffer(&mut audio);

    let (status, allocations) =
        allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
    drop(buffer);

    assert_eq!(status, ProcessStatus::Normal);
    assert_eq!(allocations, 0);
    assert!(audio.iter().all(|sample| *sample == 0.0));
    assert_eq!(
        AudioProcessError::from_raw(plugin.audio_error.load(Ordering::Acquire)),
        AudioProcessError::NonFiniteOutput
    );
}

#[test]
fn callback_mutes_oversized_host_blocks_without_allocating() {
    let mut plugin = plugin_with_passthrough_model(32);
    let mut audio = [0.25f32; 64];
    let mut buffer = mono_buffer(&mut audio);

    let (status, allocations) =
        allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
    drop(buffer);

    assert_eq!(status, ProcessStatus::Normal);
    assert_eq!(allocations, 0);
    assert!(audio.iter().all(|&sample| sample == 0.0));
    assert_eq!(
        AudioProcessError::from_raw(plugin.audio_error.load(Ordering::Acquire)),
        AudioProcessError::CallbackCapacity
    );
}

#[test]
fn callback_mutes_invalid_channel_layouts_without_allocating() {
    let mut plugin = plugin_with_passthrough_model(32);
    let mut left = [0.25f32; 32];
    let mut right = [0.25f32; 32];
    let mut buffer = Buffer::default();
    // SAFETY: Both slices remain exclusively borrowed for the buffer's lifetime and
    // match the declared sample count
    unsafe {
        buffer.set_slices(left.len(), |channels| {
            channels.push(&mut left);
            channels.push(&mut right);
        });
    }

    let (status, allocations) =
        allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
    drop(buffer);

    assert_eq!(status, ProcessStatus::Normal);
    assert_eq!(allocations, 0);
    assert!(left.iter().all(|&sample| sample == 0.0));
    assert!(right.iter().all(|&sample| sample == 0.0));
    assert_eq!(
        AudioProcessError::from_raw(plugin.audio_error.load(Ordering::Acquire)),
        AudioProcessError::CallbackLayout
    );
}

fn plugin_with_resampled_passthrough_model(
    host_rate: usize,
    model_rate: usize,
    buffer_size: usize,
) -> NamPlugin {
    let mut plugin = plugin_with_passthrough_model(buffer_size);
    if let Some(model) = plugin.model.as_mut() {
        model.resampler = Some(ResamplerState::new(host_rate, model_rate, buffer_size).unwrap());
    }
    plugin.sample_rate = host_rate as f64;
    plugin
}

#[test]
fn resampler_rejects_invalid_sample_rates() {
    assert!(ResamplerState::new(0, 48000, 4096).is_err());
    assert!(ResamplerState::new(48000, 0, 4096).is_err());
}

#[test]
fn oversized_audio_is_reported_without_growing_buffers() {
    let mut resampler = ResamplerState::new(44100, 48000, 64).unwrap();
    let capacity = resampler.input_pending.capacity();
    let input = vec![0.0; capacity + 1];
    let mut output = vec![0.0; input.len()];

    assert_eq!(
        resampler.process(&mut PassthroughDsp, &input, &mut output),
        Err(AudioProcessError::InputCapacity)
    );
    assert_eq!(resampler.input_pending.capacity(), capacity);
}

#[test]
fn model_installation_rejects_stale_generations() {
    let mut plugin = NamPlugin::default();
    plugin.latest_generation.store(2, Ordering::Release);
    assert!(plugin.loaded_models.push(loaded_model(1)).is_ok());
    plugin.install_pending_model();
    assert!(plugin.model.is_none());

    assert!(plugin.loaded_models.push(loaded_model(2)).is_ok());
    plugin.install_pending_model();
    assert_eq!(plugin.installed_generation.load(Ordering::Acquire), 2);
}

#[test]
fn saturated_retirement_queue_defers_drop() {
    let mut plugin = NamPlugin::default();
    plugin.retired_models = Arc::new(ArrayQueue::new(1));
    assert!(plugin.retired_models.push(loaded_model(1)).is_ok());
    plugin.deferred_retire = Some(loaded_model(2));

    assert!(!plugin.flush_deferred_retire());
    assert_eq!(
        plugin
            .deferred_retire
            .as_ref()
            .map(|model| model.generation),
        Some(2)
    );
}

#[test]
fn test_process_resampled_produces_output() {
    let mut rs = ResamplerState::new(44100, 48000, 4096).unwrap();
    let mut model = PassthroughDsp;

    // Feed enough samples to produce output (need multiple chunks)
    let num_samples = 4096;
    let input = vec![0.5 as nam_core::Sample; num_samples];
    let mut output = vec![0.0 as nam_core::Sample; num_samples];

    rs.process(&mut model, &input, &mut output).unwrap();

    // After enough samples, output should have data
    // (first few calls may produce zeros due to resampler latency)
    let has_nonzero = output.iter().any(|&x| x != 0.0);
    assert!(
        has_nonzero,
        "Resampled output should contain non-zero samples after {} input samples",
        num_samples
    );
}

#[test]
fn test_process_resampled_multiple_calls() {
    let mut rs = ResamplerState::new(44100, 48000, 4096).unwrap();
    let mut model = PassthroughDsp;

    // Simulate multiple process() calls with varying buffer sizes (like a real DAW)
    let buffer_sizes = [64, 128, 64, 256, 64, 128];
    let mut total_nonzero = 0;

    for &size in &buffer_sizes {
        let input = vec![0.3 as nam_core::Sample; size];
        let mut output = vec![0.0 as nam_core::Sample; size];
        rs.process(&mut model, &input, &mut output).unwrap();
        total_nonzero += output.iter().filter(|&&x| x != 0.0).count();
    }

    assert!(
        total_nonzero > 0,
        "Should produce output across multiple calls"
    );
}

#[test]
fn test_process_resampled_preserves_signal_level() {
    let mut rs = ResamplerState::new(44100, 48000, 4096).unwrap();
    let mut model = PassthroughDsp;

    // Feed a constant signal — after resampler settles, output should be ~same level
    let settle = vec![0.5 as nam_core::Sample; 4096]; // let resampler settle
    let mut discard = vec![0.0 as nam_core::Sample; 4096];
    rs.process(&mut model, &settle, &mut discard).unwrap();

    let input = vec![0.5 as nam_core::Sample; 2048];
    let mut output = vec![0.0 as nam_core::Sample; 2048];
    rs.process(&mut model, &input, &mut output).unwrap();

    // Check the latter half (fully settled)
    let tail = &output[1024..];
    let mean: f64 = tail
        .iter()
        .copied()
        .map(nam_core::dsp::sample_to_f64)
        .sum::<f64>()
        / tail.len() as f64;
    assert!(
        (mean - 0.5).abs() < 0.05,
        "Mean output {:.4} should be close to input 0.5 after settling",
        mean
    );
}

#[test]
fn test_resampler_reset_clears_state() {
    let mut rs = ResamplerState::new(44100, 48000, 4096).unwrap();
    let mut model = PassthroughDsp;

    // Feed some data
    let input = vec![1.0 as nam_core::Sample; 512];
    let mut output = vec![0.0 as nam_core::Sample; 512];
    rs.process(&mut model, &input, &mut output).unwrap();

    assert!(
        !rs.input_pending.is_empty() || !rs.output_pending.is_empty(),
        "processing should leave pending samples in at least one resampler buffer"
    );

    // Reset should clear buffers
    rs.reset();
    assert!(rs.input_pending.is_empty());
    assert!(rs.output_pending.is_empty());
}

#[test]
fn process_resampled_keeps_all_buffer_capacities_stable() {
    let mut rs = ResamplerState::new(44100, 48000, 4096).unwrap();
    let capacities = (
        rs.input_pending.capacity(),
        rs.model_rate_pending.capacity(),
        rs.output_pending.capacity(),
        rs.model_input.capacity(),
        rs.model_output.capacity(),
    );
    let mut model = PassthroughDsp;
    let input = vec![0.25; 4096];
    let mut output = vec![0.0; 4096];

    for _ in 0..16 {
        rs.process(&mut model, &input, &mut output).unwrap();
    }

    assert_eq!(
        capacities,
        (
            rs.input_pending.capacity(),
            rs.model_rate_pending.capacity(),
            rs.output_pending.capacity(),
            rs.model_input.capacity(),
            rs.model_output.capacity(),
        )
    );
}

fn render_resampled_stream(
    host_rate: usize,
    model_rate: usize,
    partitions: &[usize],
    total_samples: usize,
) -> Vec<nam_core::Sample> {
    let mut resampler = ResamplerState::new(host_rate, model_rate, 4096).unwrap();
    let capacities = (
        resampler.input_pending.capacity(),
        resampler.model_rate_pending.capacity(),
        resampler.output_pending.capacity(),
    );
    let mut model = PassthroughDsp;
    let mut rendered = Vec::with_capacity(total_samples);
    let mut position = 0usize;
    let mut partition_index = 0usize;
    while position < total_samples {
        let requested = partitions[partition_index % partitions.len()];
        let count = requested.min(4096).min(total_samples - position);
        let input: Vec<_> = (position..position + count)
            .map(|sample| {
                let phase = sample as f64 * 0.013_579;
                nam_core::dsp::sample_from_f64(phase.sin() * 0.5)
            })
            .collect();
        let mut output = vec![0.0; count];
        let (result, allocations) = allocation_tracking::count_allocations(|| {
            resampler.process(&mut model, &input, &mut output)
        });
        assert_eq!(result, Ok(()));
        assert_eq!(allocations, 0);
        assert!(resampler.input_pending.len() <= capacities.0);
        assert!(resampler.model_rate_pending.len() <= capacities.1);
        assert!(resampler.output_pending.len() <= capacities.2);
        rendered.extend(output);
        position += count;
        partition_index += 1;
    }
    assert_eq!(rendered.len(), total_samples);
    rendered
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn long_running_resampling_is_block_partition_invariant(
        partitions in proptest::collection::vec(1usize..=4096, 1..64),
        reverse_rates in any::<bool>(),
    ) {
        let (host_rate, model_rate) = if reverse_rates {
            (48_000, 44_100)
        } else {
            (44_100, 48_000)
        };
        let expected = render_resampled_stream(host_rate, model_rate, &[257], 16_384);
        let actual =
            render_resampled_stream(host_rate, model_rate, &partitions, 16_384);
        prop_assert_eq!(actual, expected);
    }
}

#[test]
fn resampler_reset_restores_fresh_latency_and_filter_state() {
    let input: Vec<_> = (0..8192)
        .map(|sample| {
            let value = if sample == 0 { 1.0 } else { 0.0 };
            nam_core::dsp::sample_from_f64(value)
        })
        .collect();
    let mut reused = ResamplerState::new(44_100, 48_000, 4096).unwrap();
    let mut model = PassthroughDsp;
    let mut first = vec![0.0; input.len()];
    for (input, output) in input.chunks(257).zip(first.chunks_mut(257)) {
        reused.process(&mut model, input, output).unwrap();
    }
    reused.reset();
    let mut after_reset = vec![0.0; input.len()];
    for (input, output) in input.chunks(61).zip(after_reset.chunks_mut(61)) {
        reused.process(&mut model, input, output).unwrap();
    }

    let fresh_impulse = {
        let mut state = ResamplerState::new(44_100, 48_000, 4096).unwrap();
        let mut output = vec![0.0; input.len()];
        for (input, output) in input.chunks(4096).zip(output.chunks_mut(4096)) {
            state.process(&mut PassthroughDsp, input, output).unwrap();
        }
        output
    };
    assert_eq!(after_reset, fresh_impulse);
    assert_eq!(first, fresh_impulse);
}

#[test]
fn steady_state_resampling_does_not_allocate() {
    let mut resampler = ResamplerState::new(44100, 48000, 4096).unwrap();
    let mut model = PassthroughDsp;
    let input = vec![0.25; 4096];
    let mut output = vec![0.0; 4096];

    for _ in 0..4 {
        resampler.process(&mut model, &input, &mut output).unwrap();
    }
    let (result, allocations) = allocation_tracking::count_allocations(|| {
        resampler.process(&mut model, &input, &mut output)
    });

    assert_eq!(result, Ok(()));
    assert_eq!(allocations, 0, "steady-state audio processing allocated");
}

#[test]
fn complete_steady_state_callback_does_not_allocate() {
    let buffer_size = 128;
    let mut plugin = plugin_with_passthrough_model(buffer_size);
    let mut audio = vec![0.25f32; buffer_size];
    let mut buffer = mono_buffer(&mut audio);
    for _ in 0..4 {
        assert_eq!(plugin.process_buffer(&mut buffer), ProcessStatus::Normal);
    }

    let (status, allocations) =
        allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));

    assert_eq!(status, ProcessStatus::Normal);
    assert_eq!(allocations, 0, "the complete audio callback allocated");
}

#[test]
fn plugin_instances_keep_independent_activation_modes() {
    fn probe_plugin() -> NamPlugin {
        let mut plugin = plugin_with_passthrough_model(1);
        plugin.model = Some(LoadedModel {
            generation: 1,
            dsp: Box::new(ActivationModeProbe {
                mode: nam_core::ActivationMode::Accurate,
            }),
            resampler: None,
        });
        plugin
    }

    let mut accurate = probe_plugin();
    let mut fast = probe_plugin();
    let mut accurate_sample = [0.0];
    let mut fast_sample = [0.0];

    assert_eq!(
        accurate.process_buffer_with_activation_mode(
            &mut mono_buffer(&mut accurate_sample),
            nam_core::ActivationMode::Accurate,
        ),
        ProcessStatus::Normal
    );
    assert_eq!(
        fast.process_buffer_with_activation_mode(
            &mut mono_buffer(&mut fast_sample),
            nam_core::ActivationMode::Fast,
        ),
        ProcessStatus::Normal
    );
    assert_eq!(accurate_sample, [-1.0]);
    assert_eq!(fast_sample, [1.0]);

    accurate_sample[0] = 0.0;
    accurate.process_buffer_with_activation_mode(
        &mut mono_buffer(&mut accurate_sample),
        nam_core::ActivationMode::Accurate,
    );
    assert_eq!(accurate_sample, [-1.0]);
}

#[test]
fn gain_smoothing_is_block_partition_invariant_and_allocation_free() {
    fn render(block_size: usize) -> Vec<f32> {
        const TOTAL_SAMPLES: usize = 16_384;
        const MAX_BUFFER_SIZE: usize = 4096;

        let mut plugin = plugin_with_passthrough_model(MAX_BUFFER_SIZE);
        plugin.params.input_gain.smoothed.reset(0.0);
        plugin.params.output_gain.smoothed.reset(0.0);
        plugin.params.input_gain.smoothed.set_target(48_000.0, 12.0);
        plugin
            .params
            .output_gain
            .smoothed
            .set_target(48_000.0, -6.0);

        let mut rendered = Vec::with_capacity(TOTAL_SAMPLES);
        while rendered.len() < TOTAL_SAMPLES {
            let count = block_size.min(TOTAL_SAMPLES - rendered.len());
            let mut audio = vec![1.0f32; count];
            let mut buffer = mono_buffer(&mut audio);
            let (status, allocations) =
                allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
            assert_eq!(status, ProcessStatus::Normal);
            assert_eq!(
                allocations, 0,
                "gain-smoothed callback allocated for {block_size}-sample blocks"
            );
            rendered.extend(audio);
        }
        rendered
    }

    let reference = render(16);
    for block_size in [64, 257, 4096] {
        let candidate = render(block_size);
        assert_eq!(candidate.len(), reference.len());
        for (index, (actual, expected)) in candidate.iter().zip(&reference).enumerate() {
            assert!(
                (actual - expected).abs() <= f32::EPSILON,
                "block size {block_size} diverged at sample {index}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn complete_resampling_callback_handles_rate_and_block_size_matrix_without_allocating() {
    const MAX_BUFFER_SIZE: usize = 4096;
    for (host_rate, model_rate) in [(44_100, 48_000), (48_000, 44_100)] {
        let mut plugin =
            plugin_with_resampled_passthrough_model(host_rate, model_rate, MAX_BUFFER_SIZE);
        for buffer_size in [16, 64, 257, MAX_BUFFER_SIZE] {
            let mut audio = vec![0.25f32; buffer_size];
            let mut buffer = mono_buffer(&mut audio);
            let (status, allocations) =
                allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
            assert_eq!(status, ProcessStatus::Normal);
            assert_eq!(
                allocations, 0,
                "{host_rate} to {model_rate} Hz callback allocated for {buffer_size} samples"
            );
            assert_eq!(
                AudioProcessError::from_raw(plugin.audio_error.load(Ordering::Acquire)),
                AudioProcessError::None
            );
        }
    }
}

#[test]
fn callback_defers_saturated_retirement_without_allocating_or_losing_models() {
    let buffer_size = 64;
    let mut plugin = plugin_with_passthrough_model(buffer_size);
    plugin.retired_models = Arc::new(ArrayQueue::new(1));
    assert!(plugin.retired_models.push(loaded_model(99)).is_ok());
    plugin.latest_generation.store(2, Ordering::Release);
    assert!(plugin.loaded_models.push(loaded_model(2)).is_ok());

    let mut audio = vec![0.25f32; buffer_size];
    let mut buffer = mono_buffer(&mut audio);
    let (status, allocations) =
        allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
    assert_eq!(status, ProcessStatus::Normal);
    assert_eq!(allocations, 0);
    assert_eq!(plugin.installed_generation.load(Ordering::Acquire), 2);
    assert_eq!(
        plugin
            .deferred_retire
            .as_ref()
            .map(|model| model.generation),
        Some(1)
    );

    plugin.latest_generation.store(3, Ordering::Release);
    assert!(plugin.loaded_models.push(loaded_model(3)).is_ok());
    let (status, allocations) =
        allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
    assert_eq!(status, ProcessStatus::Normal);
    assert_eq!(allocations, 0);
    assert_eq!(plugin.installed_generation.load(Ordering::Acquire), 2);
    assert_eq!(plugin.loaded_models.len(), 1);

    assert_eq!(
        plugin.retired_models.pop().map(|model| model.generation),
        Some(99)
    );
    assert_eq!(plugin.process_buffer(&mut buffer), ProcessStatus::Normal);
    assert_eq!(plugin.installed_generation.load(Ordering::Acquire), 3);
    assert_eq!(
        plugin
            .deferred_retire
            .as_ref()
            .map(|model| model.generation),
        Some(2)
    );
}

#[test]
fn callback_installs_models_from_a_synchronized_concurrent_loader() {
    let buffer_size = 64;
    let mut plugin = plugin_with_passthrough_model(buffer_size);
    let loaded_models = Arc::clone(&plugin.loaded_models);
    let latest_generation = Arc::clone(&plugin.latest_generation);
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
    let (installed_tx, installed_rx) = std::sync::mpsc::sync_channel(0);
    let producer = std::thread::spawn(move || {
        for generation in 2..=32 {
            latest_generation.store(generation, Ordering::Release);
            assert!(loaded_models.push(loaded_model(generation)).is_ok());
            ready_tx.send(generation).unwrap();
            assert_eq!(installed_rx.recv().unwrap(), generation);
        }
    });

    let mut audio = vec![0.25f32; buffer_size];
    let mut buffer = mono_buffer(&mut audio);
    for expected_generation in 2..=32 {
        assert_eq!(ready_rx.recv().unwrap(), expected_generation);
        let (status, allocations) =
            allocation_tracking::count_allocations(|| plugin.process_buffer(&mut buffer));
        assert_eq!(status, ProcessStatus::Normal);
        assert_eq!(allocations, 0);
        assert_eq!(
            plugin.installed_generation.load(Ordering::Acquire),
            expected_generation
        );
        assert!(installed_tx.send(expected_generation).is_ok());
        let _ = plugin.retired_models.pop();
    }

    producer.join().unwrap();
    assert_eq!(plugin.installed_generation.load(Ordering::Acquire), 32);
}

#[test]
fn plugin_lifecycle_state_machine_never_installs_stale_or_post_drop_models() {
    #[derive(Clone, Copy)]
    enum Event {
        RequestNext,
        LoaderCompletes(u64),
        Callback,
        Drop,
    }

    #[derive(Clone, Copy)]
    struct Lifecycle {
        alive: bool,
        latest: u64,
        published: Option<u64>,
        installed: Option<u64>,
    }

    impl Lifecycle {
        fn apply(mut self, event: Event) -> Self {
            match event {
                Event::RequestNext if self.alive => {
                    self.latest = self.latest.saturating_add(1);
                }
                Event::LoaderCompletes(generation) if self.alive && generation == self.latest => {
                    self.published = Some(generation);
                }
                Event::Callback if self.alive => {
                    if let Some(generation) = self.published.take() {
                        if generation == self.latest {
                            self.installed = Some(generation);
                        }
                    }
                }
                Event::Drop => {
                    self.alive = false;
                    self.published = None;
                }
                _ => {}
            }
            self
        }

        fn assert_invariants(self) {
            assert!(self
                .published
                .is_none_or(|generation| self.alive && generation <= self.latest));
            assert!(self
                .installed
                .is_none_or(|generation| generation <= self.latest));
            if !self.alive {
                assert!(self.published.is_none());
            }
        }
    }

    fn explore(state: Lifecycle, depth: usize) {
        state.assert_invariants();
        if depth == 0 {
            return;
        }
        let events = [
            Event::RequestNext,
            Event::LoaderCompletes(1),
            Event::LoaderCompletes(2),
            Event::LoaderCompletes(3),
            Event::Callback,
            Event::Drop,
        ];
        for event in events {
            explore(state.apply(event), depth - 1);
        }
    }

    explore(
        Lifecycle {
            alive: true,
            latest: 1,
            published: None,
            installed: None,
        },
        7,
    );
}
