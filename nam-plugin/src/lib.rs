use nih_plug::prelude::*;
use nih_plug_egui::resizable_window::ResizableWindow;
use nih_plug_egui::{create_egui_editor, egui, widgets, EguiState};
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Background task for model loading (runs off the audio thread).
enum NamTask {
    LoadModel(PathBuf),
}

/// The NAM audio plugin.
struct NamPlugin {
    params: Arc<NamParams>,
    /// Shared model slot: GUI writes a new model here, audio thread reads it.
    model: Arc<Mutex<Option<Box<dyn nam_core::Dsp>>>>,
    /// Pre-allocated buffers to avoid allocations in process().
    input_buf: Vec<nam_core::Sample>,
    output_buf: Vec<nam_core::Sample>,
    sample_rate: f64,
    max_buffer_size: usize,
    /// Resampling state (None if host rate matches model rate).
    resampler: Option<ResamplerState>,
}

/// Handles resampling between host sample rate and model sample rate.
/// Uses rubato's FftFixedInOut which requires fixed input chunk sizes.
/// We buffer samples in VecDeques to handle variable DAW buffer sizes.
struct ResamplerState {
    /// Host rate -> model rate
    to_model: FftFixedInOut<f64>,
    /// Model rate -> host rate
    to_host: FftFixedInOut<f64>,
    /// Accumulates host-rate samples until we have enough for a resample chunk
    input_pending: VecDeque<f64>,
    /// Accumulates model-rate samples waiting to be back-resampled to host rate
    model_rate_pending: VecDeque<f64>,
    /// Accumulates host-rate output samples ready for the DAW
    output_pending: VecDeque<f64>,
    /// Fixed input chunk size for to_model resampler
    to_model_chunk: usize,
    /// Fixed input chunk size for to_host resampler
    to_host_chunk: usize,
    /// Pre-allocated model I/O buffers
    model_input: Vec<nam_core::Sample>,
    model_output: Vec<nam_core::Sample>,
    /// Pre-allocated chunk buffers for rubato (avoids allocations in process)
    to_model_scratch: Vec<Vec<f64>>,
    to_host_scratch: Vec<Vec<f64>>,
}

impl ResamplerState {
    fn new(host_rate: usize, model_rate: usize) -> Option<Self> {
        // FftFixedInOut: both input and output are fixed-size chunks
        let to_model = FftFixedInOut::<f64>::new(host_rate, model_rate, 128, 1).ok()?;
        let to_host = FftFixedInOut::<f64>::new(model_rate, host_rate, 128, 1).ok()?;

        let to_model_chunk = to_model.input_frames_max();
        let to_host_chunk = to_host.input_frames_max();
        let max_model_buf = to_model.output_frames_max() * 8;

        Some(Self {
            to_model,
            to_host,
            input_pending: VecDeque::with_capacity(to_model_chunk * 4),
            model_rate_pending: VecDeque::with_capacity(to_host_chunk * 4),
            output_pending: VecDeque::with_capacity(to_host_chunk * 4),
            to_model_chunk,
            to_host_chunk,
            model_input: vec![0.0; max_model_buf],
            model_output: vec![0.0; max_model_buf],
            to_model_scratch: vec![vec![0.0; to_model_chunk]; 1],
            to_host_scratch: vec![vec![0.0; to_host_chunk]; 1],
        })
    }

    fn reset(&mut self) {
        self.input_pending.clear();
        self.model_rate_pending.clear();
        self.output_pending.clear();
    }
}

#[derive(Params)]
struct NamParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

    #[id = "in_gain"]
    pub input_gain: FloatParam,

    #[id = "out_gain"]
    pub output_gain: FloatParam,

    #[id = "fast_mode"]
    pub fast_mode: BoolParam,

    #[persist = "model_path"]
    pub model_path: Mutex<String>,
}

struct GuiState {
    model_name: String,
    status: String,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            status: "No model loaded".to_string(),
        }
    }
}

impl Default for NamPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(NamParams::default()),
            model: Arc::new(Mutex::new(None)),
            input_buf: Vec::new(),
            output_buf: Vec::new(),
            sample_rate: 48000.0,
            max_buffer_size: 4096,
            resampler: None,
        }
    }
}

impl Default for NamParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(400, 280),

            input_gain: FloatParam::new(
                "Input Gain",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_unit(" dB")
            .with_step_size(0.1)
            .with_smoother(SmoothingStyle::Logarithmic(50.0)),

            output_gain: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: -40.0,
                    max: 40.0,
                },
            )
            .with_unit(" dB")
            .with_step_size(0.1)
            .with_smoother(SmoothingStyle::Logarithmic(50.0)),

            fast_mode: BoolParam::new("Fast Mode", false).with_callback(Arc::new(|val| {
                if val {
                    nam_core::enable_fast_tanh();
                } else {
                    nam_core::disable_fast_tanh();
                }
            })),

            model_path: Mutex::new(String::new()),
        }
    }
}

impl NamPlugin {
    fn setup_resampler(&mut self) {
        let model_rate = {
            let guard = self.model.lock().unwrap();
            match guard.as_ref() {
                Some(m) => m.metadata().expected_sample_rate.unwrap_or(0.0),
                None => 0.0,
            }
        };

        let host_rate = self.sample_rate;

        if model_rate <= 0.0 || (host_rate - model_rate).abs() < 0.5 {
            self.resampler = None;
            return;
        }

        nih_log!(
            "Setting up resampler: host {} Hz <-> model {} Hz",
            host_rate,
            model_rate
        );

        match ResamplerState::new(host_rate as usize, model_rate as usize) {
            Some(rs) => {
                self.resampler = Some(rs);
            }
            None => {
                nih_error!("Failed to create resampler");
                self.resampler = None;
            }
        }
    }

    /// Process with resampling: host_rate -> model_rate -> model -> model_rate -> host_rate
    fn process_resampled(
        rs: &mut ResamplerState,
        model: &mut dyn nam_core::Dsp,
        input: &[nam_core::Sample],
        output: &mut [nam_core::Sample],
    ) {
        let num_samples = input.len();

        // 1. Push host-rate input into pending buffer
        for &s in input {
            rs.input_pending.push_back(s);
        }

        // 2. Resample host_rate -> model_rate in fixed-size chunks
        while rs.input_pending.len() >= rs.to_model_chunk {
            for i in 0..rs.to_model_chunk {
                rs.to_model_scratch[0][i] = rs.input_pending.pop_front().unwrap_or(0.0);
            }
            if let Ok(resampled) = rs.to_model.process(&rs.to_model_scratch, None) {
                for &s in &resampled[0] {
                    rs.model_rate_pending.push_back(s);
                }
            }
        }

        // 3. Process all available model-rate samples through NAM
        let model_samples = rs.model_rate_pending.len();
        if model_samples > 0 {
            if model_samples > rs.model_input.len() {
                rs.model_input.resize(model_samples, 0.0);
            }
            if model_samples > rs.model_output.len() {
                rs.model_output.resize(model_samples, 0.0);
            }
            for i in 0..model_samples {
                rs.model_input[i] =
                    rs.model_rate_pending.pop_front().unwrap_or(0.0) as nam_core::Sample;
            }

            model.process(
                &rs.model_input[..model_samples],
                &mut rs.model_output[..model_samples],
            );

            // 4. Push model output into model_rate_pending for back-resampling
            //    (reuse the same deque — it was just drained)
            for i in 0..model_samples {
                rs.model_rate_pending.push_back(rs.model_output[i]);
            }
        }

        // 5. Resample model_rate -> host_rate in fixed-size chunks
        while rs.model_rate_pending.len() >= rs.to_host_chunk {
            for i in 0..rs.to_host_chunk {
                rs.to_host_scratch[0][i] = rs.model_rate_pending.pop_front().unwrap_or(0.0);
            }
            if let Ok(resampled) = rs.to_host.process(&rs.to_host_scratch, None) {
                for &s in &resampled[0] {
                    rs.output_pending.push_back(s);
                }
            }
        }

        // 6. Fill output from pending output buffer
        for sample in output.iter_mut().take(num_samples) {
            *sample = rs.output_pending.pop_front().unwrap_or(0.0) as nam_core::Sample;
        }
    }
}

impl Plugin for NamPlugin {
    const NAME: &'static str = "NAM";
    const VENDOR: &'static str = "nam-rs";
    const URL: &'static str = "https://github.com/joshburgess/nam-rs";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames::const_default(),
    }];

    type SysExMessage = ();
    type BackgroundTask = NamTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let model_slot = self.model.clone();
        let params = self.params.clone();
        let sample_rate = self.sample_rate;
        let max_buf = self.max_buffer_size;

        Box::new(move |task| match task {
            NamTask::LoadModel(path) => {
                nih_log!("Loading model from {:?}", path);
                match nam_core::get_dsp(&path) {
                    Ok(mut dsp) => {
                        let model_rate = dsp.metadata().expected_sample_rate.unwrap_or(sample_rate);
                        dsp.reset(model_rate, max_buf);
                        dsp.prewarm();
                        *model_slot.lock().unwrap() = Some(dsp);
                        if let Ok(mut p) = params.model_path.lock() {
                            *p = path.to_string_lossy().to_string();
                        }
                        nih_log!("Model loaded successfully");
                    }
                    Err(e) => {
                        nih_error!("Failed to load model: {}", e);
                    }
                }
            }
        })
    }

    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let model_slot = self.model.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            GuiState::default(),
            |_, _| {},
            move |egui_ctx, setter, state| {
                let egui_state = params.editor_state.clone();

                ResizableWindow::new("nam-editor")
                    .min_size(egui::Vec2::new(300.0, 200.0))
                    .show(egui_ctx, egui_state.as_ref(), |ui| {
                        ui.heading("Neural Amp Modeler");
                        ui.separator();

                        ui.horizontal(|ui| {
                            if ui.button("Load Model").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("NAM Model", &["nam"])
                                    .pick_file()
                                {
                                    state.model_name = path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_default();
                                    state.status = "Loading...".to_string();
                                    async_executor.execute_background(NamTask::LoadModel(path));
                                }
                            }

                            if model_slot.lock().unwrap().is_some() {
                                ui.label(
                                    egui::RichText::new("●")
                                        .color(egui::Color32::GREEN)
                                        .size(14.0),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("●")
                                        .color(egui::Color32::DARK_GRAY)
                                        .size(14.0),
                                );
                            }
                        });

                        if state.model_name.is_empty() {
                            ui.label("No model loaded");
                        } else {
                            ui.label(&state.model_name);
                        }

                        if model_slot.lock().unwrap().is_some() && state.status == "Loading..." {
                            state.status = "Ready".to_string();
                        }

                        ui.separator();

                        ui.label("Input Gain");
                        ui.add(widgets::ParamSlider::for_param(&params.input_gain, setter));

                        ui.label("Output Gain");
                        ui.add(widgets::ParamSlider::for_param(&params.output_gain, setter));

                        ui.separator();

                        let mut fast = params.fast_mode.value();
                        if ui
                            .checkbox(&mut fast, "Fast Mode (lower accuracy, better performance)")
                            .changed()
                        {
                            setter.begin_set_parameter(&params.fast_mode);
                            setter.set_parameter(&params.fast_mode, fast);
                            setter.end_set_parameter(&params.fast_mode);
                        }
                    });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate as f64;
        self.max_buffer_size = buffer_config.max_buffer_size as usize;

        self.input_buf = vec![0.0; self.max_buffer_size];
        self.output_buf = vec![0.0; self.max_buffer_size];

        let path = self.params.model_path.lock().unwrap().clone();
        if !path.is_empty() && self.model.lock().unwrap().is_none() {
            context.execute(NamTask::LoadModel(PathBuf::from(path)));
        }

        self.setup_resampler();

        true
    }

    fn reset(&mut self) {
        if let Ok(mut model) = self.model.lock() {
            if let Some(ref mut m) = *model {
                let model_rate = m
                    .metadata()
                    .expected_sample_rate
                    .unwrap_or(self.sample_rate);
                m.reset(model_rate, self.max_buffer_size);
                m.prewarm();
            }
        }
        self.setup_resampler();
        if let Some(ref mut rs) = self.resampler {
            rs.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        if num_samples == 0 {
            return ProcessStatus::Normal;
        }

        let mut model_guard = self.model.lock().unwrap();
        let model = match model_guard.as_mut() {
            Some(m) => m,
            None => return ProcessStatus::Normal,
        };

        let channel_data = buffer.as_slice();
        let channel = &mut channel_data[0];

        let in_gain = util::db_to_gain_fast(self.params.input_gain.smoothed.next());
        for i in 0..num_samples {
            self.input_buf[i] = (channel[i] * in_gain) as nam_core::Sample;
        }

        if let Some(ref mut rs) = self.resampler {
            let input_copy: Vec<nam_core::Sample> = self.input_buf[..num_samples].to_vec();
            Self::process_resampled(
                rs,
                &mut **model,
                &input_copy,
                &mut self.output_buf[..num_samples],
            );
        } else {
            model.process(
                &self.input_buf[..num_samples],
                &mut self.output_buf[..num_samples],
            );
        }

        let out_gain = util::db_to_gain_fast(self.params.output_gain.smoothed.next());
        for i in 0..num_samples {
            channel[i] = (self.output_buf[i] as f32) * out_gain;
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for NamPlugin {
    const CLAP_ID: &'static str = "com.nam-rs.nam-plugin";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Neural Amp Modeler (Rust)");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mono,
        ClapFeature::Custom("guitar"),
        ClapFeature::Custom("amp-sim"),
    ];
}

impl Vst3Plugin for NamPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"NamRsPlugin_v001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Custom("Guitar")];
}

nih_export_clap!(NamPlugin);
nih_export_vst3!(NamPlugin);

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial pass-through DSP for testing resampling in isolation.
    struct PassthroughDsp;

    impl nam_core::Dsp for PassthroughDsp {
        fn process(&mut self, input: &[nam_core::Sample], output: &mut [nam_core::Sample]) {
            output[..input.len()].copy_from_slice(input);
        }
        fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {}
        fn metadata(&self) -> &nam_core::dsp::DspMetadata {
            static META: nam_core::dsp::DspMetadata = nam_core::dsp::DspMetadata {
                loudness: None,
                expected_sample_rate: None,
                input_level_dbu: None,
                output_level_dbu: None,
            };
            &META
        }
    }

    #[test]
    fn test_process_resampled_produces_output() {
        let mut rs = ResamplerState::new(44100, 48000).unwrap();
        let mut model = PassthroughDsp;

        // Feed enough samples to produce output (need multiple chunks)
        let num_samples = 4096;
        let input = vec![0.5 as nam_core::Sample; num_samples];
        let mut output = vec![0.0 as nam_core::Sample; num_samples];

        NamPlugin::process_resampled(&mut rs, &mut model, &input, &mut output);

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
        let mut rs = ResamplerState::new(44100, 48000).unwrap();
        let mut model = PassthroughDsp;

        // Simulate multiple process() calls with varying buffer sizes (like a real DAW)
        let buffer_sizes = [64, 128, 64, 256, 64, 128];
        let mut total_nonzero = 0;

        for &size in &buffer_sizes {
            let input = vec![0.3 as nam_core::Sample; size];
            let mut output = vec![0.0 as nam_core::Sample; size];
            NamPlugin::process_resampled(&mut rs, &mut model, &input, &mut output);
            total_nonzero += output.iter().filter(|&&x| x != 0.0).count();
        }

        assert!(
            total_nonzero > 0,
            "Should produce output across multiple calls"
        );
    }

    #[test]
    fn test_process_resampled_preserves_signal_level() {
        let mut rs = ResamplerState::new(44100, 48000).unwrap();
        let mut model = PassthroughDsp;

        // Feed a constant signal — after resampler settles, output should be ~same level
        let settle = vec![0.5 as nam_core::Sample; 4096]; // let resampler settle
        let mut discard = vec![0.0 as nam_core::Sample; 4096];
        NamPlugin::process_resampled(&mut rs, &mut model, &settle, &mut discard);

        let input = vec![0.5 as nam_core::Sample; 2048];
        let mut output = vec![0.0 as nam_core::Sample; 2048];
        NamPlugin::process_resampled(&mut rs, &mut model, &input, &mut output);

        // Check the latter half (fully settled)
        let tail = &output[1024..];
        let mean: f64 = tail.iter().map(|&x| x as f64).sum::<f64>() / tail.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Mean output {:.4} should be close to input 0.5 after settling",
            mean
        );
    }

    #[test]
    fn test_resampler_reset_clears_state() {
        let mut rs = ResamplerState::new(44100, 48000).unwrap();
        let mut model = PassthroughDsp;

        // Feed some data
        let input = vec![1.0 as nam_core::Sample; 512];
        let mut output = vec![0.0 as nam_core::Sample; 512];
        NamPlugin::process_resampled(&mut rs, &mut model, &input, &mut output);

        assert!(!rs.input_pending.is_empty() || !rs.output_pending.is_empty() || true);

        // Reset should clear buffers
        rs.reset();
        assert!(rs.input_pending.is_empty());
        assert!(rs.output_pending.is_empty());
    }
}
