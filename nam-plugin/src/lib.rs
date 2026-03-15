use nih_plug::prelude::*;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

/// The NAM audio plugin.
struct NamPlugin {
    params: Arc<NamParams>,
    model: Option<Box<dyn nam_core::Dsp>>,
    /// Pre-allocated buffers to avoid allocations in process().
    input_buf: Vec<nam_core::Sample>,
    output_buf: Vec<nam_core::Sample>,
    sample_rate: f64,
}

#[derive(Params)]
struct NamParams {
    /// Input gain in dB.
    #[id = "in_gain"]
    pub input_gain: FloatParam,

    /// Output gain in dB.
    #[id = "out_gain"]
    pub output_gain: FloatParam,

    /// Persisted model file path (restored when the DAW reloads the session).
    #[persist = "model_path"]
    pub model_path: Mutex<String>,
}

impl Default for NamPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(NamParams::default()),
            model: None,
            input_buf: Vec::new(),
            output_buf: Vec::new(),
            sample_rate: 48000.0,
        }
    }
}

impl Default for NamParams {
    fn default() -> Self {
        Self {
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

            model_path: Mutex::new(String::new()),
        }
    }
}

impl NamPlugin {
    /// Load a model from the given path. Called from initialize() (off the audio thread).
    fn load_model(&mut self, path: &str) {
        if path.is_empty() {
            return;
        }
        nih_log!("Loading model from: {}", path);
        match nam_core::get_dsp(std::path::Path::new(path)) {
            Ok(mut dsp) => {
                dsp.reset(self.sample_rate, self.input_buf.len().max(4096));
                dsp.prewarm();
                self.model = Some(dsp);
                nih_log!("Model loaded successfully");
            }
            Err(e) => {
                nih_error!("Failed to load model: {}", e);
            }
        }
    }
}

impl Plugin for NamPlugin {
    const NAME: &'static str = "NAM";
    const VENDOR: &'static str = "nam-rs";
    const URL: &'static str = "https://github.com/sdatkinson/NeuralAmpModelerCore";
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
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate as f64;
        let max_buf = buffer_config.max_buffer_size as usize;

        // Pre-allocate buffers
        self.input_buf = vec![0.0; max_buf];
        self.output_buf = vec![0.0; max_buf];

        // Load persisted model path (initialize runs off audio thread, safe to do I/O)
        let path = self.params.model_path.lock().unwrap().clone();
        if !path.is_empty() {
            self.load_model(&path);
        }

        true
    }

    fn reset(&mut self) {
        if let Some(ref mut model) = self.model {
            model.reset(self.sample_rate, self.input_buf.len());
            model.prewarm();
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

        let model = match self.model {
            Some(ref mut m) => m,
            None => return ProcessStatus::Normal,
        };

        // Get raw channel slice (mono: 1 channel)
        let channel_data = buffer.as_slice();
        let channel = &mut channel_data[0];

        // Copy input with gain applied
        let in_gain = util::db_to_gain_fast(self.params.input_gain.smoothed.next());
        for i in 0..num_samples {
            self.input_buf[i] = (channel[i] * in_gain) as nam_core::Sample;
        }

        // Process through model
        model.process(
            &self.input_buf[..num_samples],
            &mut self.output_buf[..num_samples],
        );

        // Copy output with gain applied
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
