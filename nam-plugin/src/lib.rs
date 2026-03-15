use nih_plug::prelude::*;
use nih_plug_egui::resizable_window::ResizableWindow;
use nih_plug_egui::{create_egui_editor, egui, widgets, EguiState};
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
}

#[derive(Params)]
struct NamParams {
    /// Persisted editor window state.
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

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

/// GUI-only state (not persisted with plugin state).
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
        }
    }
}

impl Default for NamParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(400, 250),

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
                        dsp.reset(sample_rate, max_buf);
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

                        // Model loading
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

                        // Model name display
                        if state.model_name.is_empty() {
                            ui.label("No model loaded");
                        } else {
                            ui.label(&state.model_name);
                        }

                        // Update status when model finishes loading
                        if model_slot.lock().unwrap().is_some() && state.status == "Loading..." {
                            state.status = "Ready".to_string();
                        }

                        ui.separator();

                        // Gain controls
                        ui.label("Input Gain");
                        ui.add(widgets::ParamSlider::for_param(&params.input_gain, setter));

                        ui.label("Output Gain");
                        ui.add(widgets::ParamSlider::for_param(&params.output_gain, setter));
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

        // Pre-allocate buffers
        self.input_buf = vec![0.0; self.max_buffer_size];
        self.output_buf = vec![0.0; self.max_buffer_size];

        // Load persisted model path
        let path = self.params.model_path.lock().unwrap().clone();
        if !path.is_empty() && self.model.lock().unwrap().is_none() {
            context.execute(NamTask::LoadModel(PathBuf::from(path)));
        }

        true
    }

    fn reset(&mut self) {
        if let Ok(mut model) = self.model.lock() {
            if let Some(ref mut m) = *model {
                m.reset(self.sample_rate, self.max_buffer_size);
                m.prewarm();
            }
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

        model.process(
            &self.input_buf[..num_samples],
            &mut self.output_buf[..num_samples],
        );

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
