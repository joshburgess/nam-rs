use crossbeam_queue::ArrayQueue;
use nih_plug::prelude::*;
use nih_plug_egui::resizable_window::ResizableWindow;
use nih_plug_egui::{create_egui_editor, egui, widgets, EguiState};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

mod resampler;

use resampler::{AudioProcessError, ResamplerState};

#[cfg(test)]
mod allocation_tracking {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static TRACKING: Cell<bool> = const { Cell::new(false) };
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) struct TrackingAllocator;

    // SAFETY: Every allocation operation is forwarded to `System` with unchanged arguments.
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            // SAFETY: This allocator forwards the unchanged layout to the system allocator.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            // SAFETY: This allocator forwards the unchanged layout to the system allocator.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: The pointer and layout came from the system allocator above.
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record_allocation();
            // SAFETY: The pointer and layout came from the system allocator above.
            unsafe { System.realloc(pointer, layout, new_size) }
        }
    }

    fn record_allocation() {
        TRACKING.with(|tracking| {
            if tracking.get() {
                ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
            }
        });
    }

    pub(super) fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
        ALLOCATIONS.with(|allocations| allocations.set(0));
        TRACKING.with(|tracking| tracking.set(true));
        let result = operation();
        TRACKING.with(|tracking| tracking.set(false));
        let count = ALLOCATIONS.with(Cell::get);
        (result, count)
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: allocation_tracking::TrackingAllocator =
    allocation_tracking::TrackingAllocator;

enum NamTask {
    LoadModel { generation: u64, path: PathBuf },
}

struct LoadedModel {
    generation: u64,
    dsp: Box<dyn nam_core::Dsp>,
    resampler: Option<ResamplerState>,
}

struct NamPlugin {
    params: Arc<NamParams>,
    model: Option<LoadedModel>,
    deferred_retire: Option<LoadedModel>,
    loaded_models: Arc<ArrayQueue<LoadedModel>>,
    retired_models: Arc<ArrayQueue<LoadedModel>>,
    latest_generation: Arc<AtomicU64>,
    installed_generation: Arc<AtomicU64>,
    load_status: Arc<Mutex<ModelLoadStatus>>,
    audio_error: Arc<AtomicU8>,
    plugin_alive: Arc<AtomicBool>,
    input_buf: Vec<nam_core::Sample>,
    output_buf: Vec<nam_core::Sample>,
    sample_rate: f64,
    max_buffer_size: usize,
}

#[derive(Clone, Debug)]
enum ModelLoadStatus {
    Empty,
    Loading {
        generation: u64,
        path: PathBuf,
    },
    Ready {
        generation: u64,
        path: PathBuf,
    },
    Failed {
        generation: u64,
        path: PathBuf,
        message: String,
    },
}

fn mark_load_failed(
    status: &Mutex<ModelLoadStatus>,
    latest_generation: &AtomicU64,
    generation: u64,
    path: PathBuf,
    message: String,
) {
    if latest_generation.load(Ordering::Acquire) != generation {
        return;
    }
    *status
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = ModelLoadStatus::Failed {
        generation,
        path,
        message,
    };
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

struct GuiState;

impl Default for GuiState {
    fn default() -> Self {
        Self
    }
}

impl Default for NamPlugin {
    fn default() -> Self {
        Self::new(true)
    }
}

impl NamPlugin {
    fn new(spawn_model_reaper: bool) -> Self {
        let loaded_models = Arc::new(ArrayQueue::new(1));
        let retired_models = Arc::new(ArrayQueue::new(4));
        if spawn_model_reaper {
            let retired_models_weak = Arc::downgrade(&retired_models);
            let _ = thread::Builder::new()
                .name("nam-model-reaper".to_string())
                .spawn(move || {
                    while let Some(retired_models) = retired_models_weak.upgrade() {
                        while retired_models.pop().is_some() {}
                        thread::park_timeout(Duration::from_millis(10));
                    }
                });
        }

        Self {
            params: Arc::new(NamParams::default()),
            model: None,
            deferred_retire: None,
            loaded_models,
            retired_models,
            latest_generation: Arc::new(AtomicU64::new(0)),
            installed_generation: Arc::new(AtomicU64::new(0)),
            load_status: Arc::new(Mutex::new(ModelLoadStatus::Empty)),
            audio_error: Arc::new(AtomicU8::new(AudioProcessError::None as u8)),
            plugin_alive: Arc::new(AtomicBool::new(true)),
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
            .with_smoother(SmoothingStyle::Linear(50.0)),

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
            .with_smoother(SmoothingStyle::Linear(50.0)),

            fast_mode: BoolParam::new("Fast Mode", false),

            model_path: Mutex::new(String::new()),
        }
    }
}

impl NamPlugin {
    fn flush_deferred_retire(&mut self) -> bool {
        let Some(retired) = self.deferred_retire.take() else {
            return true;
        };
        match self.retired_models.push(retired) {
            Ok(()) => true,
            Err(retired) => {
                self.deferred_retire = Some(retired);
                false
            }
        }
    }

    fn install_pending_model(&mut self) {
        if !self.flush_deferred_retire() {
            return;
        }
        let Some(loaded) = self.loaded_models.pop() else {
            return;
        };

        if loaded.generation != self.latest_generation.load(Ordering::Acquire) {
            self.deferred_retire = Some(loaded);
            self.flush_deferred_retire();
            return;
        }

        let generation = loaded.generation;
        if let Some(retired) = self.model.replace(loaded) {
            self.deferred_retire = Some(retired);
            self.flush_deferred_retire();
        }
        self.audio_error
            .store(AudioProcessError::None as u8, Ordering::Release);
        self.installed_generation
            .store(generation, Ordering::Release);
    }

    fn process_buffer(&mut self, buffer: &mut Buffer) -> ProcessStatus {
        let activation_mode = if self.params.fast_mode.value() {
            nam_core::ActivationMode::Fast
        } else {
            nam_core::ActivationMode::Accurate
        };
        self.process_buffer_with_activation_mode(buffer, activation_mode)
    }

    fn process_buffer_with_activation_mode(
        &mut self,
        buffer: &mut Buffer,
        activation_mode: nam_core::ActivationMode,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        if num_samples == 0 {
            return ProcessStatus::Normal;
        }

        let channel_data = buffer.as_slice();
        if channel_data.len() != 1 || channel_data[0].len() != num_samples {
            for channel in channel_data {
                channel.fill(0.0);
            }
            self.report_audio_error(AudioProcessError::CallbackLayout);
            return ProcessStatus::Normal;
        }
        let channel = &mut channel_data[0];
        if num_samples > self.input_buf.len() || num_samples > self.output_buf.len() {
            channel.fill(0.0);
            self.report_audio_error(AudioProcessError::CallbackCapacity);
            return ProcessStatus::Normal;
        }

        self.install_pending_model();
        let model = match self.model.as_mut() {
            Some(model) => model,
            None => return ProcessStatus::Normal,
        };
        model.dsp.set_activation_mode(activation_mode);

        for (input, &sample) in self.input_buf[..num_samples].iter_mut().zip(channel.iter()) {
            let in_gain = util::db_to_gain_fast(self.params.input_gain.smoothed.next());
            *input = (sample * in_gain) as nam_core::Sample;
        }

        let mut process_error = AudioProcessError::None;
        if let Some(resampler) = model.resampler.as_mut() {
            if let Err(error) = resampler.process(
                &mut *model.dsp,
                &self.input_buf[..num_samples],
                &mut self.output_buf[..num_samples],
            ) {
                resampler.reset();
                self.output_buf[..num_samples].fill(0.0);
                process_error = error;
            }
        } else {
            model.dsp.process(
                &self.input_buf[..num_samples],
                &mut self.output_buf[..num_samples],
            );
        }
        if process_error == AudioProcessError::None
            && mute_non_finite(&mut self.output_buf[..num_samples])
        {
            process_error = AudioProcessError::NonFiniteOutput;
        }
        self.report_audio_error(process_error);

        for (sample, &output) in channel.iter_mut().zip(&self.output_buf[..num_samples]) {
            let out_gain = util::db_to_gain_fast(self.params.output_gain.smoothed.next());
            *sample = nam_core::dsp::sample_to_f32(output) * out_gain;
        }

        ProcessStatus::Normal
    }

    fn report_audio_error(&self, process_error: AudioProcessError) {
        let reported_error = AudioProcessError::from_raw(self.audio_error.load(Ordering::Acquire));
        if process_error == AudioProcessError::NonFiniteOutput
            || reported_error != AudioProcessError::NonFiniteOutput
        {
            self.audio_error
                .store(process_error as u8, Ordering::Release);
        }
    }
}

fn mute_non_finite(samples: &mut [nam_core::Sample]) -> bool {
    let mut found = false;
    for sample in samples {
        if !sample.is_finite() {
            *sample = 0.0;
            found = true;
        }
    }
    found
}

#[cfg(feature = "benchmark-internals")]
pub mod benchmark {
    use super::{LoadedModel, NamPlugin, ProcessStatus, ResamplerState};
    use nih_plug::prelude::Buffer;
    use std::sync::atomic::Ordering;

    struct PassthroughDsp;

    impl nam_core::Dsp for PassthroughDsp {
        fn process(&mut self, input: &[nam_core::Sample], output: &mut [nam_core::Sample]) {
            output[..input.len()].copy_from_slice(input);
        }

        fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {}

        fn metadata(&self) -> &nam_core::dsp::DspMetadata {
            static METADATA: nam_core::dsp::DspMetadata = nam_core::dsp::DspMetadata {
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
            &METADATA
        }
    }

    pub struct CallbackCase {
        plugin: NamPlugin,
        audio: Vec<f32>,
    }

    impl CallbackCase {
        pub fn new(
            host_rate: usize,
            model_rate: usize,
            buffer_size: usize,
        ) -> Result<Self, rubato::ResamplerConstructionError> {
            let resampler = if host_rate == model_rate {
                None
            } else {
                Some(ResamplerState::new(host_rate, model_rate, buffer_size)?)
            };
            let mut plugin = NamPlugin::new(false);
            plugin.model = Some(LoadedModel {
                generation: 1,
                dsp: Box::new(PassthroughDsp),
                resampler,
            });
            plugin.latest_generation.store(1, Ordering::Release);
            plugin.installed_generation.store(1, Ordering::Release);
            plugin.input_buf = vec![0.0; buffer_size];
            plugin.output_buf = vec![0.0; buffer_size];
            plugin.sample_rate = host_rate as f64;
            plugin.max_buffer_size = buffer_size;

            let mut case = Self {
                plugin,
                audio: vec![0.25; buffer_size],
            };
            for _ in 0..4 {
                case.process();
            }
            Ok(case)
        }

        pub fn process(&mut self) {
            let mut buffer = Buffer::default();
            // SAFETY: The temporary buffer cannot outlive `self.audio`, and its
            // only channel contains exactly `self.audio.len()` samples.
            unsafe {
                buffer.set_slices(self.audio.len(), |channels| channels.push(&mut self.audio));
            }
            assert_eq!(
                self.plugin.process_buffer(&mut buffer),
                ProcessStatus::Normal
            );
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
        let loaded_models = self.loaded_models.clone();
        let plugin_alive = self.plugin_alive.clone();
        let latest_generation = self.latest_generation.clone();
        let load_status = self.load_status.clone();
        let params = self.params.clone();
        let sample_rate = self.sample_rate;
        let max_buf = self.max_buffer_size;

        Box::new(move |task| match task {
            NamTask::LoadModel { generation, path } => {
                if latest_generation.load(Ordering::Acquire) != generation {
                    return;
                }
                nih_log!("Loading model from {:?}", path);
                match nam_core::get_dsp(&path) {
                    Ok(mut dsp) => {
                        dsp.set_activation_mode(if params.fast_mode.value() {
                            nam_core::ActivationMode::Fast
                        } else {
                            nam_core::ActivationMode::Accurate
                        });
                        let model_rate = dsp.metadata().expected_sample_rate.unwrap_or(sample_rate);
                        dsp.reset(model_rate, max_buf);
                        dsp.prewarm();
                        let resampler = if (sample_rate - model_rate).abs() < 0.5 {
                            None
                        } else {
                            match ResamplerState::new(
                                sample_rate as usize,
                                model_rate as usize,
                                max_buf,
                            ) {
                                Ok(resampler) => Some(resampler),
                                Err(error) => {
                                    let message = format!(
                                        "Could not resample from {sample_rate:.0} Hz to {model_rate:.0} Hz: {error}"
                                    );
                                    nih_error!("{message}");
                                    mark_load_failed(
                                        &load_status,
                                        &latest_generation,
                                        generation,
                                        path,
                                        message,
                                    );
                                    return;
                                }
                            }
                        };
                        let mut loaded = LoadedModel {
                            generation,
                            dsp,
                            resampler,
                        };
                        loop {
                            if !plugin_alive.load(Ordering::Acquire)
                                || latest_generation.load(Ordering::Acquire) != generation
                            {
                                return;
                            }
                            match loaded_models.push(loaded) {
                                Ok(()) => break,
                                Err(returned) => {
                                    loaded = returned;
                                    if let Some(obsolete) = loaded_models.pop() {
                                        drop(obsolete);
                                    } else {
                                        thread::yield_now();
                                    }
                                }
                            }
                        }
                        nih_log!("Model loaded successfully");
                    }
                    Err(error) => {
                        let message = format!("Failed to load model: {error}");
                        nih_error!("{message}");
                        mark_load_failed(
                            &load_status,
                            &latest_generation,
                            generation,
                            path,
                            message,
                        );
                    }
                }
            }
        })
    }

    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let latest_generation = self.latest_generation.clone();
        let installed_generation = self.installed_generation.clone();
        let load_status = self.load_status.clone();
        let audio_error = self.audio_error.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            GuiState,
            |_, _| {},
            move |egui_ctx, setter, _state| {
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
                                    let generation =
                                        latest_generation.fetch_add(1, Ordering::AcqRel) + 1;
                                    *load_status
                                        .lock()
                                        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                        ModelLoadStatus::Loading {
                                            generation,
                                            path: path.clone(),
                                        };
                                    async_executor.execute_background(NamTask::LoadModel {
                                        generation,
                                        path,
                                    });
                                }
                            }

                            if installed_generation.load(Ordering::Acquire) != 0 {
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

                        let installed = installed_generation.load(Ordering::Acquire);
                        let mut status = load_status
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        let ready = match &*status {
                            ModelLoadStatus::Loading { generation, path }
                                if *generation == installed =>
                            {
                                Some((*generation, path.clone()))
                            }
                            _ => None,
                        };
                        if let Some((generation, path)) = ready {
                            *params
                                .model_path
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                path.to_string_lossy().to_string();
                            *status = ModelLoadStatus::Ready { generation, path };
                        }
                        match &*status {
                            ModelLoadStatus::Empty => {
                                ui.label("No model loaded");
                            }
                            ModelLoadStatus::Loading { path, .. } => {
                                ui.label(path.file_name().unwrap_or_default().to_string_lossy());
                                ui.label("Loading...");
                            }
                            ModelLoadStatus::Ready { generation, path } => {
                                debug_assert_eq!(*generation, installed);
                                ui.label(path.file_name().unwrap_or_default().to_string_lossy());
                                ui.label("Ready");
                            }
                            ModelLoadStatus::Failed {
                                generation,
                                path,
                                message,
                            } => {
                                debug_assert_eq!(
                                    *generation,
                                    latest_generation.load(Ordering::Acquire)
                                );
                                ui.label(path.file_name().unwrap_or_default().to_string_lossy());
                                ui.colored_label(egui::Color32::RED, message);
                            }
                        }
                        drop(status);

                        let process_error =
                            AudioProcessError::from_raw(audio_error.load(Ordering::Acquire));
                        if process_error != AudioProcessError::None {
                            ui.colored_label(egui::Color32::RED, process_error.message());
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

        let path = self
            .params
            .model_path
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if !path.is_empty() && self.model.is_none() {
            let path = PathBuf::from(path);
            let generation = self.latest_generation.fetch_add(1, Ordering::AcqRel) + 1;
            *self
                .load_status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = ModelLoadStatus::Loading {
                generation,
                path: path.clone(),
            };
            context.execute(NamTask::LoadModel { generation, path });
        }

        true
    }

    fn reset(&mut self) {
        self.audio_error
            .store(AudioProcessError::None as u8, Ordering::Release);
        if let Some(model) = self.model.as_mut() {
            let model_rate = model
                .dsp
                .metadata()
                .expected_sample_rate
                .unwrap_or(self.sample_rate);
            model.dsp.reset(model_rate, self.max_buffer_size);
            model.dsp.prewarm();
            if let Some(resampler) = model.resampler.as_mut() {
                resampler.reset();
            }
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.process_buffer(buffer)
    }
}

impl Drop for NamPlugin {
    fn drop(&mut self) {
        self.plugin_alive.store(false, Ordering::Release);
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
mod tests;
