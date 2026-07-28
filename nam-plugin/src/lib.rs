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
        let loaded_models = Arc::new(ArrayQueue::new(1));
        let retired_models = Arc::new(ArrayQueue::new(4));
        let retired_models_weak = Arc::downgrade(&retired_models);
        let _ = thread::Builder::new()
            .name("nam-model-reaper".to_string())
            .spawn(move || {
                while let Some(retired_models) = retired_models_weak.upgrade() {
                    while retired_models.pop().is_some() {}
                    thread::park_timeout(Duration::from_millis(10));
                }
            });

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

        self.install_pending_model();
        let model = match self.model.as_mut() {
            Some(model) => model,
            None => return ProcessStatus::Normal,
        };
        model.dsp.set_activation_mode(activation_mode);

        let channel_data = buffer.as_slice();
        let channel = &mut channel_data[0];

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
        let reported_error = AudioProcessError::from_raw(self.audio_error.load(Ordering::Acquire));
        if process_error == AudioProcessError::NonFiniteOutput
            || reported_error != AudioProcessError::NonFiniteOutput
        {
            self.audio_error
                .store(process_error as u8, Ordering::Release);
        }

        for (sample, &output) in channel.iter_mut().zip(&self.output_buf[..num_samples]) {
            let out_gain = util::db_to_gain_fast(self.params.output_gain.smoothed.next());
            *sample = nam_core::dsp::sample_to_f32(output) * out_gain;
        }

        ProcessStatus::Normal
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
            let mut plugin = NamPlugin::default();
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
mod tests {
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
        // SAFETY: The buffer does not outlive `samples`, and every channel has `samples.len()`
        // elements.
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

    fn plugin_with_resampled_passthrough_model(
        host_rate: usize,
        model_rate: usize,
        buffer_size: usize,
    ) -> NamPlugin {
        let mut plugin = plugin_with_passthrough_model(buffer_size);
        if let Some(model) = plugin.model.as_mut() {
            model.resampler =
                Some(ResamplerState::new(host_rate, model_rate, buffer_size).unwrap());
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
                    Event::LoaderCompletes(generation)
                        if self.alive && generation == self.latest =>
                    {
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
}
