use crossbeam_queue::ArrayQueue;
use nih_plug::prelude::*;
use nih_plug::wrapper::state::ParamValue;
use nih_plug_egui::resizable_window::ResizableWindow;
use nih_plug_egui::{create_egui_editor, egui, widgets, EguiState};
use parking_lot::Mutex;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
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
        static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) struct TrackingAllocator;

    // SAFETY: Every allocation operation is forwarded to `System` with unchanged arguments.
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation(layout.size());
            // SAFETY: This allocator forwards the unchanged layout to the system allocator.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation(layout.size());
            // SAFETY: This allocator forwards the unchanged layout to the system allocator.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: The pointer and layout came from the system allocator above.
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record_allocation(new_size);
            // SAFETY: The pointer and layout came from the system allocator above.
            unsafe { System.realloc(pointer, layout, new_size) }
        }
    }

    fn record_allocation(size: usize) {
        TRACKING.with(|tracking| {
            if tracking.get() {
                ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
                ALLOCATED_BYTES.with(|bytes| bytes.set(bytes.get().saturating_add(size)));
            }
        });
    }

    pub(super) fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
        let (result, allocations, _) = measure_allocations(operation);
        (result, allocations)
    }

    pub(super) fn measure_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
        ALLOCATIONS.with(|allocations| allocations.set(0));
        ALLOCATED_BYTES.with(|bytes| bytes.set(0));
        TRACKING.with(|tracking| tracking.set(true));
        let result = operation();
        TRACKING.with(|tracking| tracking.set(false));
        let count = ALLOCATIONS.with(Cell::get);
        let bytes = ALLOCATED_BYTES.with(Cell::get);
        (result, count, bytes)
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: allocation_tracking::TrackingAllocator =
    allocation_tracking::TrackingAllocator;

const MAX_SUPPORTED_SAMPLE_RATE: f64 = 768_000.0;
const MAX_SUPPORTED_BUFFER_SIZE: usize = 1_048_576;
const MAX_MODEL_PATH_BYTES: usize = 32 * 1024;

enum NamTask {
    LoadModel {
        generation: u64,
        path: PathBuf,
        host_sample_rate: f64,
        max_buffer_size: usize,
    },
}

struct LoadedModel {
    generation: u64,
    source_path: Option<PathBuf>,
    host_sample_rate: usize,
    model_sample_rate: usize,
    configured_max_buffer_size: usize,
    dsp: Box<dyn nam_core::Dsp>,
    resampler: Option<ResamplerState>,
    applied_model_size: Option<f32>,
}

impl LoadedModel {
    fn apply_model_size(&mut self, requested: f32) {
        if self
            .applied_model_size
            .is_some_and(|applied| applied != requested)
            && self.dsp.set_slimming(f64::from(requested)).is_ok()
        {
            self.applied_model_size = Some(requested);
        }
    }

    fn configure_processing(
        &mut self,
        host_sample_rate: f64,
        max_buffer_size: usize,
    ) -> Result<(), String> {
        let model_sample_rate = self
            .dsp
            .metadata()
            .expected_sample_rate
            .unwrap_or(host_sample_rate);
        let host_rate = validate_sample_rate(host_sample_rate, "host sample rate")?;
        let model_rate = validate_sample_rate(model_sample_rate, "model sample rate")?;

        let resampler = if host_rate == model_rate {
            None
        } else {
            Some(
                ResamplerState::new(host_rate, model_rate, max_buffer_size).map_err(|error| {
                    format!(
                        "Could not resample from {host_sample_rate:.0} Hz to {model_sample_rate:.0} Hz: {error}"
                    )
                })?,
            )
        };
        self.dsp.reset(model_sample_rate, max_buffer_size);
        self.dsp.prewarm();
        self.resampler = resampler;
        self.host_sample_rate = host_rate;
        self.model_sample_rate = model_rate;
        self.configured_max_buffer_size = max_buffer_size;
        Ok(())
    }
}

fn validate_sample_rate(sample_rate: f64, label: &str) -> Result<usize, String> {
    if !sample_rate.is_finite() || !(1.0..=MAX_SUPPORTED_SAMPLE_RATE).contains(&sample_rate) {
        return Err(format!(
            "Invalid {label} {sample_rate:?}; expected a finite value from 1 to {MAX_SUPPORTED_SAMPLE_RATE:.0} Hz"
        ));
    }
    Ok(sample_rate.round() as usize)
}

fn make_audio_buffer(size: usize) -> Result<Vec<nam_core::Sample>, String> {
    if size == 0 || size > MAX_SUPPORTED_BUFFER_SIZE {
        return Err(format!(
            "Invalid maximum buffer size {size}; expected 1 to {MAX_SUPPORTED_BUFFER_SIZE} samples"
        ));
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(size)
        .map_err(|error| format!("Could not reserve a {size}-sample audio buffer: {error}"))?;
    buffer.resize(size, nam_core::Sample::default());
    Ok(buffer)
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
    audio_error_occurrences: Arc<AtomicU64>,
    audio_error_block_size: Arc<AtomicUsize>,
    plugin_alive: Arc<AtomicBool>,
    host_sample_rate_bits: Arc<AtomicU64>,
    host_max_buffer_size: Arc<AtomicUsize>,
    installed_model_sample_rate: Arc<AtomicUsize>,
    installed_model_max_buffer_size: Arc<AtomicUsize>,
    installed_model_resampling: Arc<AtomicBool>,
    model_reaper: Option<thread::JoinHandle<()>>,
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
    *status.lock() = ModelLoadStatus::Failed {
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

    #[id = "model_size"]
    pub model_size: FloatParam,

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
        let plugin_alive = Arc::new(AtomicBool::new(true));
        let model_reaper = if spawn_model_reaper {
            let retired_models_weak = Arc::downgrade(&retired_models);
            let plugin_alive = plugin_alive.clone();
            thread::Builder::new()
                .name("nam-model-reaper".to_string())
                .spawn(move || {
                    while let Some(retired_models) = retired_models_weak.upgrade() {
                        while retired_models.pop().is_some() {}
                        if !plugin_alive.load(Ordering::Acquire) {
                            break;
                        }
                        thread::park_timeout(Duration::from_millis(10));
                    }
                })
                .ok()
        } else {
            None
        };

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
            audio_error_occurrences: Arc::new(AtomicU64::new(0)),
            audio_error_block_size: Arc::new(AtomicUsize::new(0)),
            plugin_alive,
            host_sample_rate_bits: Arc::new(AtomicU64::new(48_000.0f64.to_bits())),
            host_max_buffer_size: Arc::new(AtomicUsize::new(4096)),
            installed_model_sample_rate: Arc::new(AtomicUsize::new(0)),
            installed_model_max_buffer_size: Arc::new(AtomicUsize::new(0)),
            installed_model_resampling: Arc::new(AtomicBool::new(false)),
            model_reaper,
            input_buf: Vec::new(),
            output_buf: Vec::new(),
            sample_rate: 48000.0,
            max_buffer_size: 4096,
        }
    }

    fn stop_model_reaper(&mut self) {
        self.plugin_alive.store(false, Ordering::Release);
        if let Some(model_reaper) = self.model_reaper.take() {
            model_reaper.thread().unpark();
            let _ = model_reaper.join();
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

            model_size: FloatParam::new(
                "Model Size",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_step_size(0.01)
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

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
        self.installed_model_sample_rate
            .store(loaded.model_sample_rate, Ordering::Release);
        self.installed_model_max_buffer_size
            .store(loaded.configured_max_buffer_size, Ordering::Release);
        self.installed_model_resampling
            .store(loaded.resampler.is_some(), Ordering::Release);
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
            self.report_audio_error(AudioProcessError::CallbackLayout, num_samples);
            return ProcessStatus::Normal;
        }
        let channel = &mut channel_data[0];
        if num_samples > self.input_buf.len() || num_samples > self.output_buf.len() {
            channel.fill(0.0);
            self.report_audio_error(AudioProcessError::CallbackCapacity, num_samples);
            return ProcessStatus::Normal;
        }

        self.install_pending_model();
        let model = match self.model.as_mut() {
            Some(model) => model,
            None => return ProcessStatus::Normal,
        };
        model.dsp.set_activation_mode(activation_mode);
        model.apply_model_size(self.params.model_size.value());

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
        self.report_audio_error(process_error, num_samples);

        for (sample, &output) in channel.iter_mut().zip(&self.output_buf[..num_samples]) {
            let out_gain = util::db_to_gain_fast(self.params.output_gain.smoothed.next());
            *sample = nam_core::dsp::sample_to_f32(output) * out_gain;
        }

        ProcessStatus::Normal
    }

    fn report_audio_error(&self, process_error: AudioProcessError, block_size: usize) {
        if process_error != AudioProcessError::None {
            self.audio_error_occurrences.fetch_add(1, Ordering::Relaxed);
            self.audio_error_block_size
                .store(block_size, Ordering::Relaxed);
        }
        let reported_error = AudioProcessError::from_raw(self.audio_error.load(Ordering::Acquire));
        if process_error == AudioProcessError::NonFiniteOutput
            || reported_error != AudioProcessError::NonFiniteOutput
        {
            self.audio_error
                .store(process_error as u8, Ordering::Release);
        }
    }

    #[cfg(test)]
    fn diagnostics_summary(&self) -> String {
        build_plugin_diagnostics(
            &self.params,
            &self.load_status,
            &self.latest_generation,
            &self.installed_generation,
            &self.host_sample_rate_bits,
            &self.host_max_buffer_size,
            &self.installed_model_sample_rate,
            &self.installed_model_max_buffer_size,
            &self.installed_model_resampling,
            &self.audio_error,
            &self.audio_error_occurrences,
            &self.audio_error_block_size,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn build_plugin_diagnostics(
    params: &NamParams,
    load_status: &Mutex<ModelLoadStatus>,
    latest_generation: &AtomicU64,
    installed_generation: &AtomicU64,
    host_sample_rate_bits: &AtomicU64,
    host_max_buffer_size: &AtomicUsize,
    installed_model_sample_rate: &AtomicUsize,
    installed_model_max_buffer_size: &AtomicUsize,
    installed_model_resampling: &AtomicBool,
    audio_error: &AtomicU8,
    audio_error_occurrences: &AtomicU64,
    audio_error_block_size: &AtomicUsize,
) -> String {
    let status = match &*load_status.lock() {
        ModelLoadStatus::Empty => "empty".to_string(),
        ModelLoadStatus::Loading { generation, path } => {
            format!("loading generation {generation}: {}", path.display())
        }
        ModelLoadStatus::Ready { generation, path } => {
            format!("ready generation {generation}: {}", path.display())
        }
        ModelLoadStatus::Failed {
            generation,
            path,
            message,
        } => format!(
            "failed generation {generation}: {} ({message})",
            path.display()
        ),
    };
    let process_error = AudioProcessError::from_raw(audio_error.load(Ordering::Acquire));
    format!(
        "NAM Plugin Diagnostics\n\
         build: {}\n\
         model_path: {}\n\
         load_status: {status}\n\
         generation: latest={}, installed={}\n\
         host: {:.0} Hz, max {} samples\n\
         model: {} Hz, max {} samples, resampling={}\n\
         audio_error: {} (code {}, occurrences {}, last_block_size {})\n",
        nam_core::build_info::SUMMARY,
        params.model_path.lock(),
        latest_generation.load(Ordering::Acquire),
        installed_generation.load(Ordering::Acquire),
        f64::from_bits(host_sample_rate_bits.load(Ordering::Acquire)),
        host_max_buffer_size.load(Ordering::Acquire),
        installed_model_sample_rate.load(Ordering::Acquire),
        installed_model_max_buffer_size.load(Ordering::Acquire),
        installed_model_resampling.load(Ordering::Acquire),
        process_error.message(),
        process_error as u8,
        audio_error_occurrences.load(Ordering::Acquire),
        audio_error_block_size.load(Ordering::Acquire),
    )
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
    use super::{LoadedModel, NamParams, NamPlugin, NamTask, ProcessStatus, ResamplerState};
    use nih_plug::prelude::{
        Buffer, BufferConfig, FloatParam, FloatRange, InitContext, Params, Plugin, PluginApi,
        ProcessMode, TaskExecutor,
    };
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

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

    struct LifecycleInitContext<'a> {
        executor: &'a TaskExecutor<NamPlugin>,
    }

    impl InitContext<NamPlugin> for LifecycleInitContext<'_> {
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Standalone
        }

        fn execute(&self, task: NamTask) {
            (self.executor)(task);
        }

        fn set_latency_samples(&self, _samples: u32) {}

        fn set_current_voice_capacity(&self, _capacity: u32) {}
    }

    pub struct LifecycleCase {
        plugin: NamPlugin,
        executor: TaskExecutor<NamPlugin>,
    }

    impl Default for LifecycleCase {
        fn default() -> Self {
            let mut plugin = NamPlugin::new(false);
            let executor = Plugin::task_executor(&mut plugin);
            Self { plugin, executor }
        }
    }

    impl LifecycleCase {
        pub fn restore_model(
            &mut self,
            path: &Path,
            sample_rate: f32,
            max_buffer_size: u32,
        ) -> bool {
            *self.plugin.params.model_path.lock() = path.to_string_lossy().into_owned();
            self.initialize(sample_rate, max_buffer_size)
        }

        pub fn clear_model(&mut self, sample_rate: f32, max_buffer_size: u32) -> bool {
            self.plugin.params.model_path.lock().clear();
            self.initialize(sample_rate, max_buffer_size)
        }

        pub fn restore_serialized_model_path(
            &mut self,
            serialized: String,
            sample_rate: f32,
            max_buffer_size: u32,
        ) -> bool {
            let fields =
                std::collections::BTreeMap::from([(String::from("model_path"), serialized)]);
            self.plugin.params.deserialize_fields(&fields);
            self.initialize(sample_rate, max_buffer_size)
        }

        pub fn set_model_size(&mut self, model_size: f32) {
            let model_size = if model_size.is_finite() {
                model_size.clamp(0.0, 1.0)
            } else {
                1.0
            };
            let model_path = self.plugin.params.model_path.lock().clone();
            self.plugin.params = Arc::new(NamParams {
                model_size: FloatParam::new(
                    "Model Size",
                    model_size,
                    FloatRange::Linear { min: 0.0, max: 1.0 },
                )
                .with_step_size(0.01),
                model_path: parking_lot::Mutex::new(model_path),
                ..NamParams::default()
            });
        }

        fn initialize(&mut self, sample_rate: f32, max_buffer_size: u32) -> bool {
            let mut context = LifecycleInitContext {
                executor: &self.executor,
            };
            let config = BufferConfig {
                sample_rate,
                min_buffer_size: None,
                max_buffer_size,
                process_mode: ProcessMode::Realtime,
            };
            Plugin::initialize(
                &mut self.plugin,
                &NamPlugin::AUDIO_IO_LAYOUTS[0],
                &config,
                &mut context,
            )
        }

        pub fn process(&mut self, buffer_size: usize) -> bool {
            if buffer_size > 4096 {
                return false;
            }
            let mut audio = vec![0.1f32; buffer_size];
            let mut buffer = Buffer::default();
            // SAFETY: The buffer cannot outlive `audio`, and its only channel contains exactly
            // `buffer_size` samples.
            unsafe {
                buffer.set_slices(buffer_size, |channels| channels.push(&mut audio));
            }
            self.plugin.process_buffer(&mut buffer) == ProcessStatus::Normal
                && audio.iter().all(|sample| sample.is_finite())
        }

        pub fn reset(&mut self) {
            Plugin::reset(&mut self.plugin);
        }

        pub fn deactivate(&mut self) {
            Plugin::deactivate(&mut self.plugin);
        }
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
                source_path: None,
                host_sample_rate: host_rate,
                model_sample_rate: model_rate,
                configured_max_buffer_size: buffer_size,
                dsp: Box::new(PassthroughDsp),
                resampler,
                applied_model_size: None,
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

        fn new_model(
            model_name: &str,
            buffer_size: usize,
            model_size: Option<f32>,
        ) -> Result<Self, nam_core::NamError> {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../nam-core/test_fixtures/models")
                .join(model_name);
            let mut dsp = nam_core::get_dsp(&path)?;
            let sample_rate = dsp.metadata().expected_sample_rate.unwrap_or(48_000.0);
            let applied_model_size = match model_size {
                Some(model_size) => {
                    dsp.set_slimming(f64::from(model_size))?;
                    Some(model_size)
                }
                None => None,
            };
            dsp.reset(sample_rate, buffer_size);
            dsp.prewarm();

            let mut plugin = NamPlugin::new(false);
            if let Some(model_size) = model_size {
                plugin.params = Arc::new(NamParams {
                    model_size: FloatParam::new(
                        "Model Size",
                        model_size,
                        FloatRange::Linear { min: 0.0, max: 1.0 },
                    ),
                    ..NamParams::default()
                });
            }
            plugin.model = Some(LoadedModel {
                generation: 1,
                source_path: Some(path),
                host_sample_rate: sample_rate.round() as usize,
                model_sample_rate: sample_rate.round() as usize,
                configured_max_buffer_size: buffer_size,
                dsp,
                resampler: None,
                applied_model_size,
            });
            plugin.latest_generation.store(1, Ordering::Release);
            plugin.installed_generation.store(1, Ordering::Release);
            plugin.input_buf = vec![0.0; buffer_size];
            plugin.output_buf = vec![0.0; buffer_size];
            plugin.sample_rate = sample_rate;
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

        pub fn new_a1(buffer_size: usize) -> Result<Self, nam_core::NamError> {
            Self::new_model("wavenet_a1_standard.nam", buffer_size, None)
        }

        pub fn new_a2(buffer_size: usize) -> Result<Self, nam_core::NamError> {
            Self::new_model("wavenet_a2_max.nam", buffer_size, None)
        }

        pub fn new_packed_a2_small(buffer_size: usize) -> Result<Self, nam_core::NamError> {
            Self::new_model("upstream_packed_a2_export.nam", buffer_size, Some(0.25))
        }

        pub fn new_packed_a2_full(buffer_size: usize) -> Result<Self, nam_core::NamError> {
            Self::new_model("upstream_packed_a2_export.nam", buffer_size, Some(1.0))
        }

        pub fn set_model_size(&mut self, model_size: f32) {
            if let Some(model) = self.plugin.model.as_mut() {
                model.apply_model_size(model_size);
            }
        }
    }
}

impl Plugin for NamPlugin {
    const NAME: &'static str = "NAM";
    const VENDOR: &'static str = "nam-rs";
    const URL: &'static str = "https://github.com/joshburgess/nam-rs";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = nam_core::build_info::VERSION;

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

        Box::new(move |task| match task {
            NamTask::LoadModel {
                generation,
                path,
                host_sample_rate,
                max_buffer_size,
            } => {
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
                        let requested_model_size = params.model_size.value();
                        let applied_model_size = dsp
                            .set_slimming(f64::from(requested_model_size))
                            .is_ok()
                            .then_some(requested_model_size);
                        let mut loaded = LoadedModel {
                            generation,
                            source_path: Some(path.clone()),
                            host_sample_rate: 0,
                            model_sample_rate: 0,
                            configured_max_buffer_size: 0,
                            dsp,
                            resampler: None,
                            applied_model_size,
                        };
                        if let Err(message) =
                            loaded.configure_processing(host_sample_rate, max_buffer_size)
                        {
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
                        *params.model_path.lock() = path.to_string_lossy().into_owned();
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
        let audio_error_occurrences = self.audio_error_occurrences.clone();
        let audio_error_block_size = self.audio_error_block_size.clone();
        let host_sample_rate_bits = self.host_sample_rate_bits.clone();
        let host_max_buffer_size = self.host_max_buffer_size.clone();
        let installed_model_sample_rate = self.installed_model_sample_rate.clone();
        let installed_model_max_buffer_size = self.installed_model_max_buffer_size.clone();
        let installed_model_resampling = self.installed_model_resampling.clone();

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
                                    *load_status.lock() = ModelLoadStatus::Loading {
                                        generation,
                                        path: path.clone(),
                                    };
                                    async_executor.execute_background(NamTask::LoadModel {
                                        generation,
                                        path,
                                        host_sample_rate: f64::from_bits(
                                            host_sample_rate_bits.load(Ordering::Acquire),
                                        ),
                                        max_buffer_size: host_max_buffer_size
                                            .load(Ordering::Acquire),
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
                        let mut status = load_status.lock();
                        let ready = match &*status {
                            ModelLoadStatus::Loading { generation, path }
                                if *generation == installed =>
                            {
                                Some((*generation, path.clone()))
                            }
                            _ => None,
                        };
                        if let Some((generation, path)) = ready {
                            *params.model_path.lock() = path.to_string_lossy().to_string();
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

                        if ui.button("Copy Diagnostics").clicked() {
                            ui.ctx().copy_text(build_plugin_diagnostics(
                                &params,
                                &load_status,
                                &latest_generation,
                                &installed_generation,
                                &host_sample_rate_bits,
                                &host_max_buffer_size,
                                &installed_model_sample_rate,
                                &installed_model_max_buffer_size,
                                &installed_model_resampling,
                                &audio_error,
                                &audio_error_occurrences,
                                &audio_error_block_size,
                            ));
                        }

                        ui.separator();

                        ui.label("Input Gain");
                        ui.add(widgets::ParamSlider::for_param(&params.input_gain, setter));

                        ui.label("Output Gain");
                        ui.add(widgets::ParamSlider::for_param(&params.output_gain, setter));

                        ui.label("Model Size");
                        ui.add(widgets::ParamSlider::for_param(&params.model_size, setter));

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

    fn filter_state(state: &mut PluginState) {
        for (id, min, max) in [
            ("in_gain", -24.0, 24.0),
            ("out_gain", -40.0, 40.0),
            ("model_size", 0.0, 1.0),
        ] {
            let Some(value) = state.params.get_mut(id) else {
                continue;
            };
            let valid = match value {
                ParamValue::F32(value) if value.is_finite() => {
                    *value = value.clamp(min, max);
                    true
                }
                _ => false,
            };
            if !valid {
                state.params.remove(id);
            }
        }

        let remove_model_path = state.fields.get("model_path").is_some_and(|serialized| {
            serialized.len() > MAX_MODEL_PATH_BYTES
                || serde_json::from_str::<String>(serialized)
                    .map(|path| path.len() > MAX_MODEL_PATH_BYTES)
                    .unwrap_or(true)
        });
        if remove_model_path {
            state.fields.remove("model_path");
        }
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        nih_log!("NAM build {}", nam_core::build_info::SUMMARY);
        let sample_rate = f64::from(buffer_config.sample_rate);
        let max_buffer_size = buffer_config.max_buffer_size as usize;
        if let Err(message) = validate_sample_rate(sample_rate, "host sample rate") {
            nih_error!("{message}");
            return false;
        }
        let input_buf = match make_audio_buffer(max_buffer_size) {
            Ok(buffer) => buffer,
            Err(message) => {
                nih_error!("{message}");
                return false;
            }
        };
        let output_buf = match make_audio_buffer(max_buffer_size) {
            Ok(buffer) => buffer,
            Err(message) => {
                nih_error!("{message}");
                return false;
            }
        };

        self.sample_rate = sample_rate;
        self.max_buffer_size = max_buffer_size;
        self.host_sample_rate_bits
            .store(sample_rate.to_bits(), Ordering::Release);
        self.host_max_buffer_size
            .store(max_buffer_size, Ordering::Release);
        self.input_buf = input_buf;
        self.output_buf = output_buf;

        let persisted_path = self.params.model_path.lock().clone();
        if persisted_path.is_empty() {
            self.latest_generation.fetch_add(1, Ordering::AcqRel);
            self.model = None;
            self.deferred_retire = None;
            while self.loaded_models.pop().is_some() {}
            self.installed_generation.store(0, Ordering::Release);
            self.installed_model_sample_rate.store(0, Ordering::Release);
            self.installed_model_max_buffer_size
                .store(0, Ordering::Release);
            self.installed_model_resampling
                .store(false, Ordering::Release);
            *self.load_status.lock() = ModelLoadStatus::Empty;
            return true;
        }

        let path = PathBuf::from(persisted_path);
        let loaded_path_matches = self
            .model
            .as_ref()
            .and_then(|model| model.source_path.as_ref())
            == Some(&path);
        if let Some(model) = self.model.as_mut() {
            model.apply_model_size(self.params.model_size.value());
            model
                .dsp
                .set_activation_mode(if self.params.fast_mode.value() {
                    nam_core::ActivationMode::Fast
                } else {
                    nam_core::ActivationMode::Accurate
                });
            if let Err(message) = model.configure_processing(sample_rate, max_buffer_size) {
                nih_error!("{message}");
                if loaded_path_matches {
                    let generation = model.generation;
                    mark_load_failed(
                        &self.load_status,
                        &self.latest_generation,
                        generation,
                        path,
                        message,
                    );
                }
                return false;
            }
            self.installed_model_sample_rate
                .store(model.model_sample_rate, Ordering::Release);
            self.installed_model_max_buffer_size
                .store(model.configured_max_buffer_size, Ordering::Release);
            self.installed_model_resampling
                .store(model.resampler.is_some(), Ordering::Release);
        }
        if loaded_path_matches {
            return true;
        }

        let generation = self.latest_generation.fetch_add(1, Ordering::AcqRel) + 1;
        while self.loaded_models.pop().is_some() {}
        *self.load_status.lock() = ModelLoadStatus::Loading {
            generation,
            path: path.clone(),
        };
        context.execute(NamTask::LoadModel {
            generation,
            path,
            host_sample_rate: sample_rate,
            max_buffer_size,
        });
        true
    }

    fn reset(&mut self) {
        self.audio_error
            .store(AudioProcessError::None as u8, Ordering::Release);
        if let Some(model) = self.model.as_mut() {
            model.apply_model_size(self.params.model_size.value());
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
        self.stop_model_reaper();
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
