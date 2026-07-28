use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use crate::background::BackgroundOperation;
pub(crate) use crate::diagnostics::build_diagnostics_summary;
#[cfg(test)]
use crate::environment::remove_managed_miniforge;
use crate::environment::{default_python_name, managed_miniforge_dir, MANAGED_MINIFORGE_MARKER};
#[cfg(test)]
use crate::environment::{miniforge_installer_info, MINIFORGE_VERSION};
#[cfg(test)]
use crate::environment_service::parse_environment_report;
use crate::errors::TrainingErrorKind;
use crate::install_service::InstallOperation;
use crate::run_controller::RunFinalization;
#[cfg(test)]
pub(crate) use crate::run_manifest::make_run_id;
#[cfg(test)]
use crate::run_manifest::save_training_log;
use crate::run_manifest::{cleanup_run_resources, unix_timestamp_secs, CleanupReport};
pub(crate) use crate::run_manifest::{prepare_training_run, ActiveTrainingRun};
use crate::settings::Settings;
use crate::ui;
pub use crate::validation::{validate_audio_files, ValidationSeverity};
use crate::worker::{TrainingState, WorkerHandle, WorkerMessage, WorkerMessageReceiver};

/// Top-level application state.
pub struct TrainerApp {
    // File paths
    pub(crate) input_path: Option<PathBuf>,
    pub(crate) output_paths: Vec<PathBuf>,
    pub(crate) destination_dir: Option<PathBuf>,

    // Training configuration
    pub(crate) config: TrainingConfig,
    pub(crate) metadata: ModelMetadata,
    pub(crate) allow_overwrite_outputs: bool,

    // Sub-window visibility
    pub(crate) show_advanced: bool,
    pub(crate) show_metadata: bool,

    // Training state
    pub(crate) training_log: Vec<String>,
    pub(crate) epoch_history: Vec<EpochStats>,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) run: TrainingRunContext,
    pub(crate) user_action_error: Option<String>,

    // Persistent settings
    pub(crate) settings: Settings,

    // Python executable path
    pub(crate) python_path: PathBuf,

    // Device selection
    pub(crate) selected_device: DeviceId,

    // Python discovery and environment status
    pub(crate) discovered_pythons: Option<Vec<crate::environment_service::PythonEntry>>,
    pub(crate) python_discovery_rx:
        Option<BackgroundOperation<Vec<crate::environment_service::PythonEntry>>>,
    pub(crate) python_status: PythonStatus,
    pub(crate) cuda_install: Option<CudaInstall>,
    python_check_rx: Option<BackgroundOperation<DetectionResult>>,

    // NAM install state
    pub(crate) install_state: InstallState,
    pub(crate) install_log: Vec<String>,
    pub(crate) install_rx: Option<InstallOperation>,
    pub(crate) pending_destructive_action: Option<InstallAction>,
}

pub(crate) enum TrainingRunContext {
    Idle(RunData),
    Running {
        data: RunData,
        worker: Option<WorkerHandle>,
        messages: WorkerMessageReceiver,
    },
    Finishing {
        data: RunData,
        worker: Option<WorkerHandle>,
        messages: WorkerMessageReceiver,
        result: TrainingRunResult,
    },
    Finished {
        data: RunData,
        result: TrainingRunResult,
    },
}

#[derive(Default)]
pub(crate) struct RunData {
    pub(crate) artifacts: Option<ActiveTrainingRun>,
    pub(crate) completed_models: Vec<PathBuf>,
    pub(crate) failed_files: Vec<TrainingFileFailure>,
    pub(crate) current_file_index: usize,
    pub(crate) total_files: usize,
    pub(crate) started_at: Option<Instant>,
    pub(crate) last_epoch_at: Option<Instant>,
    pub(crate) avg_epoch_secs: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TrainingRunResult {
    Complete,
    Error(String),
    Cancelled,
}

impl Default for TrainingRunContext {
    fn default() -> Self {
        Self::Idle(RunData::default())
    }
}

impl TrainingRunContext {
    pub(crate) fn activate(
        &mut self,
        worker: Option<WorkerHandle>,
        messages: WorkerMessageReceiver,
    ) {
        let data = std::mem::take(self.data_mut());
        *self = Self::Running {
            data,
            worker,
            messages,
        };
    }

    pub(crate) fn state(&self) -> TrainingState {
        match self {
            Self::Idle(_) => TrainingState::Idle,
            Self::Running { .. } => TrainingState::Training,
            Self::Finishing { result, .. } | Self::Finished { result, .. } => {
                result.as_training_state()
            }
        }
    }

    pub(crate) fn data(&self) -> &RunData {
        match self {
            Self::Idle(data)
            | Self::Running { data, .. }
            | Self::Finishing { data, .. }
            | Self::Finished { data, .. } => data,
        }
    }

    pub(crate) fn data_mut(&mut self) -> &mut RunData {
        match self {
            Self::Idle(data)
            | Self::Running { data, .. }
            | Self::Finishing { data, .. }
            | Self::Finished { data, .. } => data,
        }
    }

    pub(crate) fn finish(&mut self, result: TrainingRunResult) {
        let previous = std::mem::take(self);
        *self = match previous {
            Self::Running {
                data,
                worker,
                messages,
            } => Self::Finishing {
                data,
                worker,
                messages,
                result,
            },
            Self::Finishing {
                data,
                worker,
                messages,
                ..
            } => Self::Finishing {
                data,
                worker,
                messages,
                result,
            },
            Self::Idle(data) | Self::Finished { data, .. } => Self::Finished { data, result },
        };
    }

    pub(crate) fn finish_error(&mut self, message: impl Into<String>) {
        self.finish(TrainingRunResult::Error(message.into()));
    }

    pub(crate) fn messages(&self) -> Option<&WorkerMessageReceiver> {
        match self {
            Self::Running { messages, .. } | Self::Finishing { messages, .. } => Some(messages),
            Self::Idle(_) | Self::Finished { .. } => None,
        }
    }

    fn worker_mut(&mut self) -> Option<&mut WorkerHandle> {
        match self {
            Self::Running { worker, .. } | Self::Finishing { worker, .. } => worker.as_mut(),
            Self::Idle(_) | Self::Finished { .. } => None,
        }
    }

    pub(crate) fn worker_exited(&mut self) {
        let previous = std::mem::take(self);
        *self = match previous {
            Self::Finishing { data, result, .. } => Self::Finished { data, result },
            other => other,
        };
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    #[cfg(test)]
    fn is_active(&self) -> bool {
        matches!(self, Self::Running { .. } | Self::Finishing { .. })
    }

    #[cfg(test)]
    fn set_test_state(&mut self, state: TrainingState) {
        match state {
            TrainingState::Idle => self.reset(),
            TrainingState::Training => {
                let (_tx, rx) = crate::worker::worker_message_channel(1);
                self.activate(None, rx);
            }
            TrainingState::Complete => self.finish(TrainingRunResult::Complete),
            TrainingState::Error(message) => self.finish_error(message),
        }
    }
}

impl TrainingRunResult {
    fn as_training_state(&self) -> TrainingState {
        match self {
            Self::Complete => TrainingState::Complete,
            Self::Error(message) => TrainingState::Error(message.clone()),
            Self::Cancelled => TrainingState::Idle,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstallState {
    Idle,
    Installing(InstallAction),
    Done,
    Failed,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstallAction {
    InstallingPython,
    InstallingNam,
    InstallingCudaTorch,
    UninstallingNam,
    UninstallingMiniforge,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaInstall {
    pub cuda_version: String,
    pub wheel_index: String,
    pub gpu_names: Vec<String>,
}

pub struct DetectionResult {
    pub status: PythonStatus,
    pub cuda_install: Option<CudaInstall>,
}

/// Minimum Python version required by neural-amp-modeler.
pub const NAM_MIN_PYTHON: (u32, u32) = (3, 10);
pub const DEFAULT_OUTPUT_SAMPLES_PER_DATUM: u32 = 8192;
const MAX_TRAINING_LOG_LINES: usize = 2_000;
const MAX_INSTALL_LOG_LINES: usize = 1_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingFileFailure {
    pub file: String,
    pub kind: TrainingErrorKind,
    pub message: String,
}

#[derive(Clone, Debug)]
pub enum PythonStatus {
    Unknown,
    Ok {
        version: String,
        devices: Vec<TrainingDevice>,
        warnings: Vec<String>,
        report: EnvironmentReport,
    },
    VersionTooOld {
        version: String,
    },
    /// Python executable not found or not a real Python (e.g. Windows Store alias).
    NotFound,
    Error(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingDevice {
    pub id: DeviceId,
    pub name: String, // "CPU", "CUDA 0: NVIDIA RTX 4090", "Apple GPU (MPS)"
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct DeviceId(String);

impl DeviceId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn starts_with(&self, prefix: &str) -> bool {
        self.0.starts_with(prefix)
    }
}

impl From<&str> for DeviceId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for DeviceId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl PartialEq<&str> for DeviceId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnvironmentReport {
    pub nam_version: Option<String>,
    pub torch_version: Option<String>,
    pub packed_full_config_supported: bool,
}

#[derive(Debug)]
pub enum InstallMessage {
    Log(String),
    PythonInstalled { python_path: PathBuf },
    Done { success: bool },
}

#[derive(Clone)]
pub struct TrainingConfig {
    pub architecture: Architecture,
    pub epochs: u32,
    pub latency: Option<i32>,
    pub threshold_esr: Option<f64>,
    pub batch_size: u32,
    pub lr: f64,
    pub lr_decay: f64,
    pub save_plot: bool,
    pub fit_mrstft: bool,
    pub ignore_checks: bool,
    pub num_output_samples_per_datum: u32,
    pub use_full_config_trainer: bool,
    pub output_model_basename: String,
    pub batch_name_template: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            architecture: Architecture::Standard,
            epochs: 100,
            latency: None,
            threshold_esr: None,
            batch_size: 16,
            lr: 0.004,
            lr_decay: 0.007,
            save_plot: true,
            fit_mrstft: true,
            ignore_checks: false,
            num_output_samples_per_datum: DEFAULT_OUTPUT_SAMPLES_PER_DATUM,
            use_full_config_trainer: false,
            output_model_basename: String::new(),
            batch_name_template: "{stem}".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Packed,
    Standard,
    Lite,
    Feather,
    Nano,
}

impl Architecture {
    pub fn label(self) -> &'static str {
        match self {
            Self::Packed => "Packed A2",
            Self::Standard => "Standard",
            Self::Lite => "Lite",
            Self::Feather => "Feather",
            Self::Nano => "Nano",
        }
    }

    pub fn all() -> &'static [Architecture] {
        &[
            Self::Packed,
            Self::Standard,
            Self::Lite,
            Self::Feather,
            Self::Nano,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Packed => "packed",
            Self::Standard => "standard",
            Self::Lite => "lite",
            Self::Feather => "feather",
            Self::Nano => "nano",
        }
    }

    pub fn parse_lossy(s: &str) -> Self {
        match s {
            "packed" | "a2" | "packed_a2" => Self::Packed,
            "lite" => Self::Lite,
            "feather" => Self::Feather,
            "nano" => Self::Nano,
            _ => Self::Standard,
        }
    }

    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Packed => {
                "Current upstream A2 packed WaveNet training, exports a SlimmableContainer"
            }
            Self::Standard => {
                "Best quality, largest model, slowest to train (~30 min for 100 epochs on GPU)"
            }
            Self::Lite => "Good quality with faster training and smaller model size",
            Self::Feather => {
                "Lightweight model for low-latency use, trades some accuracy for speed"
            }
            Self::Nano => "Smallest and fastest model, best for quick tests or low-power devices",
        }
    }
}

impl FromStr for Architecture {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse_lossy(s))
    }
}

#[derive(Clone, Default)]
pub struct ModelMetadata {
    pub name: String,
    pub modeled_by: String,
    pub gear_make: String,
    pub gear_model: String,
    pub gear_type: Option<GearType>,
    pub tone_type: Option<ToneType>,
    pub input_level_dbu: String,
    pub output_level_dbu: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GearType {
    Amp,
    Pedal,
    PedalAmp,
    AmpCab,
    AmpPedalCab,
    Preamp,
    Studio,
}

impl GearType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Amp => "Amp",
            Self::Pedal => "Pedal",
            Self::PedalAmp => "Pedal + Amp",
            Self::AmpCab => "Amp + Cab",
            Self::AmpPedalCab => "Amp + Pedal + Cab",
            Self::Preamp => "Preamp",
            Self::Studio => "Studio",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Amp => "amp",
            Self::Pedal => "pedal",
            Self::PedalAmp => "pedal_amp",
            Self::AmpCab => "amp_cab",
            Self::AmpPedalCab => "amp_pedal_cab",
            Self::Preamp => "preamp",
            Self::Studio => "studio",
        }
    }

    fn parse_known(s: &str) -> Option<Self> {
        match s {
            "amp" => Some(Self::Amp),
            "pedal" => Some(Self::Pedal),
            "pedal_amp" => Some(Self::PedalAmp),
            "amp_cab" => Some(Self::AmpCab),
            "amp_pedal_cab" => Some(Self::AmpPedalCab),
            "preamp" => Some(Self::Preamp),
            "studio" => Some(Self::Studio),
            _ => None,
        }
    }

    pub fn all() -> &'static [GearType] {
        &[
            Self::Amp,
            Self::Pedal,
            Self::PedalAmp,
            Self::AmpCab,
            Self::AmpPedalCab,
            Self::Preamp,
            Self::Studio,
        ]
    }
}

impl FromStr for GearType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_known(s).ok_or(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToneType {
    Clean,
    Overdrive,
    Crunch,
    HiGain,
    Fuzz,
}

impl ToneType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Clean => "Clean",
            Self::Overdrive => "Overdrive",
            Self::Crunch => "Crunch",
            Self::HiGain => "Hi Gain",
            Self::Fuzz => "Fuzz",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Overdrive => "overdrive",
            Self::Crunch => "crunch",
            Self::HiGain => "hi_gain",
            Self::Fuzz => "fuzz",
        }
    }

    fn parse_known(s: &str) -> Option<Self> {
        match s {
            "clean" => Some(Self::Clean),
            "overdrive" => Some(Self::Overdrive),
            "crunch" => Some(Self::Crunch),
            "hi_gain" => Some(Self::HiGain),
            "fuzz" => Some(Self::Fuzz),
            _ => None,
        }
    }

    pub fn all() -> &'static [ToneType] {
        &[
            Self::Clean,
            Self::Overdrive,
            Self::Crunch,
            Self::HiGain,
            Self::Fuzz,
        ]
    }
}

impl FromStr for ToneType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_known(s).ok_or(())
    }
}

#[derive(Clone)]
pub struct EpochStats {
    pub epoch: u32,
    pub train_loss: f64,
    #[allow(dead_code)] // used in training log text, not in plot
    pub val_loss: f64,
    pub esr: f64,
}

impl TrainerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (settings, settings_warning) = match Settings::load() {
            Ok(settings) => (settings, None),
            Err(error) => (
                Settings::default(),
                Some(format!(
                    "Warning: settings could not be loaded and defaults were used: {error}"
                )),
            ),
        };

        // Restore training config from settings, falling back to defaults
        let mut config = TrainingConfig::default();
        if let Some(ref arch_str) = settings.architecture {
            config.architecture = Architecture::parse_lossy(arch_str);
        }
        if let Some(v) = settings.epochs {
            config.epochs = v;
        }
        if let Some(v) = settings.batch_size {
            config.batch_size = v;
        }
        if let Some(v) = settings.lr {
            config.lr = v;
        }
        if let Some(v) = settings.lr_decay {
            config.lr_decay = v;
        }
        config.latency = settings.latency;
        config.threshold_esr = settings.threshold_esr;
        if let Some(v) = settings.save_plot {
            config.save_plot = v;
        }
        if let Some(v) = settings.ignore_checks {
            config.ignore_checks = v;
        }
        if let Some(v) = settings.num_output_samples_per_datum {
            config.num_output_samples_per_datum = v;
        }
        if let Some(v) = settings.use_full_config_trainer {
            config.use_full_config_trainer = v;
        }
        if let Some(ref v) = settings.output_model_basename {
            config.output_model_basename = v.clone();
        }
        if let Some(ref v) = settings.batch_name_template {
            config.batch_name_template = v.clone();
        }

        // Restore metadata from settings
        let metadata = ModelMetadata {
            name: settings.meta_name.clone().unwrap_or_default(),
            modeled_by: settings.meta_modeled_by.clone().unwrap_or_default(),
            gear_make: settings.meta_gear_make.clone().unwrap_or_default(),
            gear_model: settings.meta_gear_model.clone().unwrap_or_default(),
            gear_type: settings
                .meta_gear_type
                .as_deref()
                .and_then(|value| value.parse().ok()),
            tone_type: settings
                .meta_tone_type
                .as_deref()
                .and_then(|value| value.parse().ok()),
            input_level_dbu: settings.meta_input_level_dbu.clone().unwrap_or_default(),
            output_level_dbu: settings.meta_output_level_dbu.clone().unwrap_or_default(),
        };

        let mut app = Self {
            input_path: settings.last_input_path.clone(),
            output_paths: Vec::new(),
            destination_dir: settings.last_destination.clone(),
            config,
            metadata,
            allow_overwrite_outputs: settings.allow_overwrite_outputs.unwrap_or(false),
            show_advanced: false,
            show_metadata: false,
            training_log: settings_warning.into_iter().collect(),
            epoch_history: Vec::new(),
            model_path: None,
            run: TrainingRunContext::default(),
            user_action_error: None,
            python_path: settings
                .python_path
                .clone()
                .unwrap_or_else(default_python_name),
            selected_device: "cpu".into(),
            discovered_pythons: None,
            python_discovery_rx: None,
            python_status: PythonStatus::Unknown,
            cuda_install: None,
            python_check_rx: None,
            install_state: InstallState::Idle,
            install_log: Vec::new(),
            install_rx: None,
            pending_destructive_action: None,
            settings,
        };
        app.restore_interrupted_run();
        app.check_python();
        app
    }

    /// Save the current training config and metadata to persistent settings.
    pub fn save_config(&mut self) {
        self.settings.architecture = Some(self.config.architecture.as_str().to_string());
        self.settings.epochs = Some(self.config.epochs);
        self.settings.batch_size = Some(self.config.batch_size);
        self.settings.lr = Some(self.config.lr);
        self.settings.lr_decay = Some(self.config.lr_decay);
        self.settings.latency = self.config.latency;
        self.settings.threshold_esr = self.config.threshold_esr;
        self.settings.save_plot = Some(self.config.save_plot);
        self.settings.ignore_checks = Some(self.config.ignore_checks);
        self.settings.num_output_samples_per_datum = Some(self.config.num_output_samples_per_datum);
        self.settings.use_full_config_trainer = Some(self.config.use_full_config_trainer);
        self.settings.allow_overwrite_outputs = Some(self.allow_overwrite_outputs);
        self.settings.output_model_basename = non_empty_opt(&self.config.output_model_basename);
        self.settings.batch_name_template = non_empty_opt(&self.config.batch_name_template);
        self.persist_settings();
    }

    /// Save the current metadata fields to persistent settings.
    pub fn save_metadata(&mut self) {
        self.settings.meta_name = non_empty_opt(&self.metadata.name);
        self.settings.meta_modeled_by = non_empty_opt(&self.metadata.modeled_by);
        self.settings.meta_gear_make = non_empty_opt(&self.metadata.gear_make);
        self.settings.meta_gear_model = non_empty_opt(&self.metadata.gear_model);
        self.settings.meta_gear_type = self.metadata.gear_type.map(|g| g.as_str().to_string());
        self.settings.meta_tone_type = self.metadata.tone_type.map(|t| t.as_str().to_string());
        self.settings.meta_input_level_dbu = non_empty_opt(&self.metadata.input_level_dbu);
        self.settings.meta_output_level_dbu = non_empty_opt(&self.metadata.output_level_dbu);
        self.persist_settings();
    }

    pub fn persist_settings(&mut self) {
        #[cfg(not(test))]
        if let Err(error) = self.settings.save() {
            self.push_training_log(format!("Warning: failed to save settings: {error}"));
        }
    }

    /// Spawn a background thread to verify Python + NAM are available and detect GPU.
    pub fn check_python(&mut self) {
        self.python_check_rx = Some(crate::environment_service::spawn_environment_detection(
            self.python_path.clone(),
        ));
    }

    /// Install neural-amp-modeler into the selected Python environment.
    /// If an NVIDIA GPU was detected, installs a CUDA-enabled PyTorch wheel
    /// first so the user gets a working GPU setup from a single button click.
    pub fn install_nam(&mut self) {
        self.install_rx = Some(crate::install_service::spawn_nam_install(
            self.python_path.clone(),
            self.cuda_install.clone(),
        ));
        self.install_state = InstallState::Installing(InstallAction::InstallingNam);
        self.install_log.clear();
    }

    /// Reinstall PyTorch with CUDA support using the wheel index detected by
    /// `detect_environment.py`. No-op if no CUDA install was detected.
    pub fn install_cuda_torch(&mut self) {
        let Some(ci) = self.cuda_install.clone() else {
            return;
        };
        self.install_rx = Some(crate::install_service::spawn_cuda_install(
            self.python_path.clone(),
            ci.clone(),
        ));
        self.install_state = InstallState::Installing(InstallAction::InstallingCudaTorch);
        self.install_log.clear();
        self.install_log.push(format!(
            "Reinstalling PyTorch with CUDA {} for {}...",
            ci.cuda_version,
            ci.gpu_names.join(", "),
        ));
    }

    /// Install Python via Miniforge into ~/miniforge3.
    pub fn install_python(&mut self) {
        self.install_state = InstallState::Installing(InstallAction::InstallingPython);
        self.install_log.clear();
        self.install_log
            .push("Installing Python via Miniforge...".into());

        let Some(install_dir) = managed_miniforge_dir() else {
            self.install_state = InstallState::Failed;
            self.install_log
                .push("Could not determine the user home directory.".into());
            return;
        };
        self.install_rx = Some(crate::install_service::spawn_miniforge_install(install_dir));
    }

    pub fn poll_install(&mut self) {
        let rx = match self.install_rx.take() {
            Some(rx) => rx,
            None => return,
        };

        let mut done = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                InstallMessage::Log(line) => {
                    self.push_install_log(line);
                }
                InstallMessage::PythonInstalled { python_path } => {
                    // Auto-select the newly installed Python
                    self.python_path = python_path.clone();
                    self.settings.python_path = Some(python_path);
                    self.persist_settings();
                    // Refresh the discovery list
                    self.discovered_pythons = None;
                    self.python_discovery_rx = None;
                }
                InstallMessage::Done { success } => {
                    if success {
                        self.install_state = InstallState::Done;
                        self.push_install_log("Installation complete!");
                        self.python_status = PythonStatus::Unknown;
                        self.check_python();
                    } else {
                        self.install_state = InstallState::Failed;
                        self.push_install_log("Installation failed.");
                    }
                    done = true;
                }
            }
        }

        let dropped = rx.take_dropped_progress();
        if dropped > 0 {
            self.push_install_log(format!(
                "{dropped} additional installer log lines were suppressed."
            ));
        }

        if !done {
            // Put the receiver back if we're not done yet
            self.install_rx = Some(rx);
        }
    }

    fn push_install_log(&mut self, line: impl Into<String>) {
        self.install_log.push(line.into());
        let excess = self.install_log.len().saturating_sub(MAX_INSTALL_LOG_LINES);
        if excess > 0 {
            self.install_log.drain(..excess);
        }
    }

    /// Remove ~/miniforge3 in a background thread with progress feedback.
    pub fn uninstall_miniforge(&mut self) {
        self.install_rx = Some(crate::install_service::spawn_miniforge_uninstall(
            managed_miniforge_dir(),
        ));
        self.install_state = InstallState::Installing(InstallAction::UninstallingMiniforge);
        self.install_log.clear();
        self.install_log
            .push("Removing ~/miniforge3 (includes NAM if installed there)...".into());

        // Reset to system python immediately so the UI updates
        let default_python = default_python_name();
        self.python_path = default_python.clone();
        self.settings.python_path = Some(default_python);
        self.persist_settings();
        self.discovered_pythons = None;
        self.python_status = PythonStatus::Unknown;
        self.check_python();
    }

    pub fn managed_miniforge_path() -> Option<std::path::PathBuf> {
        managed_miniforge_dir()
    }

    pub fn is_managed_miniforge_install() -> bool {
        managed_miniforge_dir()
            .map(|path| path.join(MANAGED_MINIFORGE_MARKER).is_file())
            .unwrap_or(false)
    }

    /// Uninstall neural-amp-modeler from the selected Python environment.
    pub fn uninstall_nam(&mut self) {
        self.install_rx = Some(crate::install_service::spawn_nam_uninstall(
            self.python_path.clone(),
        ));
        self.install_state = InstallState::Installing(InstallAction::UninstallingNam);
        self.install_log.clear();
        self.install_log
            .push("Uninstalling neural-amp-modeler...".into());
    }

    /// Start a demo training simulation (for testing the progress UI).
    pub fn start_demo_training(&mut self) {
        use crate::worker::{event_to_message, protocol};

        self.training_log.clear();
        self.epoch_history.clear();
        let data = self.run.data_mut();
        data.completed_models.clear();
        data.failed_files.clear();
        data.current_file_index = 0;
        data.total_files = 1;
        data.started_at = Some(Instant::now());
        data.last_epoch_at = None;
        data.avg_epoch_secs = None;
        self.push_training_log("Demo training started...");

        let (tx, rx) = crate::worker::worker_message_channel(64);
        let epochs = self.config.epochs;
        let cancelled = tx.cancellation_flag();
        let demo_cancelled = std::sync::Arc::clone(&cancelled);
        let join = std::thread::spawn(move || {
            let _ = tx.send(event_to_message(protocol::WorkerEvent::TrainingStart {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "demo".into(),
                file_index: Some(0),
                sequence: 1,
                file: "demo_output.wav".into(),
                total_epochs: epochs,
            }));
            for epoch in 1..=epochs {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if demo_cancelled.load(std::sync::atomic::Ordering::Acquire) {
                    return;
                }
                // Simulate decreasing loss with some noise
                let progress = epoch as f64 / epochs as f64;
                let noise = ((epoch as f64 * 7.3).sin() * 0.02).abs();
                let train_loss = 0.5 * (-3.0 * progress).exp() + noise;
                let val_loss = 0.55 * (-2.8 * progress).exp() + noise * 1.2;
                let esr = 0.4 * (-2.5 * progress).exp() + noise * 0.5;

                let _ = tx.send(event_to_message(protocol::WorkerEvent::EpochEnd {
                    protocol_version: protocol::PROTOCOL_VERSION,
                    run_id: "demo".into(),
                    file_index: Some(0),
                    sequence: u64::from(epoch) + 1,
                    epoch,
                    train_loss,
                    val_loss,
                    esr,
                }));
                let _ = tx.send(event_to_message(protocol::WorkerEvent::Log {
                    protocol_version: protocol::PROTOCOL_VERSION,
                    run_id: "demo".into(),
                    file_index: Some(0),
                    sequence: u64::from(epoch) + u64::from(epochs) + 1,
                    message: format!(
                        "Epoch {epoch}/{epochs}: train={train_loss:.6} val={val_loss:.6} ESR={esr:.6}"
                    ),
                }));
            }
            let _ = tx.send(event_to_message(protocol::WorkerEvent::TrainingComplete {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "demo".into(),
                file_index: Some(0),
                sequence: u64::from(epochs) * 2 + 2,
                file: "demo_output.wav".into(),
                validation_esr: 0.0,
                model_path: "/tmp/demo_model.nam".into(),
            }));
            let _ = tx.send(event_to_message(protocol::WorkerEvent::AllComplete {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "demo".into(),
                file_index: None,
                sequence: u64::from(epochs) * 2 + 3,
            }));
            let _ = tx.send(WorkerMessage::WorkerExited { exit_code: Some(0) });
        });
        let worker = WorkerHandle::from_join(cancelled, join);
        self.run.activate(Some(worker), rx);
    }

    pub fn can_train(&self) -> bool {
        self.input_path.is_some()
            && !self.output_paths.is_empty()
            && self.destination_dir.is_some()
            && self.run.state() == TrainingState::Idle
    }

    pub fn cancel_training(&mut self) {
        if let Some(worker) = self.run.worker_mut() {
            worker.cancel();
        }
        self.push_training_log("Training cancelled.");
        self.finalize_run(RunFinalization::Cancelled);
    }

    pub fn set_allow_overwrite_outputs(&mut self, allow: bool) {
        self.allow_overwrite_outputs = allow;
        self.save_config();
    }

    pub fn reset_after_error(&mut self) {
        let report = cleanup_run_resources(self.run.data().artifacts.as_ref());
        if !report.is_complete() {
            self.report_cleanup_failures(&report);
            return;
        }
        self.clear_active_run_settings();
        self.run.reset();
        self.epoch_history.clear();
        self.training_log.clear();
    }

    pub fn prepare_train_again(&mut self) {
        let report = cleanup_run_resources(self.run.data().artifacts.as_ref());
        if !report.is_complete() {
            self.report_cleanup_failures(&report);
            return;
        }
        self.clear_active_run_settings();
        self.run.reset();
        self.epoch_history.clear();
        self.training_log.clear();
        self.model_path = None;
    }

    pub fn diagnostics_text(&self) -> String {
        build_diagnostics_summary(self)
    }

    pub(crate) fn report_user_action_error(&mut self, action: &str, error: impl std::fmt::Display) {
        let message = format!("{action} failed: {error}");
        self.push_training_log(format!("Error: {message}"));
        self.user_action_error = Some(message);
    }

    pub(crate) fn report_cleanup_failures(&mut self, report: &CleanupReport) {
        for failure in &report.failures {
            self.push_training_log(format!(
                "Cleanup failed while attempting to {} {}: {}",
                failure.operation,
                failure.path.display(),
                failure.message
            ));
        }
        self.user_action_error = Some(format!(
            "Cleanup is incomplete for {} path(s). Recovery information was retained so cleanup can be retried.",
            report.failures.len()
        ));
    }

    pub fn push_training_log(&mut self, line: impl Into<String>) {
        let line = line.into();
        if let Some(ref run) = self.run.data().artifacts {
            if let Err(error) = run.append_log(&line) {
                if self.user_action_error.is_none() {
                    self.user_action_error = Some(format!("Training log write failed: {error}"));
                }
            }
        }
        self.training_log.push(line);
        if self.training_log.len() > MAX_TRAINING_LOG_LINES {
            let excess = self.training_log.len() - MAX_TRAINING_LOG_LINES;
            self.training_log.drain(0..excess);
        }
    }

    pub fn record_active_run_settings(&mut self) {
        if let Some(ref run) = self.run.data().artifacts {
            self.settings.active_run_id = Some(run.id.to_string());
            self.settings.active_run_log_path = Some(run.log_path.clone());
            self.settings.active_run_manifest_path = Some(run.manifest_path.clone());
            self.settings.active_run_staging_dir = Some(run.staging_dir().to_path_buf());
            self.settings.active_run_reserved_paths = run.reserved_paths().to_vec();
            self.persist_settings();
        }
    }

    pub fn clear_active_run_settings(&mut self) {
        self.settings.active_run_id = None;
        self.settings.active_run_log_path = None;
        self.settings.active_run_manifest_path = None;
        self.settings.active_run_staging_dir = None;
        self.settings.active_run_reserved_paths.clear();
        #[cfg(not(test))]
        self.persist_settings();
    }

    pub(crate) fn record_recent_successful_run(
        &mut self,
        model_path: &Path,
        manifest_path: PathBuf,
        final_esr: f64,
    ) {
        self.settings.recent_runs.insert(
            0,
            crate::settings::RecentRun {
                model_path: model_path.to_path_buf(),
                manifest_path,
                esr: Some(final_esr),
                architecture: self.config.architecture.as_str().to_string(),
                device: self.selected_device.to_string(),
                completed_unix_seconds: unix_timestamp_secs(),
            },
        );
        self.settings.recent_runs.truncate(20);
        #[cfg(not(test))]
        self.persist_settings();
    }
}

impl Drop for TrainerApp {
    fn drop(&mut self) {
        if let Some(run) = self.run.data().artifacts.as_ref() {
            let _ = run.flush_log();
        }
    }
}

// ── Desktop notifications ──────────────────────────────────────────────

pub(crate) fn send_notification(title: &str, body: &str) {
    let title = title.to_string();
    let body = body.to_string();
    // Run in background thread so it never blocks the UI
    std::thread::spawn(move || {
        let _ = SystemNotifier.notify(&title, &body);
    });
}

trait Notifier {
    fn notify(&self, title: &str, body: &str) -> Result<(), std::io::Error>;
}

struct SystemNotifier;

impl Notifier for SystemNotifier {
    fn notify(&self, title: &str, body: &str) -> Result<(), std::io::Error> {
        let spec = notification_command(NotificationPlatform::current(), title, body);
        std::process::Command::new(spec.program)
            .args(&spec.args)
            .hide_console()
            .output()?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum NotificationPlatform {
    MacOs,
    Windows,
    Linux,
}

impl NotificationPlatform {
    fn current() -> Self {
        if cfg!(target_os = "macos") {
            Self::MacOs
        } else if cfg!(target_os = "windows") {
            Self::Windows
        } else {
            Self::Linux
        }
    }
}

struct NotificationCommand {
    program: &'static str,
    args: Vec<String>,
}

fn notification_command(
    platform: NotificationPlatform,
    title: &str,
    body: &str,
) -> NotificationCommand {
    match platform {
        NotificationPlatform::MacOs => NotificationCommand {
            program: "osascript",
            args: vec![
                "-e".into(),
                format!(
                    "display notification \"{}\" with title \"{}\"",
                    body.replace('\"', "\\\""),
                    title.replace('\"', "\\\"")
                ),
            ],
        },
        NotificationPlatform::Windows => NotificationCommand {
            program: "powershell",
            args: vec![
                "-NoProfile".into(),
                "-Command".into(),
                format!(
                    "[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); \
                     $n = New-Object System.Windows.Forms.NotifyIcon; \
                     $n.Icon = [System.Drawing.SystemIcons]::Information; \
                     $n.BalloonTipTitle = '{}'; \
                     $n.BalloonTipText = '{}'; \
                     $n.Visible = $true; \
                     $n.ShowBalloonTip(5000); \
                     Start-Sleep -Seconds 6; \
                     $n.Dispose()",
                    title.replace('\'', "''"),
                    body.replace('\'', "''")
                ),
            ],
        },
        NotificationPlatform::Linux => NotificationCommand {
            program: "notify-send",
            args: vec![title.to_string(), body.to_string()],
        },
    }
}

// ── Platform helpers ────────────────────────────────────────────────────

/// Chainable helper that suppresses the child console window on Windows.
/// No-op on other platforms so call sites can be unconditional.
pub(crate) trait HideConsoleExt {
    fn hide_console(&mut self) -> &mut Self;
}

#[cfg(windows)]
impl HideConsoleExt for std::process::Command {
    fn hide_console(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        self.creation_flags(CREATE_NO_WINDOW)
    }
}

#[cfg(not(windows))]
impl HideConsoleExt for std::process::Command {
    fn hide_console(&mut self) -> &mut Self {
        self
    }
}

fn non_empty_opt(s: &str) -> Option<String> {
    if s.trim().is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

impl TrainerApp {
    fn poll_python_check(&mut self) {
        if let Some(ref rx) = self.python_check_rx {
            if let Ok(result) = rx.try_recv() {
                // Auto-select best device when status changes to Ok
                if let PythonStatus::Ok { ref devices, .. } = result.status {
                    let best = devices
                        .iter()
                        .find(|d| d.id.starts_with("cuda"))
                        .or_else(|| devices.iter().find(|d| d.id == "mps"))
                        .or(devices.first());
                    if let Some(dev) = best {
                        self.selected_device = dev.id.clone();
                    }
                }
                self.python_status = result.status;
                self.cuda_install = result.cuda_install;
                self.python_check_rx = None;
            }
        }
    }
}

impl eframe::App for TrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let worker_messages_pending = self.poll_worker();
        self.poll_python_check();
        self.poll_install();

        // Handle drag-and-drop of WAV files
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                let wav_files: Vec<PathBuf> = i
                    .raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| f.path.as_ref())
                    .filter(|p| {
                        p.extension()
                            .map(|e| e.eq_ignore_ascii_case("wav"))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect();

                if wav_files.len() == 1 && self.input_path.is_none() {
                    // Single WAV dropped with no input set: use as input
                    let p = wav_files[0].clone();
                    self.settings.last_input_path = Some(p.clone());
                    self.persist_settings();
                    self.input_path = Some(p);
                } else if !wav_files.is_empty() {
                    // Multiple WAVs or input already set: use as output
                    self.output_paths = wav_files;
                }
            }
        });

        // Update window title with training progress
        match self.run.state() {
            TrainingState::Training => {
                if let Some(last) = self.epoch_history.last() {
                    let pct = (last.epoch as f32 / self.config.epochs.max(1) as f32 * 100.0) as u32;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                        "NAM Trainer - {}% (ESR: {:.4})",
                        pct, last.esr
                    )));
                }
            }
            TrainingState::Complete => {
                let esr_str = self
                    .epoch_history
                    .last()
                    .map(|e| format!(" (ESR: {:.4})", e.esr))
                    .unwrap_or_default();
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                    "NAM Trainer - Complete{}",
                    esr_str
                )));
            }
            _ => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Title("NAM Trainer".to_string()));
            }
        }

        // Request continuous repaints while training or installing
        let needs_repaint = self.run.state() == TrainingState::Training
            || worker_messages_pending
            || matches!(self.install_state, InstallState::Installing(_))
            || matches!(self.python_status, PythonStatus::Unknown)
            || self.python_discovery_rx.is_some();
        if needs_repaint {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui::main_panel::show(self, ui);
            });
        });

        if self.show_advanced {
            ui::advanced_options::show(self, ctx);
        }
        if self.show_metadata {
            ui::metadata_panel::show(self, ctx);
        }
    }
}

#[cfg(test)]
#[path = "app_validation_tests.rs"]
mod validation_tests;

#[cfg(test)]
#[path = "app_run_tests.rs"]
mod run_tests;

#[cfg(test)]
#[path = "app_ui_tests.rs"]
mod ui_tests;
