use std::sync::mpsc;
use std::time::Instant;

use crate::settings::Settings;
use crate::ui;
use crate::worker::{TrainingState, WorkerHandle, WorkerMessage};

/// Top-level application state.
pub struct TrainerApp {
    // File paths
    pub input_path: Option<String>,
    pub output_paths: Vec<String>,
    pub destination_dir: Option<String>,

    // Training configuration
    pub config: TrainingConfig,
    pub metadata: ModelMetadata,

    // Sub-window visibility
    pub show_advanced: bool,
    pub show_metadata: bool,

    // Training state
    pub training_state: TrainingState,
    pub training_log: Vec<String>,
    pub epoch_history: Vec<EpochStats>,
    pub worker: Option<WorkerHandle>,
    pub message_rx: Option<mpsc::Receiver<WorkerMessage>>,
    pub model_path: Option<String>,

    // Batch training progress (file N of M)
    pub current_file_index: usize,
    pub total_files: usize,

    // ETA tracking
    pub training_start_time: Option<Instant>,
    pub last_epoch_time: Option<Instant>,
    pub avg_epoch_secs: Option<f64>,

    // Persistent settings
    pub settings: Settings,

    // Python executable path
    pub python_path: String,

    // Device selection
    pub selected_device: String, // "cpu", "cuda:0", "mps", etc.

    // Python discovery and environment status
    pub discovered_pythons: Option<Vec<crate::ui::main_panel::PythonEntry>>,
    pub python_discovery_rx: Option<mpsc::Receiver<Vec<crate::ui::main_panel::PythonEntry>>>,
    pub python_status: PythonStatus,
    pub cuda_install: Option<CudaInstall>,
    python_check_rx: Option<mpsc::Receiver<DetectionResult>>,

    // NAM install state
    pub install_state: InstallState,
    pub install_log: Vec<String>,
    pub install_rx: Option<mpsc::Receiver<InstallMessage>>,
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

#[derive(Clone, Debug)]
pub enum PythonStatus {
    Unknown,
    Ok {
        version: String,
        devices: Vec<TrainingDevice>,
        warnings: Vec<String>,
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
    pub id: String,   // "cpu", "cuda:0", "cuda:1", "mps"
    pub name: String, // "CPU", "CUDA 0: NVIDIA RTX 4090", "Apple GPU (MPS)"
}

#[derive(Debug)]
pub enum InstallMessage {
    Log(String),
    PythonInstalled { python_path: String },
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
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Standard,
    Lite,
    Feather,
    Nano,
}

impl Architecture {
    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Lite => "Lite",
            Self::Feather => "Feather",
            Self::Nano => "Nano",
        }
    }

    pub fn all() -> &'static [Architecture] {
        &[Self::Standard, Self::Lite, Self::Feather, Self::Nano]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Lite => "lite",
            Self::Feather => "feather",
            Self::Nano => "nano",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "lite" => Self::Lite,
            "feather" => Self::Feather,
            "nano" => Self::Nano,
            _ => Self::Standard,
        }
    }

    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Standard => "Best quality, largest model, slowest to train (~30 min for 100 epochs on GPU)",
            Self::Lite => "Good quality with faster training and smaller model size",
            Self::Feather => "Lightweight model for low-latency use, trades some accuracy for speed",
            Self::Nano => "Smallest and fastest model, best for quick tests or low-power devices",
        }
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

    pub fn from_str(s: &str) -> Option<Self> {
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

    pub fn from_str(s: &str) -> Option<Self> {
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
        let settings = Settings::load();

        // Restore training config from settings, falling back to defaults
        let mut config = TrainingConfig::default();
        if let Some(ref arch_str) = settings.architecture {
            config.architecture = Architecture::from_str(arch_str);
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

        // Restore metadata from settings
        let metadata = ModelMetadata {
            name: settings.meta_name.clone().unwrap_or_default(),
            modeled_by: settings.meta_modeled_by.clone().unwrap_or_default(),
            gear_make: settings.meta_gear_make.clone().unwrap_or_default(),
            gear_model: settings.meta_gear_model.clone().unwrap_or_default(),
            gear_type: settings
                .meta_gear_type
                .as_deref()
                .and_then(GearType::from_str),
            tone_type: settings
                .meta_tone_type
                .as_deref()
                .and_then(ToneType::from_str),
            input_level_dbu: settings.meta_input_level_dbu.clone().unwrap_or_default(),
            output_level_dbu: settings.meta_output_level_dbu.clone().unwrap_or_default(),
        };

        let mut app = Self {
            input_path: settings.last_input_path.clone(),
            output_paths: Vec::new(),
            destination_dir: settings.last_destination.clone(),
            config,
            metadata,
            show_advanced: false,
            show_metadata: false,
            training_state: TrainingState::Idle,
            training_log: Vec::new(),
            epoch_history: Vec::new(),
            worker: None,
            message_rx: None,
            model_path: None,
            current_file_index: 0,
            total_files: 0,
            training_start_time: None,
            last_epoch_time: None,
            avg_epoch_secs: None,
            python_path: settings
                .python_path
                .clone()
                .unwrap_or_else(default_python_name),
            selected_device: "cpu".to_string(),
            discovered_pythons: None,
            python_discovery_rx: None,
            python_status: PythonStatus::Unknown,
            cuda_install: None,
            python_check_rx: None,
            install_state: InstallState::Idle,
            install_log: Vec::new(),
            install_rx: None,
            settings,
        };
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
        self.settings.save();
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
        self.settings.save();
    }

    /// Spawn a background thread to verify Python + NAM are available and detect GPU.
    pub fn check_python(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.python_check_rx = Some(rx);
        let python = self.python_path.clone();

        std::thread::spawn(move || {
            let script = include_str!("../python/detect_environment.py");
            let output = std::process::Command::new(&python)
                .args(["-c", script])
                .hide_console()
                .output();

            let result = match output {
                Ok(out) if out.status.success() => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
                        let version = val
                            .get("version")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();

                        let cuda_install = parse_cuda_install(&val);

                        // Check Python version meets minimum
                        let version_ok = parse_version_tuple(&version)
                            .map(|(maj, min)| (maj, min) >= NAM_MIN_PYTHON)
                            .unwrap_or(false);

                        let status = if !version_ok {
                            PythonStatus::VersionTooOld { version }
                        } else {
                            let nam_ok = val.get("nam").and_then(|v| v.as_bool()).unwrap_or(false);

                            // Parse devices list
                            let devices: Vec<TrainingDevice> = val
                                .get("devices")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|d| {
                                            let id = d.get("id")?.as_str()?.to_string();
                                            let name = d.get("name")?.as_str()?.to_string();
                                            Some(TrainingDevice { id, name })
                                        })
                                        .collect()
                                })
                                .unwrap_or_else(|| {
                                    vec![TrainingDevice {
                                        id: "cpu".into(),
                                        name: "CPU".into(),
                                    }]
                                });

                            let warnings: Vec<String> = val
                                .get("warnings")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|w| w.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default();

                            if nam_ok {
                                PythonStatus::Ok {
                                    version,
                                    devices,
                                    warnings,
                                }
                            } else {
                                PythonStatus::Error("NAM not installed".into())
                            }
                        };
                        DetectionResult {
                            status,
                            cuda_install,
                        }
                    } else {
                        DetectionResult {
                            status: PythonStatus::Error("Unexpected Python output".into()),
                            cuda_install: None,
                        }
                    }
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let stderr_lower = stderr.to_lowercase();
                    // Windows Store alias and similar "not a real Python" cases
                    let status = if stderr_lower.contains("was not found")
                        || stderr_lower.contains("not recognized")
                        || stderr_lower.contains("not found")
                    {
                        PythonStatus::NotFound
                    } else {
                        PythonStatus::Error(format!(
                            "Python error: {}",
                            stderr.lines().next().unwrap_or("unknown")
                        ))
                    };
                    DetectionResult {
                        status,
                        cuda_install: None,
                    }
                }
                Err(_) => DetectionResult {
                    status: PythonStatus::NotFound,
                    cuda_install: None,
                },
            };
            let _ = tx.send(result);
        });
    }

    /// Install neural-amp-modeler into the selected Python environment.
    /// If an NVIDIA GPU was detected, installs a CUDA-enabled PyTorch wheel
    /// first so the user gets a working GPU setup from a single button click.
    pub fn install_nam(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.install_rx = Some(rx);
        self.install_state = InstallState::Installing(InstallAction::InstallingNam);
        self.install_log.clear();

        let python = self.python_path.clone();
        let cuda_install = self.cuda_install.clone();

        std::thread::spawn(move || {
            if let Some(ref ci) = cuda_install {
                let _ = tx.send(InstallMessage::Log(format!(
                    "NVIDIA GPU detected ({}). Installing PyTorch with CUDA {} first...",
                    ci.gpu_names.join(", "),
                    ci.cuda_version,
                )));
                let torch_args = [
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "--index-url",
                    ci.wheel_index.as_str(),
                ];
                if !run_pip(&python, &torch_args, &tx) {
                    let _ = tx.send(InstallMessage::Log(
                        "PyTorch CUDA install failed. Aborting.".into(),
                    ));
                    let _ = tx.send(InstallMessage::Done { success: false });
                    return;
                }
            }

            let _ = tx.send(InstallMessage::Log(
                "Installing neural-amp-modeler...".into(),
            ));
            let success = run_pip(
                &python,
                &["-m", "pip", "install", "neural-amp-modeler"],
                &tx,
            );
            let _ = tx.send(InstallMessage::Done { success });
        });
    }

    /// Reinstall PyTorch with CUDA support using the wheel index detected by
    /// `detect_environment.py`. No-op if no CUDA install was detected.
    pub fn install_cuda_torch(&mut self) {
        let Some(ci) = self.cuda_install.clone() else {
            return;
        };
        let (tx, rx) = mpsc::channel();
        self.install_rx = Some(rx);
        self.install_state = InstallState::Installing(InstallAction::InstallingCudaTorch);
        self.install_log.clear();
        self.install_log.push(format!(
            "Reinstalling PyTorch with CUDA {} for {}...",
            ci.cuda_version,
            ci.gpu_names.join(", "),
        ));

        let python = self.python_path.clone();
        std::thread::spawn(move || {
            let args = [
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "torch",
                "--index-url",
                ci.wheel_index.as_str(),
            ];
            let success = run_pip(&python, &args, &tx);
            let _ = tx.send(InstallMessage::Done { success });
        });
    }

    /// Install Python via Miniforge into ~/miniforge3.
    pub fn install_python(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.install_rx = Some(rx);
        self.install_state = InstallState::Installing(InstallAction::InstallingPython);
        self.install_log.clear();
        self.install_log
            .push("Installing Python via Miniforge...".into());

        let home = home_dir_string();
        let install_dir = std::path::PathBuf::from(&home)
            .join("miniforge3")
            .to_string_lossy()
            .into_owned();

        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};

            // Platform-specific installer URL and file extension
            let (installer_url, installer_ext) = miniforge_installer_info();
            let installer_url = match installer_url {
                Some(url) => url,
                None => {
                    let _ = tx.send(InstallMessage::Log(
                        "Automatic Python install is not supported on this platform.".into(),
                    ));
                    let _ = tx.send(InstallMessage::Done { success: false });
                    return;
                }
            };

            let installer_path = format!(
                "{}/miniforge_installer{}",
                std::env::temp_dir().display(),
                installer_ext
            );

            // Step 1: Download
            let _ = tx.send(InstallMessage::Log(format!(
                "Downloading Miniforge from {installer_url}..."
            )));

            let dl = download_file(installer_url, &installer_path);
            let mut dl_child = match dl {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(InstallMessage::Log(format!(
                        "Failed to start download: {e}"
                    )));
                    let _ = tx.send(InstallMessage::Done { success: false });
                    return;
                }
            };

            if let Some(stderr) = dl_child.stderr.take() {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = tx.send(InstallMessage::Log(line));
                }
            }

            let dl_status = dl_child.wait();
            if !dl_status.map(|s| s.success()).unwrap_or(false) {
                let _ = tx.send(InstallMessage::Log("Download failed.".into()));
                let _ = tx.send(InstallMessage::Done { success: false });
                return;
            }
            let _ = tx.send(InstallMessage::Log("Download complete.".into()));

            // Step 2: Run installer in batch mode
            let _ = tx.send(InstallMessage::Log(format!(
                "Installing to {install_dir}..."
            )));
            let install = run_miniforge_installer(&installer_path, &install_dir);

            let mut inst_child = match install {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(InstallMessage::Log(format!("Failed to run installer: {e}")));
                    let _ = tx.send(InstallMessage::Done { success: false });
                    return;
                }
            };

            // Stream stdout
            if let Some(stdout) = inst_child.stdout.take() {
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = tx.send(InstallMessage::Log(line));
                }
            }

            let inst_status = inst_child.wait();
            if !inst_status.map(|s| s.success()).unwrap_or(false) {
                let _ = tx.send(InstallMessage::Log("Miniforge installation failed.".into()));
                let _ = tx.send(InstallMessage::Done { success: false });
                return;
            }

            // Clean up installer
            let _ = std::fs::remove_file(&installer_path);

            let python_path = miniforge_python_path(&install_dir);
            let _ = tx.send(InstallMessage::Log(format!(
                "Python installed at {python_path}"
            )));

            // Send special done message with the new python path
            let _ = tx.send(InstallMessage::PythonInstalled { python_path });
            let _ = tx.send(InstallMessage::Done { success: true });
        });
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
                    self.install_log.push(line);
                }
                InstallMessage::PythonInstalled { python_path } => {
                    // Auto-select the newly installed Python
                    self.python_path = python_path.clone();
                    self.settings.python_path = Some(python_path);
                    self.settings.save();
                    // Refresh the discovery list
                    self.discovered_pythons = None;
                    self.python_discovery_rx = None;
                }
                InstallMessage::Done { success } => {
                    if success {
                        self.install_state = InstallState::Done;
                        self.install_log.push("Installation complete!".into());
                        self.python_status = PythonStatus::Unknown;
                        self.check_python();
                    } else {
                        self.install_state = InstallState::Failed;
                        self.install_log.push("Installation failed.".into());
                    }
                    done = true;
                }
            }
        }

        if !done {
            // Put the receiver back if we're not done yet
            self.install_rx = Some(rx);
        }
    }

    /// Remove ~/miniforge3 in a background thread with progress feedback.
    pub fn uninstall_miniforge(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.install_rx = Some(rx);
        self.install_state = InstallState::Installing(InstallAction::UninstallingMiniforge);
        self.install_log.clear();
        self.install_log
            .push("Removing ~/miniforge3 (includes NAM if installed there)...".into());

        std::thread::spawn(move || {
            let home = home_dir_string();
            let miniforge_dir = std::path::PathBuf::from(&home).join("miniforge3");

            if miniforge_dir.exists() {
                let _ = tx.send(InstallMessage::Log(format!(
                    "Deleting {}...",
                    miniforge_dir.display()
                )));
                match std::fs::remove_dir_all(&miniforge_dir) {
                    Ok(_) => {
                        let _ = tx.send(InstallMessage::Log(
                            "Miniforge removed successfully.".into(),
                        ));
                        let _ = tx.send(InstallMessage::Done { success: true });
                    }
                    Err(e) => {
                        let _ = tx.send(InstallMessage::Log(format!("Failed to remove: {e}")));
                        let _ = tx.send(InstallMessage::Done { success: false });
                    }
                }
            } else {
                let _ = tx.send(InstallMessage::Log("~/miniforge3 does not exist.".into()));
                let _ = tx.send(InstallMessage::Done { success: true });
            }
        });

        // Reset to system python immediately so the UI updates
        let default_python = default_python_name();
        self.python_path = default_python.clone();
        self.settings.python_path = Some(default_python);
        self.settings.save();
        self.discovered_pythons = None;
        self.python_status = PythonStatus::Unknown;
        self.check_python();
    }

    /// Uninstall neural-amp-modeler from the selected Python environment.
    pub fn uninstall_nam(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.install_rx = Some(rx);
        self.install_state = InstallState::Installing(InstallAction::UninstallingNam);
        self.install_log.clear();
        self.install_log
            .push("Uninstalling neural-amp-modeler...".into());

        let python = self.python_path.clone();

        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};

            let result = std::process::Command::new(&python)
                .args(["-m", "pip", "uninstall", "-y", "neural-amp-modeler"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .hide_console()
                .spawn();

            let mut child = match result {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(InstallMessage::Log(format!("Failed to run pip: {e}")));
                    let _ = tx.send(InstallMessage::Done { success: false });
                    return;
                }
            };

            if let Some(stderr) = child.stderr.take() {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = tx.send(InstallMessage::Log(line));
                }
            }

            let status = child.wait();
            let success = status.map(|s| s.success()).unwrap_or(false);
            if success {
                let _ = tx.send(InstallMessage::Log("NAM uninstalled successfully.".into()));
            }
            let _ = tx.send(InstallMessage::Done { success });
        });
    }

    /// Start a demo training simulation (for testing the progress UI).
    pub fn start_demo_training(&mut self) {
        use crate::worker::{TrainingState, WorkerMessage};

        self.training_state = TrainingState::Training;
        self.training_log.clear();
        self.epoch_history.clear();
        self.training_log.push("Demo training started...".into());

        let (tx, rx) = mpsc::channel();
        self.message_rx = Some(rx);

        let epochs = self.config.epochs;
        std::thread::spawn(move || {
            for epoch in 1..=epochs {
                std::thread::sleep(std::time::Duration::from_millis(200));
                // Simulate decreasing loss with some noise
                let progress = epoch as f64 / epochs as f64;
                let noise = ((epoch as f64 * 7.3).sin() * 0.02).abs();
                let train_loss = 0.5 * (-3.0 * progress).exp() + noise;
                let val_loss = 0.55 * (-2.8 * progress).exp() + noise * 1.2;
                let esr = 0.4 * (-2.5 * progress).exp() + noise * 0.5;

                let _ = tx.send(WorkerMessage::EpochEnd {
                    epoch,
                    train_loss,
                    val_loss,
                    esr,
                });
                let _ = tx.send(WorkerMessage::Log(format!(
                    "Epoch {epoch}/{epochs}: train={train_loss:.6} val={val_loss:.6} ESR={esr:.6}"
                )));
            }
            let _ = tx.send(WorkerMessage::TrainingComplete {
                model_path: "/tmp/demo_model.nam".into(),
            });
        });
    }

    pub fn can_train(&self) -> bool {
        self.input_path.is_some()
            && !self.output_paths.is_empty()
            && self.destination_dir.is_some()
            && self.training_state == TrainingState::Idle
    }

    /// Drain worker messages and update state.
    pub fn poll_worker(&mut self) {
        let rx = match self.message_rx.as_ref() {
            Some(rx) => rx,
            None => return,
        };

        while let Ok(msg) = rx.try_recv() {
            match msg {
                WorkerMessage::Log(text) => {
                    self.training_log.push(text);
                }
                WorkerMessage::TrainingStart { ref file, total_epochs } => {
                    self.current_file_index += 1;
                    self.training_log.push(format!(
                        "Training {} ({} epochs)...",
                        file, total_epochs
                    ));
                }
                WorkerMessage::EpochEnd {
                    epoch,
                    train_loss,
                    val_loss,
                    esr,
                } => {
                    // Update ETA tracking
                    let now = Instant::now();
                    if let Some(last) = self.last_epoch_time {
                        let elapsed = now.duration_since(last).as_secs_f64();
                        // Exponential moving average for smoother estimates
                        self.avg_epoch_secs = Some(match self.avg_epoch_secs {
                            Some(avg) => avg * 0.7 + elapsed * 0.3,
                            None => elapsed,
                        });
                    }
                    self.last_epoch_time = Some(now);

                    self.epoch_history.push(EpochStats {
                        epoch,
                        train_loss,
                        val_loss,
                        esr,
                    });
                    self.training_log.push(format!(
                        "Epoch {}: loss={:.6} val_loss={:.6} ESR={:.6}",
                        epoch, train_loss, val_loss, esr
                    ));
                }
                WorkerMessage::TrainingComplete { model_path } => {
                    let final_esr = self
                        .epoch_history
                        .last()
                        .map(|e| e.esr)
                        .unwrap_or(0.0);
                    self.training_log.push(format!(
                        "Training complete! ESR={:.6} Model: {}",
                        final_esr, model_path
                    ));
                    self.model_path = Some(model_path);
                    self.training_state = TrainingState::Complete;
                    self.worker = None;

                    // Send desktop notification
                    send_notification(
                        "NAM Trainer",
                        &format!("Training complete! (ESR: {:.6})", final_esr),
                    );
                }
                WorkerMessage::Error(err) => {
                    self.training_log.push(format!("Error: {}", err));
                    self.training_state = TrainingState::Error(err);
                    self.worker = None;
                }
                WorkerMessage::WorkerExited { exit_code } => {
                    if self.training_state == TrainingState::Training {
                        let msg = match exit_code {
                            Some(code) => {
                                format!("Worker process exited unexpectedly (exit code {code})")
                            }
                            None => "Worker process exited unexpectedly".into(),
                        };
                        self.training_state = TrainingState::Error(msg);
                    }
                    self.worker = None;
                }
            }
        }
    }
}

// ── Audio validation ───────────────────────────────────────────────────

/// Validate that audio files are suitable for training. Returns a list of
/// warnings/errors. An empty list means everything looks good.
pub fn validate_audio_files(
    input_path: &str,
    output_paths: &[String],
) -> Vec<String> {
    let mut issues = Vec::new();

    let input_spec = match hound::WavReader::open(input_path) {
        Ok(r) => r.spec(),
        Err(e) => {
            issues.push(format!("Cannot read input file: {e}"));
            return issues;
        }
    };

    let input_duration = match hound::WavReader::open(input_path) {
        Ok(r) => {
            let samples = r.len() as f64;
            let channels = input_spec.channels as f64;
            let rate = input_spec.sample_rate as f64;
            samples / channels / rate
        }
        Err(_) => 0.0,
    };

    if input_duration < 1.0 {
        issues.push(format!(
            "Input file is very short ({:.1}s). Training needs at least a few seconds of audio.",
            input_duration
        ));
    }

    for output_path in output_paths {
        let basename = std::path::Path::new(output_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| output_path.clone());

        let output_spec = match hound::WavReader::open(output_path) {
            Ok(r) => r.spec(),
            Err(e) => {
                issues.push(format!("{basename}: cannot read file: {e}"));
                continue;
            }
        };

        let output_duration = match hound::WavReader::open(output_path) {
            Ok(r) => {
                let samples = r.len() as f64;
                let channels = output_spec.channels as f64;
                let rate = output_spec.sample_rate as f64;
                samples / channels / rate
            }
            Err(_) => 0.0,
        };

        if output_spec.sample_rate != input_spec.sample_rate {
            issues.push(format!(
                "{basename}: sample rate {}Hz does not match input ({}Hz)",
                output_spec.sample_rate, input_spec.sample_rate
            ));
        }

        if output_duration < 1.0 {
            issues.push(format!(
                "{basename}: very short ({:.1}s)",
                output_duration
            ));
        }

        let ratio = if input_duration > 0.0 {
            output_duration / input_duration
        } else {
            1.0
        };
        if ratio < 0.5 || ratio > 2.0 {
            issues.push(format!(
                "{basename}: duration ({:.1}s) differs significantly from input ({:.1}s)",
                output_duration, input_duration
            ));
        }
    }

    issues
}

// ── Desktop notifications ──────────────────────────────────────────────

fn send_notification(title: &str, body: &str) {
    let title = title.to_string();
    let body = body.to_string();
    // Run in background thread so it never blocks the UI
    std::thread::spawn(move || {
        let _ = send_notification_sync(&title, &body);
    });
}

fn send_notification_sync(title: &str, body: &str) -> Result<(), std::io::Error> {
    if cfg!(target_os = "macos") {
        std::process::Command::new("osascript")
            .args([
                "-e",
                &format!(
                    "display notification \"{}\" with title \"{}\"",
                    body.replace('\"', "\\\""),
                    title.replace('\"', "\\\"")
                ),
            ])
            .hide_console()
            .output()?;
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
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
            ])
            .hide_console()
            .output()?;
    } else {
        // Linux: notify-send
        std::process::Command::new("notify-send")
            .args([title, body])
            .hide_console()
            .output()?;
    }
    Ok(())
}

// ── Platform helpers ────────────────────────────────────────────────────

fn home_dir_string() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/tmp".into())
}

fn default_python_name() -> String {
    if cfg!(target_os = "windows") {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

/// Returns (Option<url>, file_extension) for the Miniforge installer.
fn miniforge_installer_info() -> (Option<&'static str>, &'static str) {
    if cfg!(target_os = "macos") {
        let url = if cfg!(target_arch = "aarch64") {
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
        } else {
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
        };
        (Some(url), ".sh")
    } else if cfg!(target_os = "linux") {
        let url = if cfg!(target_arch = "aarch64") {
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
        } else {
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        };
        (Some(url), ".sh")
    } else if cfg!(target_os = "windows") {
        (
            Some("https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"),
            ".exe",
        )
    } else {
        (None, "")
    }
}

/// Download a file using curl (macOS/Linux) or PowerShell (Windows).
fn download_file(url: &str, dest: &str) -> Result<std::process::Child, std::io::Error> {
    if cfg!(target_os = "windows") {
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing",
                    url, dest
                ),
            ])
            .stderr(std::process::Stdio::piped())
            .hide_console()
            .spawn()
    } else {
        std::process::Command::new("curl")
            .args(["-fSL", "-o", dest, url])
            .stderr(std::process::Stdio::piped())
            .hide_console()
            .spawn()
    }
}

/// Run the Miniforge installer in silent/batch mode.
fn run_miniforge_installer(
    installer_path: &str,
    install_dir: &str,
) -> Result<std::process::Child, std::io::Error> {
    if cfg!(target_os = "windows") {
        // Run the NSIS installer through PowerShell with -WindowStyle Hidden
        // so any sub-process console windows the installer spawns stay hidden.
        // Direct invocation with CREATE_NO_WINDOW only hides the parent.
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-WindowStyle",
                "Hidden",
                "-Command",
                &format!(
                    "Start-Process -FilePath '{}' -ArgumentList '/S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D={}' -Wait -WindowStyle Hidden",
                    installer_path, install_dir
                ),
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .hide_console()
            .spawn()
    } else {
        std::process::Command::new("bash")
            .args([installer_path, "-b", "-u", "-p", install_dir])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .hide_console()
            .spawn()
    }
}

/// Returns the path to the python executable inside a miniforge install.
fn miniforge_python_path(install_dir: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{install_dir}\\python.exe")
    } else {
        format!("{install_dir}/bin/python")
    }
}

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

fn parse_version_tuple(version: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() >= 2 {
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        Some((major, minor))
    } else {
        None
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

/// Run `python <args>` with stderr streamed to `tx` as install log lines.
/// Returns true on exit-zero, false otherwise. Used by install_nam and
/// install_cuda_torch to share streaming + error handling.
fn run_pip(python: &str, args: &[&str], tx: &mpsc::Sender<InstallMessage>) -> bool {
    let spawn_result = std::process::Command::new(python)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .hide_console()
        .spawn();

    let mut child = match spawn_result {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(InstallMessage::Log(format!("Failed to run pip: {e}")));
            return false;
        }
    };

    if let Some(stderr) = child.stderr.take() {
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            let _ = tx.send(InstallMessage::Log(line));
        }
    }

    child.wait().map(|s| s.success()).unwrap_or(false)
}

fn parse_cuda_install(val: &serde_json::Value) -> Option<CudaInstall> {
    let ci = val.get("cuda_install")?;
    if ci.is_null() {
        return None;
    }
    let cuda_version = ci.get("cuda_version")?.as_str()?.to_string();
    let wheel_index = ci.get("wheel_index")?.as_str()?.to_string();
    let gpu_names = ci
        .get("gpu_names")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|n| n.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    Some(CudaInstall {
        cuda_version,
        wheel_index,
        gpu_names,
    })
}

impl eframe::App for TrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        self.poll_python_check();
        self.poll_install();

        // Handle drag-and-drop of WAV files
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                let wav_files: Vec<String> = i
                    .raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| f.path.as_ref())
                    .filter(|p| {
                        p.extension()
                            .map(|e| e.eq_ignore_ascii_case("wav"))
                            .unwrap_or(false)
                    })
                    .map(|p| p.display().to_string())
                    .collect();

                if wav_files.len() == 1 && self.input_path.is_none() {
                    // Single WAV dropped with no input set: use as input
                    let p = wav_files[0].clone();
                    self.settings.last_input_path = Some(p.clone());
                    self.settings.save();
                    self.input_path = Some(p);
                } else if !wav_files.is_empty() {
                    // Multiple WAVs or input already set: use as output
                    self.output_paths = wav_files;
                }
            }
        });

        // Update window title with training progress
        match &self.training_state {
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
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(
                    "NAM Trainer".to_string(),
                ));
            }
        }

        // Request continuous repaints while training or installing
        let needs_repaint = self.training_state == TrainingState::Training
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
