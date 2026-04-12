use std::sync::mpsc;

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

    // Persistent settings
    pub settings: Settings,

    // Python executable path
    pub python_path: String,

    // Device selection
    pub selected_device: String, // "cpu", "cuda:0", "mps", etc.

    // Python discovery and environment status
    pub discovered_pythons: Option<Vec<crate::ui::main_panel::PythonEntry>>,
    pub python_status: PythonStatus,
    python_check_rx: Option<mpsc::Receiver<PythonStatus>>,

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
    UninstallingNam,
    UninstallingMiniforge,
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
    pub val_loss: f64,
    pub esr: f64,
}

impl TrainerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let settings = Settings::load();
        let mut app = Self {
            input_path: settings.last_input_path.clone(),
            output_paths: Vec::new(),
            destination_dir: settings.last_destination.clone(),
            config: TrainingConfig::default(),
            metadata: ModelMetadata::default(),
            show_advanced: false,
            show_metadata: false,
            training_state: TrainingState::Idle,
            training_log: Vec::new(),
            epoch_history: Vec::new(),
            worker: None,
            message_rx: None,
            python_path: settings
                .python_path
                .clone()
                .unwrap_or_else(default_python_name),
            selected_device: "cpu".to_string(),
            discovered_pythons: None,
            python_status: PythonStatus::Unknown,
            python_check_rx: None,
            install_state: InstallState::Idle,
            install_log: Vec::new(),
            install_rx: None,
            settings,
        };
        app.check_python();
        app
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
                .output();

            let status = match output {
                Ok(out) if out.status.success() => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
                        let version = val
                            .get("version")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();

                        // Check Python version meets minimum
                        let version_ok = parse_version_tuple(&version)
                            .map(|(maj, min)| (maj, min) >= NAM_MIN_PYTHON)
                            .unwrap_or(false);

                        if !version_ok {
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
                        }
                    } else {
                        PythonStatus::Error("Unexpected Python output".into())
                    }
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let stderr_lower = stderr.to_lowercase();
                    // Windows Store alias and similar "not a real Python" cases
                    if stderr_lower.contains("was not found")
                        || stderr_lower.contains("not recognized")
                        || stderr_lower.contains("not found")
                    {
                        PythonStatus::NotFound
                    } else {
                        PythonStatus::Error(format!(
                            "Python error: {}",
                            stderr.lines().next().unwrap_or("unknown")
                        ))
                    }
                }
                Err(_) => PythonStatus::NotFound,
            };
            let _ = tx.send(status);
        });
    }

    /// Install neural-amp-modeler into the selected Python environment.
    pub fn install_nam(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.install_rx = Some(rx);
        self.install_state = InstallState::Installing(InstallAction::InstallingNam);
        self.install_log.clear();
        self.install_log
            .push("Installing neural-amp-modeler...".into());

        let python = self.python_path.clone();

        std::thread::spawn(move || {
            let result = std::process::Command::new(&python)
                .args(["-m", "pip", "install", "neural-amp-modeler"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn();

            let mut child = match result {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(InstallMessage::Log(format!("Failed to run pip: {e}")));
                    let _ = tx.send(InstallMessage::Done { success: false });
                    return;
                }
            };

            // Stream stderr (pip writes progress there)
            if let Some(stderr) = child.stderr.take() {
                use std::io::{BufRead, BufReader};
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = tx.send(InstallMessage::Log(line));
                }
            }

            let status = child.wait();
            let success = status.map(|s| s.success()).unwrap_or(false);
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
        let install_dir = format!("{home}/miniforge3");

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
                esr: 0.01,
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
                WorkerMessage::EpochEnd {
                    epoch,
                    train_loss,
                    val_loss,
                    esr,
                } => {
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
                WorkerMessage::TrainingComplete { model_path, esr } => {
                    self.training_log.push(format!(
                        "Training complete! ESR={:.6} Model: {}",
                        esr, model_path
                    ));
                    self.training_state = TrainingState::Complete;
                    self.worker = None;
                }
                WorkerMessage::Error(err) => {
                    self.training_log.push(format!("Error: {}", err));
                    self.training_state = TrainingState::Error(err);
                    self.worker = None;
                }
                WorkerMessage::WorkerExited => {
                    if self.training_state == TrainingState::Training {
                        self.training_state =
                            TrainingState::Error("Worker process exited unexpectedly".into());
                    }
                    self.worker = None;
                }
            }
        }
    }
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
            .spawn()
    } else {
        std::process::Command::new("curl")
            .args(["-fSL", "-o", dest, url])
            .stderr(std::process::Stdio::piped())
            .spawn()
    }
}

/// Run the Miniforge installer in silent/batch mode.
fn run_miniforge_installer(
    installer_path: &str,
    install_dir: &str,
) -> Result<std::process::Child, std::io::Error> {
    if cfg!(target_os = "windows") {
        // Windows .exe installer with /S (silent) /D=path (destination)
        std::process::Command::new(installer_path)
            .args([
                "/S",
                "/InstallationType=JustMe",
                "/RegisterPython=0",
                "/AddToPath=0",
                &format!("/D={}", install_dir),
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
    } else {
        std::process::Command::new("bash")
            .args([installer_path, "-b", "-u", "-p", install_dir])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
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
            if let Ok(status) = rx.try_recv() {
                // Auto-select best device when status changes to Ok
                if let PythonStatus::Ok { ref devices, .. } = status {
                    let best = devices
                        .iter()
                        .find(|d| d.id.starts_with("cuda"))
                        .or_else(|| devices.iter().find(|d| d.id == "mps"))
                        .or(devices.first());
                    if let Some(dev) = best {
                        self.selected_device = dev.id.clone();
                    }
                }
                self.python_status = status;
                self.python_check_rx = None;
            }
        }
    }
}

impl eframe::App for TrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        self.poll_python_check();
        self.poll_install();

        // Request continuous repaints while training or installing
        let needs_repaint = self.training_state == TrainingState::Training
            || matches!(self.install_state, InstallState::Installing(_))
            || matches!(self.python_status, PythonStatus::Unknown);
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
