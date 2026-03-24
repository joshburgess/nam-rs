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

    // Python discovery and environment status
    pub discovered_pythons: Option<Vec<crate::ui::main_panel::PythonEntry>>,
    pub python_status: PythonStatus,
    python_check_rx: Option<mpsc::Receiver<PythonStatus>>,
}

#[derive(Clone, Debug)]
pub enum PythonStatus {
    Unknown,
    Ok { gpu: Option<String> },
    Error(String),
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
                .unwrap_or_else(|| "python3".to_string()),
            discovered_pythons: None,
            python_status: PythonStatus::Unknown,
            python_check_rx: None,
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
            let script = r#"
import json, sys
result = {"nam": False, "gpu": None}
try:
    from nam.train import core
    result["nam"] = True
except ImportError:
    pass
try:
    import torch
    if torch.cuda.is_available():
        result["gpu"] = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        result["gpu"] = "mps"
except ImportError:
    pass
print(json.dumps(result))
"#;
            let output = std::process::Command::new(&python)
                .args(["-c", script])
                .output();

            let status = match output {
                Ok(out) if out.status.success() => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
                        let nam_ok = val.get("nam").and_then(|v| v.as_bool()).unwrap_or(false);
                        let gpu = val.get("gpu").and_then(|v| v.as_str()).map(String::from);
                        if nam_ok {
                            PythonStatus::Ok { gpu }
                        } else {
                            PythonStatus::Error(
                                "NAM not installed. Run: pip install neural-amp-modeler".into(),
                            )
                        }
                    } else {
                        PythonStatus::Error("Unexpected Python output".into())
                    }
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    PythonStatus::Error(format!("Python error: {}", stderr.lines().next().unwrap_or("unknown")))
                }
                Err(e) => PythonStatus::Error(format!("Cannot run '{}': {}", python, e)),
            };
            let _ = tx.send(status);
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
                    self.training_log
                        .push(format!("Training complete! ESR={:.6} Model: {}", esr, model_path));
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

    fn poll_python_check(&mut self) {
        if let Some(ref rx) = self.python_check_rx {
            if let Ok(status) = rx.try_recv() {
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

        egui::CentralPanel::default().show(ctx, |ui| {
            ui::main_panel::show(self, ui);
        });

        if self.show_advanced {
            ui::advanced_options::show(self, ctx);
        }
        if self.show_metadata {
            ui::metadata_panel::show(self, ctx);
        }
    }
}
