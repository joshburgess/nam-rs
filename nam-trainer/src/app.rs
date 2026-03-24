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
        Self {
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
            settings,
        }
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
}

impl eframe::App for TrainerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();

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
