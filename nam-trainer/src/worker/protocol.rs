#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Request sent from Rust GUI to Python worker via stdin (single JSON line).
#[derive(Serialize)]
pub struct TrainRequest {
    pub input_path: String,
    pub output_paths: Vec<String>,
    pub destination: String,
    pub architecture: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub lr_decay: f64,
    pub latency: Option<i32>,
    pub threshold_esr: Option<f64>,
    pub save_plot: bool,
    pub fit_mrstft: bool,
    pub metadata: MetadataRequest,
}

#[derive(Serialize)]
pub struct MetadataRequest {
    pub name: Option<String>,
    pub modeled_by: Option<String>,
    pub gear_make: Option<String>,
    pub gear_model: Option<String>,
    pub gear_type: Option<String>,
    pub tone_type: Option<String>,
    pub input_level_dbu: Option<f64>,
    pub output_level_dbu: Option<f64>,
}

/// Events sent from Python worker to Rust GUI via stdout (one JSON line per event).
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum WorkerEvent {
    #[serde(rename = "training_start")]
    TrainingStart { file: String, total_epochs: u32 },

    #[serde(rename = "epoch_end")]
    EpochEnd {
        epoch: u32,
        train_loss: f64,
        val_loss: f64,
        esr: f64,
    },

    #[serde(rename = "training_complete")]
    TrainingComplete {
        file: String,
        validation_esr: f64,
        model_path: String,
    },

    #[serde(rename = "training_failed")]
    TrainingFailed { file: String, error: String },

    #[serde(rename = "all_complete")]
    AllComplete,

    #[serde(rename = "error")]
    Error { message: String },

    #[serde(rename = "log")]
    Log { message: String },
}
