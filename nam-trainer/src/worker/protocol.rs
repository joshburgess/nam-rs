use serde::{Deserialize, Serialize};

/// Request sent from Rust GUI to Python worker via stdin (single JSON line).
#[derive(Serialize)]
pub struct TrainRequest {
    pub input_path: String,
    pub output_paths: Vec<String>,
    pub destination: String,
    pub architecture: String,
    pub packed: bool,
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub lr_decay: f64,
    pub latency: Option<i32>,
    pub threshold_esr: Option<f64>,
    pub save_plot: bool,
    pub fit_mrstft: bool,
    pub ignore_checks: bool,
    pub num_output_samples_per_datum: u32,
    pub use_full_config_trainer: bool,
    pub device: String,
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
        #[allow(dead_code)]
        file: String,
        #[allow(dead_code)]
        validation_esr: f64,
        model_path: String,
    },

    #[serde(rename = "training_failed")]
    TrainingFailed {
        #[allow(dead_code)] // present in JSON, matched with `..`
        file: String,
        error: String,
    },

    #[serde(rename = "all_complete")]
    AllComplete,

    #[serde(rename = "error")]
    Error { message: String },

    #[serde(rename = "log")]
    Log { message: String },
}

#[cfg(test)]
mod tests {
    use super::{MetadataRequest, TrainRequest};

    #[test]
    fn train_request_serializes_packed_flag() {
        let request = TrainRequest {
            input_path: "input.wav".into(),
            output_paths: vec!["output.wav".into()],
            destination: "models".into(),
            architecture: "packed".into(),
            packed: true,
            epochs: 1,
            batch_size: 2,
            lr: 0.004,
            lr_decay: 0.007,
            latency: None,
            threshold_esr: None,
            save_plot: true,
            fit_mrstft: true,
            ignore_checks: false,
            num_output_samples_per_datum: 8192,
            use_full_config_trainer: false,
            device: "cpu".into(),
            metadata: MetadataRequest {
                name: None,
                modeled_by: None,
                gear_make: None,
                gear_model: None,
                gear_type: None,
                tone_type: None,
                input_level_dbu: None,
                output_level_dbu: None,
            },
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["architecture"], "packed");
        assert_eq!(json["packed"], true);
        assert_eq!(json["ignore_checks"], false);
        assert_eq!(json["num_output_samples_per_datum"], 8192);
        assert_eq!(json["use_full_config_trainer"], false);
    }
}
