use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WorkerErrorKind {
    Dependency,
    DataValidation,
    Device,
    UserCancel,
    Subprocess,
    Training,
    Protocol,
}

/// Request sent from Rust GUI to Python worker via stdin (single JSON line).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrainRequest {
    pub protocol_version: u32,
    pub run_id: String,
    pub input_path: String,
    pub output_paths: Vec<String>,
    pub destination: String,
    pub output_model_basename: Option<String>,
    pub batch_name_template: Option<String>,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerEvent {
    #[serde(rename = "training_start")]
    TrainingStart {
        protocol_version: u32,
        run_id: String,
        file_index: Option<usize>,
        sequence: u64,
        file: String,
        total_epochs: u32,
    },

    #[serde(rename = "epoch_end")]
    EpochEnd {
        protocol_version: u32,
        run_id: String,
        file_index: Option<usize>,
        sequence: u64,
        epoch: u32,
        train_loss: f64,
        val_loss: f64,
        esr: f64,
    },

    #[serde(rename = "training_complete")]
    TrainingComplete {
        protocol_version: u32,
        run_id: String,
        file_index: Option<usize>,
        sequence: u64,
        file: String,
        validation_esr: f64,
        model_path: String,
    },

    #[serde(rename = "training_failed")]
    TrainingFailed {
        protocol_version: u32,
        run_id: String,
        file_index: Option<usize>,
        sequence: u64,
        file: String,
        error_kind: WorkerErrorKind,
        error: String,
    },

    #[serde(rename = "all_complete")]
    AllComplete {
        protocol_version: u32,
        run_id: String,
        file_index: Option<usize>,
        sequence: u64,
    },

    #[serde(rename = "error")]
    Error {
        protocol_version: u32,
        run_id: String,
        file_index: Option<usize>,
        sequence: u64,
        error_kind: WorkerErrorKind,
        message: String,
    },

    #[serde(rename = "log")]
    Log {
        protocol_version: u32,
        run_id: String,
        file_index: Option<usize>,
        sequence: u64,
        message: String,
    },
}

impl WorkerEvent {
    pub fn protocol_version(&self) -> u32 {
        match self {
            Self::TrainingStart {
                protocol_version, ..
            }
            | Self::EpochEnd {
                protocol_version, ..
            }
            | Self::TrainingComplete {
                protocol_version, ..
            }
            | Self::TrainingFailed {
                protocol_version, ..
            }
            | Self::AllComplete {
                protocol_version, ..
            }
            | Self::Error {
                protocol_version, ..
            }
            | Self::Log {
                protocol_version, ..
            } => *protocol_version,
        }
    }

    pub fn run_id(&self) -> &str {
        match self {
            Self::TrainingStart { run_id, .. }
            | Self::EpochEnd { run_id, .. }
            | Self::TrainingComplete { run_id, .. }
            | Self::TrainingFailed { run_id, .. }
            | Self::AllComplete { run_id, .. }
            | Self::Error { run_id, .. }
            | Self::Log { run_id, .. } => run_id,
        }
    }

    pub fn file_index(&self) -> Option<usize> {
        match self {
            Self::TrainingStart { file_index, .. }
            | Self::EpochEnd { file_index, .. }
            | Self::TrainingComplete { file_index, .. }
            | Self::TrainingFailed { file_index, .. }
            | Self::AllComplete { file_index, .. }
            | Self::Error { file_index, .. }
            | Self::Log { file_index, .. } => *file_index,
        }
    }

    pub fn sequence(&self) -> u64 {
        match self {
            Self::TrainingStart { sequence, .. }
            | Self::EpochEnd { sequence, .. }
            | Self::TrainingComplete { sequence, .. }
            | Self::TrainingFailed { sequence, .. }
            | Self::AllComplete { sequence, .. }
            | Self::Error { sequence, .. }
            | Self::Log { sequence, .. } => *sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MetadataRequest, TrainRequest, WorkerEvent, PROTOCOL_VERSION};

    #[test]
    fn train_request_serializes_packed_flag() {
        let request = TrainRequest {
            protocol_version: PROTOCOL_VERSION,
            run_id: "run-test".into(),
            input_path: "input.wav".into(),
            output_paths: vec!["output.wav".into()],
            destination: "models".into(),
            output_model_basename: None,
            batch_name_template: None,
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

        let actual = serde_json::to_value(&request).unwrap();
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/train_request_packed.json"
        ))
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn worker_epoch_event_fixture_deserializes() {
        let event: WorkerEvent = serde_json::from_str(include_str!(
            "../../tests/fixtures/worker_epoch_end_event.json"
        ))
        .unwrap();

        match event {
            WorkerEvent::EpochEnd {
                protocol_version,
                epoch,
                train_loss,
                val_loss,
                esr,
                ..
            } => {
                assert_eq!(protocol_version, PROTOCOL_VERSION);
                assert_eq!(epoch, 3);
                assert_eq!(train_loss, 0.125);
                assert_eq!(val_loss, 0.25);
                assert_eq!(esr, 0.375);
            }
            _ => panic!("fixture should deserialize to epoch_end"),
        }
    }
}
