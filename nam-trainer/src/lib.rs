mod app;
mod artifacts;
mod background;
mod diagnostics;
mod environment;
mod environment_service;
mod errors;
mod install_service;
mod persistence;
mod recovery;
mod run_controller;
mod run_manifest;
mod settings;
mod training_validation;
mod ui;
mod validation;
mod worker;
mod worker_reducer;

use std::path::{Path, PathBuf};

pub use app::{Architecture, DeviceId, ModelMetadata, TrainerApp, TrainingConfig};
pub use artifacts::{sanitize_model_basename, TrainingRunArtifacts};
pub use errors::{classify_training_error, TrainingErrorDetails, TrainingErrorKind};
pub use run_manifest::validate_training_manifest_json;
pub use settings::validate_settings_json;
pub use validation::{validate_audio_files, ValidationIssue, ValidationSeverity};
pub use worker::protocol::{
    MetadataRequest, TrainRequest, WorkerErrorKind, WorkerEvent, PROTOCOL_VERSION,
};
pub use worker::TrainingState;

pub fn configure_smoke_test(report_path: PathBuf) -> std::io::Result<()> {
    write_smoke_test_report(&report_path)
}

fn write_smoke_test_report(path: &Path) -> std::io::Result<()> {
    std::fs::write(path, nam_core::build_info::json())
}

pub fn decode_worker_event_json(json: &str) -> serde_json::Result<WorkerEvent> {
    serde_json::from_str(json)
}

#[cfg(test)]
mod build_tests {
    #[test]
    fn smoke_test_report_contains_embedded_metadata() -> std::io::Result<()> {
        let directory = tempfile::tempdir()?;
        let report = directory.path().join("smoke.json");
        super::write_smoke_test_report(&report)?;
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(report)?)?;
        assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
        assert!(value["git_commit"].is_string());
        assert!(value["target"].is_string());
        assert!(value["features"].is_array());
        Ok(())
    }
}
