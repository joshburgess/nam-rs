use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const SETTINGS_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub schema_version: u32,
    pub last_input_path: Option<PathBuf>,
    pub last_destination: Option<PathBuf>,
    pub python_path: Option<PathBuf>,

    // Training configuration (persisted across restarts)
    pub architecture: Option<String>,
    pub epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub lr: Option<f64>,
    pub lr_decay: Option<f64>,
    pub latency: Option<i32>,
    pub threshold_esr: Option<f64>,
    pub save_plot: Option<bool>,
    pub ignore_checks: Option<bool>,
    pub num_output_samples_per_datum: Option<u32>,
    pub use_full_config_trainer: Option<bool>,
    pub allow_overwrite_outputs: Option<bool>,
    pub output_model_basename: Option<String>,
    pub batch_name_template: Option<String>,

    // Model metadata (persisted across restarts)
    pub meta_name: Option<String>,
    pub meta_modeled_by: Option<String>,
    pub meta_gear_make: Option<String>,
    pub meta_gear_model: Option<String>,
    pub meta_gear_type: Option<String>,
    pub meta_tone_type: Option<String>,
    pub meta_input_level_dbu: Option<String>,
    pub meta_output_level_dbu: Option<String>,

    // Active run recovery. These are left populated while a worker is running
    // so a restarted GUI can point the user back to the last run artifacts.
    pub active_run_id: Option<String>,
    pub active_run_log_path: Option<PathBuf>,
    pub active_run_manifest_path: Option<PathBuf>,
    pub active_run_staging_dir: Option<PathBuf>,
    pub active_run_reserved_paths: Vec<PathBuf>,

    #[serde(default)]
    pub recent_runs: Vec<RecentRun>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            schema_version: SETTINGS_SCHEMA_VERSION,
            last_input_path: None,
            last_destination: None,
            python_path: None,
            architecture: None,
            epochs: None,
            batch_size: None,
            lr: None,
            lr_decay: None,
            latency: None,
            threshold_esr: None,
            save_plot: None,
            ignore_checks: None,
            num_output_samples_per_datum: None,
            use_full_config_trainer: None,
            allow_overwrite_outputs: None,
            output_model_basename: None,
            batch_name_template: None,
            meta_name: None,
            meta_modeled_by: None,
            meta_gear_make: None,
            meta_gear_model: None,
            meta_gear_type: None,
            meta_tone_type: None,
            meta_input_level_dbu: None,
            meta_output_level_dbu: None,
            active_run_id: None,
            active_run_log_path: None,
            active_run_manifest_path: None,
            active_run_staging_dir: None,
            active_run_reserved_paths: Vec::new(),
            recent_runs: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RecentRun {
    pub model_path: PathBuf,
    pub manifest_path: PathBuf,
    pub esr: Option<f64>,
    pub architecture: String,
    pub device: String,
    pub completed_unix_seconds: u64,
}

impl Settings {
    fn config_path() -> std::io::Result<PathBuf> {
        directories::ProjectDirs::from("com", "nam-rs", "nam-trainer")
            .map(|dirs| dirs.config_dir().join("settings.json"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "could not determine the settings directory",
                )
            })
    }

    pub fn load() -> std::io::Result<Self> {
        Self::load_from_path(&Self::config_path()?)
    }

    #[cfg_attr(test, allow(dead_code))]
    pub fn save(&self) -> std::io::Result<()> {
        self.save_to_path(&Self::config_path()?)
    }

    fn load_from_path(path: &std::path::Path) -> std::io::Result<Self> {
        let json = match std::fs::read_to_string(path) {
            Ok(json) => json,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(error),
        };
        Self::from_json(&json)
    }

    fn from_json(json: &str) -> std::io::Result<Self> {
        let mut settings: Self = serde_json::from_str(json).map_err(std::io::Error::other)?;
        if settings.schema_version > SETTINGS_SCHEMA_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "settings schema {} is newer than supported schema {}",
                    settings.schema_version, SETTINGS_SCHEMA_VERSION
                ),
            ));
        }
        settings.schema_version = SETTINGS_SCHEMA_VERSION;
        Ok(settings)
    }

    fn save_to_path(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut settings = self.clone();
        settings.schema_version = SETTINGS_SCHEMA_VERSION;
        let json = serde_json::to_vec_pretty(&settings).map_err(std::io::Error::other)?;
        crate::persistence::atomic_write(path, &json)
    }
}

pub fn validate_settings_json(json: &str) -> std::io::Result<()> {
    Settings::from_json(json).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::{Settings, SETTINGS_SCHEMA_VERSION};
    use proptest::prelude::*;

    #[test]
    fn trainer_option_settings_round_trip() {
        let settings = Settings {
            architecture: Some("packed".into()),
            ignore_checks: Some(true),
            num_output_samples_per_datum: Some(4096),
            use_full_config_trainer: Some(true),
            allow_overwrite_outputs: Some(true),
            output_model_basename: Some("custom-model".into()),
            batch_name_template: Some("{index}-{stem}".into()),
            active_run_id: Some("run-1".into()),
            active_run_log_path: Some("/tmp/run.training.log".into()),
            active_run_manifest_path: Some("/tmp/run.training-manifest.json".into()),
            recent_runs: vec![super::RecentRun {
                model_path: "/tmp/model.nam".into(),
                manifest_path: "/tmp/model.training-manifest.json".into(),
                esr: Some(0.123),
                architecture: "packed".into(),
                device: "cpu".into(),
                completed_unix_seconds: 123,
            }],
            ..Settings::default()
        };

        let json = serde_json::to_string(&settings).unwrap();
        let restored: Settings = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.architecture.as_deref(), Some("packed"));
        assert_eq!(restored.ignore_checks, Some(true));
        assert_eq!(restored.num_output_samples_per_datum, Some(4096));
        assert_eq!(restored.use_full_config_trainer, Some(true));
        assert_eq!(restored.allow_overwrite_outputs, Some(true));
        assert_eq!(
            restored.output_model_basename.as_deref(),
            Some("custom-model")
        );
        assert_eq!(
            restored.batch_name_template.as_deref(),
            Some("{index}-{stem}")
        );
        assert_eq!(restored.active_run_id.as_deref(), Some("run-1"));
        assert_eq!(
            restored.active_run_log_path.as_deref(),
            Some(std::path::Path::new("/tmp/run.training.log"))
        );
        assert_eq!(
            restored.active_run_manifest_path.as_deref(),
            Some(std::path::Path::new("/tmp/run.training-manifest.json"))
        );
        assert_eq!(restored.recent_runs.len(), 1);
        assert_eq!(restored.recent_runs[0].architecture, "packed");
    }

    #[test]
    fn legacy_settings_without_schema_version_migrate() {
        let (_temp, path) = unique_settings_path("legacy");
        std::fs::write(
            &path,
            r#"{"architecture":"packed","epochs":42,"recent_runs":[]}"#,
        )
        .unwrap();

        let settings = Settings::load_from_path(&path).unwrap();

        assert_eq!(settings.schema_version, SETTINGS_SCHEMA_VERSION);
        assert_eq!(settings.architecture.as_deref(), Some("packed"));
        assert_eq!(settings.epochs, Some(42));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn future_settings_schema_is_rejected() {
        let (_temp, path) = unique_settings_path("future");
        std::fs::write(
            &path,
            format!(
                r#"{{"schema_version":{},"recent_runs":[]}}"#,
                SETTINGS_SCHEMA_VERSION + 1
            ),
        )
        .unwrap();

        let error = Settings::load_from_path(&path).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn settings_save_replaces_existing_file_atomically() {
        let (_temp, path) = unique_settings_path("atomic");
        let first = Settings {
            epochs: Some(10),
            ..Settings::default()
        };
        first.save_to_path(&path).unwrap();
        let second = Settings {
            epochs: Some(20),
            ..Settings::default()
        };

        second.save_to_path(&path).unwrap();

        assert_eq!(Settings::load_from_path(&path).unwrap().epochs, Some(20));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_settings_are_rejected_without_panicking() {
        let error = Settings::from_json("{\"schema_version\":").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
    }

    proptest! {
        #[test]
        fn legacy_settings_migrate_arbitrary_supported_values(
            epochs in any::<u32>(),
            architecture in proptest::option::of("[a-z_]{0,24}"),
            unknown_value in any::<i64>(),
        ) {
            let json = serde_json::json!({
                "epochs": epochs,
                "architecture": architecture,
                "future_unknown_field": unknown_value,
            })
            .to_string();

            let migrated = Settings::from_json(&json).unwrap();

            prop_assert_eq!(migrated.schema_version, SETTINGS_SCHEMA_VERSION);
            prop_assert_eq!(migrated.epochs, Some(epochs));
            prop_assert_eq!(migrated.architecture, architecture);
        }
    }

    fn unique_settings_path(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::Builder::new()
            .prefix(&format!("nam-trainer-settings-{name}-"))
            .tempdir()
            .unwrap();
        let path = temp.path().join("settings.json");
        (temp, path)
    }
}
