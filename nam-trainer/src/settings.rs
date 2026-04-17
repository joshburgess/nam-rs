use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Settings {
    pub last_input_path: Option<String>,
    pub last_destination: Option<String>,
    pub python_path: Option<String>,

    // Training configuration (persisted across restarts)
    pub architecture: Option<String>,
    pub epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub lr: Option<f64>,
    pub lr_decay: Option<f64>,
    pub latency: Option<i32>,
    pub threshold_esr: Option<f64>,
    pub save_plot: Option<bool>,

    // Model metadata (persisted across restarts)
    pub meta_name: Option<String>,
    pub meta_modeled_by: Option<String>,
    pub meta_gear_make: Option<String>,
    pub meta_gear_model: Option<String>,
    pub meta_gear_type: Option<String>,
    pub meta_tone_type: Option<String>,
    pub meta_input_level_dbu: Option<String>,
    pub meta_output_level_dbu: Option<String>,
}

impl Settings {
    fn config_path() -> Option<PathBuf> {
        let dirs = directories::ProjectDirs::from("com", "nam-rs", "nam-trainer")?;
        Some(dirs.config_dir().join("settings.json"))
    }

    pub fn load() -> Self {
        Self::config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}
