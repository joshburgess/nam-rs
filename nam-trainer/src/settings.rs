use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Settings {
    pub last_input_path: Option<String>,
    pub last_destination: Option<String>,
    pub python_path: Option<String>,
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
