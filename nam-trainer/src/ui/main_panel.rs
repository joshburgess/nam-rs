use crate::app::TrainerApp;
use crate::worker::{self, TrainingState};

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    ui.heading("NAM Trainer");

    // Python/GPU status line
    ui.horizontal(|ui| {
        match &app.python_status {
            crate::app::PythonStatus::Unknown => {
                ui.colored_label(egui::Color32::GRAY, "Checking Python...");
            }
            crate::app::PythonStatus::Ok { gpu } => {
                let gpu_label = match gpu.as_deref() {
                    Some("cuda") => "CUDA",
                    Some("mps") => "MPS",
                    _ => "CPU only",
                };
                ui.colored_label(egui::Color32::GREEN, format!("Python OK ({gpu_label})"));
            }
            crate::app::PythonStatus::Error(msg) => {
                ui.colored_label(egui::Color32::RED, format!("Python: {msg}"));
            }
        }
    });

    ui.add_space(4.0);

    // Input audio
    ui.horizontal(|ui| {
        if ui.button("Input Audio...").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("WAV files", &["wav"])
                .pick_file()
            {
                let p = path.display().to_string();
                app.settings.last_input_path = Some(p.clone());
                app.settings.save();
                app.input_path = Some(p);
            }
        }
        if let Some(ref p) = app.input_path {
            ui.label(truncate_path(p, 50));
        } else {
            ui.colored_label(egui::Color32::GRAY, "No file selected");
        }
        if ui.small_button("Download standard input").clicked() {
            let _ = open::that(
                "https://drive.google.com/drive/folders/1cZTalC-9sSugQn3pn5JGBxuUxJbmHEoZ",
            );
        }
    });

    // Output audio (multi-select)
    ui.horizontal(|ui| {
        if ui.button("Output Audio...").clicked() {
            if let Some(paths) = rfd::FileDialog::new()
                .add_filter("WAV files", &["wav"])
                .pick_files()
            {
                app.output_paths = paths.iter().map(|p| p.display().to_string()).collect();
            }
        }
        if app.output_paths.is_empty() {
            ui.colored_label(egui::Color32::GRAY, "No file(s) selected");
        } else if app.output_paths.len() == 1 {
            ui.label(truncate_path(&app.output_paths[0], 50));
        } else {
            ui.label(format!("{} files selected", app.output_paths.len()));
        }
    });

    // Destination directory
    ui.horizontal(|ui| {
        if ui.button("Train Destination...").clicked() {
            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                let p = path.display().to_string();
                app.settings.last_destination = Some(p.clone());
                app.settings.save();
                app.destination_dir = Some(p);
            }
        }
        if let Some(ref p) = app.destination_dir {
            ui.label(truncate_path(p, 50));
        } else {
            ui.colored_label(egui::Color32::GRAY, "No directory selected");
        }
    });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    // Option buttons
    ui.horizontal(|ui| {
        if ui.button("Advanced Options").clicked() {
            app.show_advanced = !app.show_advanced;
        }
        if ui.button("Metadata").clicked() {
            app.show_metadata = !app.show_metadata;
        }
    });

    ui.add_space(4.0);

    // Python executable selector
    // Auto-discover on first frame
    if app.discovered_pythons.is_none() {
        app.discovered_pythons = Some(discover_pythons());
    }

    ui.horizontal(|ui| {
        ui.label("Python:");

        let discovered = app.discovered_pythons.as_ref().cloned().unwrap_or_default();

        // Show combo box with discovered Pythons + current selection + "Browse..." option
        let current_label = if app.python_path.is_empty() {
            "(select)".to_string()
        } else {
            // Show just the path, truncated
            truncate_path(&app.python_path, 45)
        };

        let mut changed = false;
        egui::ComboBox::from_id_salt("python_combo")
            .selected_text(current_label)
            .width(350.0)
            .show_ui(ui, |ui| {
                for entry in &discovered {
                    let label = format!("{} ({})", entry.label, entry.path);
                    if ui
                        .selectable_value(&mut app.python_path, entry.path.clone(), label)
                        .changed()
                    {
                        changed = true;
                    }
                }
                ui.separator();
                if ui.selectable_label(false, "Browse for Python executable...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                        app.python_path = path.display().to_string();
                        changed = true;
                    }
                }
            });

        if changed {
            app.settings.python_path = Some(app.python_path.clone());
            app.settings.save();
            app.python_status = crate::app::PythonStatus::Unknown;
            app.check_python();
        }
    });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    // Train / Cancel button
    match &app.training_state {
        TrainingState::Idle => {
            let can_train = app.can_train();
            if ui
                .add_enabled(can_train, egui::Button::new("Train"))
                .clicked()
            {
                start_training(app);
            }
        }
        TrainingState::Training => {
            if ui.button("Cancel").clicked() {
                if let Some(ref mut w) = app.worker {
                    w.cancel();
                }
                app.training_state = TrainingState::Idle;
                app.worker = None;
                app.message_rx = None;
                app.training_log.push("Training cancelled.".into());
            }
            ui.spinner();
            if let Some(last) = app.epoch_history.last() {
                ui.label(format!(
                    "Epoch {}/{} - ESR: {:.6}",
                    last.epoch, app.config.epochs, last.esr
                ));
            }
        }
        TrainingState::Complete => {
            ui.colored_label(egui::Color32::GREEN, "Training complete!");
            if ui.button("Train Again").clicked() {
                app.training_state = TrainingState::Idle;
                app.epoch_history.clear();
                app.training_log.clear();
            }
        }
        TrainingState::Error(msg) => {
            ui.colored_label(egui::Color32::RED, format!("Error: {}", msg));
            if ui.button("Reset").clicked() {
                app.training_state = TrainingState::Idle;
                app.epoch_history.clear();
                app.training_log.clear();
            }
        }
    }

    ui.add_space(8.0);

    // Training log / progress
    if !app.training_log.is_empty() {
        super::progress::show(app, ui);
    }
}

fn start_training(app: &mut TrainerApp) {
    app.training_state = TrainingState::Training;
    app.training_log.clear();
    app.epoch_history.clear();
    app.training_log.push("Starting training...".into());

    let (handle, rx) = worker::spawn(app);
    app.worker = Some(handle);
    app.message_rx = Some(rx);
}

#[derive(Clone)]
pub struct PythonEntry {
    pub label: String,
    pub path: String,
}

fn discover_pythons() -> Vec<PythonEntry> {
    let mut found: Vec<PythonEntry> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Candidates to search for on PATH
    let candidates = ["python3", "python"];

    for name in &candidates {
        if let Ok(output) = std::process::Command::new("which").arg(name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                // Resolve symlinks to get the real path
                let resolved = std::fs::canonicalize(&path)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| path.clone());
                if !seen.contains(&resolved) {
                    seen.insert(resolved.clone());
                    // Try to get version
                    let version = std::process::Command::new(&path)
                        .args(["--version"])
                        .output()
                        .ok()
                        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                        .unwrap_or_default();
                    let label = if version.is_empty() {
                        name.to_string()
                    } else {
                        version
                    };
                    found.push(PythonEntry {
                        label,
                        path: path.clone(),
                    });
                }
            }
        }
    }

    // Look for conda environments
    if let Some(home) = dirs_for_home() {
        let conda_dirs = [
            home.join("miniconda3").join("envs"),
            home.join("anaconda3").join("envs"),
            home.join("miniforge3").join("envs"),
            home.join(".conda").join("envs"),
        ];
        for envs_dir in &conda_dirs {
            if let Ok(entries) = std::fs::read_dir(envs_dir) {
                for entry in entries.flatten() {
                    let env_python = entry.path().join("bin").join("python");
                    if env_python.exists() {
                        let path = env_python.display().to_string();
                        let resolved = std::fs::canonicalize(&path)
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| path.clone());
                        if !seen.contains(&resolved) {
                            seen.insert(resolved);
                            let env_name = entry.file_name().to_string_lossy().to_string();
                            found.push(PythonEntry {
                                label: format!("conda: {}", env_name),
                                path,
                            });
                        }
                    }
                }
            }
        }
    }

    found
}

fn dirs_for_home() -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    {
        std::env::var("HOME").ok().map(std::path::PathBuf::from)
    }
    #[cfg(not(unix))]
    {
        std::env::var("USERPROFILE").ok().map(std::path::PathBuf::from)
    }
}

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}
