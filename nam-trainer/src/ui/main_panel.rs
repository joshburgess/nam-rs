use crate::app::TrainerApp;
use crate::worker::{self, TrainingState};

const BUTTON_WIDTH: f32 = 130.0;

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.heading("NAM Trainer");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Python/GPU status badge
            // Show install_state spinner during active installs, otherwise show python_status
            let (min_maj, min_min) = crate::app::NAM_MIN_PYTHON;
            let installing = app.install_state == crate::app::InstallState::Installing;

            if installing {
                ui.spinner();
                ui.colored_label(
                    egui::Color32::from_rgb(255, 180, 60),
                    "Installing...",
                );
            } else {
                match &app.python_status {
                    crate::app::PythonStatus::Unknown => {
                        ui.spinner();
                        ui.colored_label(egui::Color32::GRAY, "Checking Python...");
                    }
                    crate::app::PythonStatus::Ok { gpu, version } => {
                        let gpu_label = match gpu.as_deref() {
                            Some("cuda") => "CUDA GPU",
                            Some("mps") => "Apple GPU",
                            _ => "CPU",
                        };
                        ui.colored_label(
                            egui::Color32::from_rgb(80, 200, 120),
                            format!("Ready — Python {version}, {gpu_label}"),
                        );
                    }
                    crate::app::PythonStatus::VersionTooOld { version } => {
                        let clicked = ui
                            .button("Install Python")
                            .on_hover_text(
                                "Downloads and installs Miniforge (Python 3.12+) to ~/miniforge3",
                            )
                            .clicked();
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 100, 100),
                            format!("Python {version} — requires {min_maj}.{min_min}+"),
                        );
                        if clicked {
                            app.install_python();
                        }
                    }
                    crate::app::PythonStatus::Error(msg) => {
                        if msg.contains("not installed") {
                            let clicked = ui
                                .button("Install NAM")
                                .on_hover_text("Runs: pip install neural-amp-modeler")
                                .clicked();
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 180, 60),
                                "NAM not installed",
                            );
                            if clicked {
                                app.install_nam();
                            }
                        } else {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 100, 100),
                                msg.as_str(),
                            );
                        }
                    }
                }
            }
        });
    });

    ui.add_space(6.0);

    // ── Files section ───────────────────────────────────────────────────
    egui::Frame::group(ui.style())
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.strong("Audio Files");
            ui.add_space(4.0);

            // Input audio
            ui.horizontal(|ui| {
                let btn = egui::Button::new("Input Audio...")
                    .min_size(egui::vec2(BUTTON_WIDTH, 0.0));
                if ui.add(btn).clicked() {
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
                    ui.label(file_name(p));
                } else {
                    ui.colored_label(egui::Color32::GRAY, "No file selected");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("Download standard input")
                        .on_hover_text("Opens the NAM input files on Google Drive")
                        .clicked()
                    {
                        let _ = open::that(
                            "https://drive.google.com/drive/folders/1cZTalC-9sSugQn3pn5JGBxuUxJbmHEoZ",
                        );
                    }
                });
            });

            // Output audio
            ui.horizontal(|ui| {
                let btn = egui::Button::new("Output Audio...")
                    .min_size(egui::vec2(BUTTON_WIDTH, 0.0));
                if ui.add(btn).clicked() {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("WAV files", &["wav"])
                        .pick_files()
                    {
                        app.output_paths =
                            paths.iter().map(|p| p.display().to_string()).collect();
                    }
                }
                match app.output_paths.len() {
                    0 => {
                        ui.colored_label(egui::Color32::GRAY, "No file(s) selected");
                    }
                    1 => {
                        ui.label(file_name(&app.output_paths[0]));
                    }
                    n => {
                        ui.label(format!("{n} files selected"));
                    }
                }
            });

            // Destination
            ui.horizontal(|ui| {
                let btn = egui::Button::new("Output Directory...")
                    .min_size(egui::vec2(BUTTON_WIDTH, 0.0));
                if ui.add(btn).clicked() {
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
        });

    ui.add_space(6.0);

    // ── Configuration section ───────────────────────────────────────────
    egui::Frame::group(ui.style())
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.strong("Configuration");
            ui.add_space(4.0);

            // Architecture quick selector (inline, no popup needed for this)
            ui.horizontal(|ui| {
                ui.label("Model:");
                for &arch in crate::app::Architecture::all() {
                    ui.selectable_value(&mut app.config.architecture, arch, arch.label());
                }
            });

            ui.horizontal(|ui| {
                ui.label(format!("Epochs: {}", app.config.epochs));
                ui.add_space(16.0);
                ui.label(format!("Batch size: {}", app.config.batch_size));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Metadata...").clicked() {
                        app.show_metadata = !app.show_metadata;
                    }
                    if ui.button("Advanced...").clicked() {
                        app.show_advanced = !app.show_advanced;
                    }
                });
            });
        });

    ui.add_space(6.0);

    // ── Python section ──────────────────────────────────────────────────
    egui::Frame::group(ui.style())
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.strong("Python Environment");
            ui.add_space(4.0);

            // Auto-discover on first frame
            if app.discovered_pythons.is_none() {
                app.discovered_pythons = Some(discover_pythons());
            }

            ui.horizontal(|ui| {
                let discovered = app.discovered_pythons.as_ref().cloned().unwrap_or_default();

                let current_label = if app.python_path.is_empty() {
                    "(select Python)".to_string()
                } else {
                    truncate_path(&app.python_path, 50)
                };

                let mut changed = false;
                egui::ComboBox::from_id_salt("python_combo")
                    .selected_text(current_label)
                    .width(ui.available_width() - 8.0)
                    .show_ui(ui, |ui| {
                        for entry in &discovered {
                            let label = format!("{} ({})", entry.label, entry.path);
                            if ui
                                .selectable_value(
                                    &mut app.python_path,
                                    entry.path.clone(),
                                    label,
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        }
                        ui.separator();
                        if ui
                            .selectable_label(false, "Browse for Python executable...")
                            .clicked()
                        {
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

            // Uninstall miniforge option
            let miniforge_dir = home_dir()
                .map(|h| h.join("miniforge3"))
                .unwrap_or_default();
            if miniforge_dir.exists() {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("Uninstall Miniforge")
                            .on_hover_text(format!(
                                "Removes {}",
                                miniforge_dir.display()
                            ))
                            .clicked()
                        {
                            app.uninstall_miniforge();
                        }
                    });
                });
            }
        });

    // ── Install log (shown during/after install) ─────────────────────────
    if !app.install_log.is_empty() {
        ui.add_space(4.0);
        egui::Frame::group(ui.style())
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Installation");
                    if app.install_state == crate::app::InstallState::Installing {
                        ui.spinner();
                    }
                    if matches!(
                        app.install_state,
                        crate::app::InstallState::Done | crate::app::InstallState::Failed
                    ) {
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.small_button("Dismiss").clicked() {
                                    app.install_log.clear();
                                    app.install_state = crate::app::InstallState::Idle;
                                }
                            },
                        );
                    }
                });
                ui.add_space(2.0);
                egui::ScrollArea::vertical()
                    .max_height(120.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &app.install_log {
                            ui.label(
                                egui::RichText::new(line).monospace().size(11.0),
                            );
                        }
                    });
            });
    }

    ui.add_space(8.0);

    // ── Train button ────────────────────────────────────────────────────
    match &app.training_state {
        TrainingState::Idle => {
            let can_train = app.can_train();
            let btn = egui::Button::new(
                egui::RichText::new("Train").size(18.0),
            )
            .min_size(egui::vec2(ui.available_width(), 36.0));

            if ui.add_enabled(can_train, btn).clicked() {
                start_training(app);
            }
            if !can_train {
                let mut missing = Vec::new();
                if app.input_path.is_none() {
                    missing.push("input audio");
                }
                if app.output_paths.is_empty() {
                    missing.push("output audio");
                }
                if app.destination_dir.is_none() {
                    missing.push("output directory");
                }
                if !missing.is_empty() {
                    ui.colored_label(
                        egui::Color32::GRAY,
                        format!("Select {} to begin", missing.join(", ")),
                    );
                }
            }
        }
        TrainingState::Training => {
            ui.horizontal(|ui| {
                let cancel_btn = egui::Button::new(
                    egui::RichText::new("Cancel").color(egui::Color32::from_rgb(255, 100, 100)),
                )
                .min_size(egui::vec2(100.0, 32.0));
                if ui.add(cancel_btn).clicked() {
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
                        "Epoch {}/{} — ESR: {:.6}",
                        last.epoch, app.config.epochs, last.esr
                    ));
                }
            });
        }
        TrainingState::Complete => {
            ui.horizontal(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(80, 200, 120),
                    egui::RichText::new("Training complete!").size(16.0),
                );
                if ui.button("Train Again").clicked() {
                    app.training_state = TrainingState::Idle;
                    app.epoch_history.clear();
                    app.training_log.clear();
                }
            });
        }
        TrainingState::Error(msg) => {
            ui.colored_label(
                egui::Color32::from_rgb(255, 100, 100),
                format!("Error: {}", msg),
            );
            if ui.button("Reset").clicked() {
                app.training_state = TrainingState::Idle;
                app.epoch_history.clear();
                app.training_log.clear();
            }
        }
    }

    // ── Progress / Log ──────────────────────────────────────────────────
    if !app.training_log.is_empty() {
        ui.add_space(4.0);
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

// ── Python discovery ────────────────────────────────────────────────────

#[derive(Clone)]
pub struct PythonEntry {
    pub label: String,
    pub path: String,
}

fn discover_pythons() -> Vec<PythonEntry> {
    let mut found: Vec<PythonEntry> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let candidates = ["python3", "python"];

    for name in &candidates {
        if let Ok(output) = std::process::Command::new("which").arg(name).output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let resolved = std::fs::canonicalize(&path)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| path.clone());
                if !seen.contains(&resolved) {
                    seen.insert(resolved.clone());
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

    // Conda / miniforge / anaconda environments
    if let Some(home) = home_dir() {
        let env_dirs = [
            home.join("miniconda3").join("envs"),
            home.join("anaconda3").join("envs"),
            home.join("miniforge3").join("envs"),
            home.join(".conda").join("envs"),
        ];
        for envs_dir in &env_dirs {
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

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(std::path::PathBuf::from)
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn file_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}
