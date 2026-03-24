use crate::app::TrainerApp;
use crate::worker::{self, TrainingState};

// ── Color palette ───────────────────────────────────────────────────────
const GREEN: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);
const AMBER: egui::Color32 = egui::Color32::from_rgb(255, 180, 60);
const RED: egui::Color32 = egui::Color32::from_rgb(255, 100, 100);
const DIM: egui::Color32 = egui::Color32::from_rgb(140, 140, 140);
const SECTION_MARGIN: f32 = 10.0;
const BUTTON_WIDTH: f32 = 130.0;
const SECTION_GAP: f32 = 6.0;

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    ui.add_space(2.0);

    // ── Header ──────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("NAM Trainer").size(20.0).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            show_status_badge(app, ui);
        });
    });

    ui.add_space(SECTION_GAP);

    // ── Audio Files ─────────────────────────────────────────────────────
    section(ui, "Audio Files", |ui| {
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
                ui.colored_label(GREEN, "\u{2713}"); // checkmark
                ui.label(file_name(p));
            } else {
                ui.colored_label(DIM, "No file selected");
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
                    ui.colored_label(DIM, "No file(s) selected");
                }
                1 => {
                    ui.colored_label(GREEN, "\u{2713}");
                    ui.label(file_name(&app.output_paths[0]));
                }
                n => {
                    ui.colored_label(GREEN, "\u{2713}");
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
                ui.colored_label(GREEN, "\u{2713}");
                ui.label(truncate_path(p, 45));
            } else {
                ui.colored_label(DIM, "No directory selected");
            }
        });
    });

    ui.add_space(SECTION_GAP);

    // ── Configuration ───────────────────────────────────────────────────
    section(ui, "Configuration", |ui| {
        ui.horizontal(|ui| {
            ui.label("Model:");
            ui.add_space(4.0);
            for &arch in crate::app::Architecture::all() {
                ui.selectable_value(&mut app.config.architecture, arch, arch.label());
            }
        });

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(format!("Epochs: {}", app.config.epochs)).monospace());
            ui.add_space(12.0);
            ui.label(egui::RichText::new(format!("Batch: {}", app.config.batch_size)).monospace());
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

    ui.add_space(SECTION_GAP);

    // ── Python Environment ──────────────────────────────────────────────
    section(ui, "Python Environment", |ui| {
        // Auto-discover on first frame
        if app.discovered_pythons.is_none() {
            app.discovered_pythons = Some(discover_pythons());
        }

        let full_width = ui.available_width();
        let mut changed = false;
        let discovered = app.discovered_pythons.as_ref().cloned().unwrap_or_default();

        let current_label = if app.python_path.is_empty() {
            "(select Python)".to_string()
        } else {
            truncate_path(&app.python_path, 55)
        };

        egui::ComboBox::from_id_salt("python_combo")
            .selected_text(current_label)
            .width(full_width)
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

        // Management buttons
        let not_installing =
            !matches!(app.install_state, crate::app::InstallState::Installing(_));
        if not_installing {
            let miniforge_dir = home_dir()
                .map(|h| h.join("miniforge3"))
                .unwrap_or_default();
            let has_miniforge = miniforge_dir.exists();
            let has_nam = matches!(app.python_status, crate::app::PythonStatus::Ok { .. });

            if has_miniforge || has_nam {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if has_miniforge {
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
                            }
                            if has_nam {
                                if ui
                                    .small_button("Uninstall NAM")
                                    .on_hover_text(
                                        "Runs: pip uninstall neural-amp-modeler",
                                    )
                                    .clicked()
                                {
                                    app.uninstall_nam();
                                }
                            }
                        },
                    );
                });
            }
        }
    });

    // ── Install/Uninstall log ───────────────────────────────────────────
    if !app.install_log.is_empty() {
        ui.add_space(SECTION_GAP);
        egui::Frame::group(ui.style())
            .inner_margin(SECTION_MARGIN)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let header = match &app.install_state {
                        crate::app::InstallState::Installing(action) => match action {
                            crate::app::InstallAction::InstallingPython => "Installing Python",
                            crate::app::InstallAction::InstallingNam => "Installing NAM",
                            crate::app::InstallAction::UninstallingNam => "Uninstalling NAM",
                            crate::app::InstallAction::UninstallingMiniforge => {
                                "Removing Miniforge"
                            }
                        },
                        _ => "Setup",
                    };
                    ui.strong(header);
                    if matches!(
                        app.install_state,
                        crate::app::InstallState::Installing(_)
                    ) {
                        ui.add(egui::Spinner::new().color(AMBER));
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
                            ui.label(egui::RichText::new(line).monospace().size(11.0));
                        }
                    });
            });
    }

    // ── Train ───────────────────────────────────────────────────────────
    let env_ready = matches!(app.python_status, crate::app::PythonStatus::Ok { .. });
    let no_active_install =
        !matches!(app.install_state, crate::app::InstallState::Installing(_))
            && app.install_log.is_empty();

    if env_ready && no_active_install {
        ui.add_space(SECTION_GAP + 2.0);

        match &app.training_state {
            TrainingState::Idle => {
                let can_train = app.can_train();
                let btn_text = egui::RichText::new("Train")
                    .size(16.0)
                    .strong();
                let btn = egui::Button::new(btn_text)
                    .min_size(egui::vec2(ui.available_width(), 34.0));

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
                        ui.add_space(4.0);
                        ui.colored_label(DIM, format!("Select {} to begin", missing.join(", ")));
                    }
                }
            }
            TrainingState::Training => {
                ui.horizontal(|ui| {
                    let cancel_btn = egui::Button::new(
                        egui::RichText::new("Cancel").color(RED),
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
                    ui.add(egui::Spinner::new().color(AMBER));
                    if let Some(last) = app.epoch_history.last() {
                        ui.label(format!(
                            "Epoch {}/{} \u{2014} ESR: {:.6}",
                            last.epoch, app.config.epochs, last.esr
                        ));
                    }
                });
            }
            TrainingState::Complete => {
                ui.horizontal(|ui| {
                    ui.colored_label(GREEN, egui::RichText::new("Training complete!").size(15.0));
                    if ui.button("Train Again").clicked() {
                        app.training_state = TrainingState::Idle;
                        app.epoch_history.clear();
                        app.training_log.clear();
                    }
                });
            }
            TrainingState::Error(msg) => {
                ui.colored_label(RED, format!("Error: {}", msg));
                if ui.button("Reset").clicked() {
                    app.training_state = TrainingState::Idle;
                    app.epoch_history.clear();
                    app.training_log.clear();
                }
            }
        }
    }

    // Demo button — always available for testing the progress UI
    if app.training_state == TrainingState::Idle && app.training_log.is_empty() {
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("Demo")
                    .on_hover_text("Simulate training to preview the progress UI")
                    .clicked()
                {
                    app.start_demo_training();
                }
            });
        });
    }

    // ── Training Progress ───────────────────────────────────────────────
    if !app.training_log.is_empty() {
        ui.add_space(4.0);
        super::progress::show(app, ui);
    }
}

// ── Status badge ────────────────────────────────────────────────────────

fn show_status_badge(app: &mut TrainerApp, ui: &mut egui::Ui) {
    let (min_maj, min_min) = crate::app::NAM_MIN_PYTHON;
    let active_action = match &app.install_state {
        crate::app::InstallState::Installing(action) => Some(action.clone()),
        _ => None,
    };

    if let Some(action) = &active_action {
        ui.add(egui::Spinner::new().color(AMBER));
        let label = match action {
            crate::app::InstallAction::InstallingPython => "Installing Python...",
            crate::app::InstallAction::InstallingNam => "Installing NAM...",
            crate::app::InstallAction::UninstallingNam => "Uninstalling NAM...",
            crate::app::InstallAction::UninstallingMiniforge => "Removing Miniforge...",
        };
        ui.colored_label(AMBER, label);
    } else {
        match &app.python_status {
            crate::app::PythonStatus::Unknown => {
                ui.spinner();
                ui.colored_label(DIM, "Checking Python...");
            }
            crate::app::PythonStatus::Ok { gpu, version } => {
                let gpu_label = match gpu.as_deref() {
                    Some("cuda") => "CUDA GPU",
                    Some("mps") => "Apple GPU",
                    _ => "CPU",
                };
                ui.colored_label(GREEN, format!("Ready \u{2014} Python {version}, {gpu_label}"));
            }
            crate::app::PythonStatus::VersionTooOld { version } => {
                let clicked = ui
                    .button("Install Python")
                    .on_hover_text(
                        "Downloads and installs Miniforge (Python 3.12+) to ~/miniforge3",
                    )
                    .clicked();
                ui.colored_label(
                    RED,
                    format!("Python {version} too old \u{2014} NAM requires {min_maj}.{min_min}+"),
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
                    ui.colored_label(AMBER, "NAM not installed");
                    if clicked {
                        app.install_nam();
                    }
                } else {
                    ui.colored_label(RED, msg.as_str());
                }
            }
        }
    }
}

// ── Section helper ──────────────────────────────────────────────────────

fn section(ui: &mut egui::Ui, title: &str, content: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style())
        .inner_margin(SECTION_MARGIN)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 6.0;
            ui.strong(title);
            ui.add_space(2.0);
            content(ui);
        });
}

// ── Training ────────────────────────────────────────────────────────────

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

    // Platform-specific: which command and candidate names
    #[cfg(not(target_os = "windows"))]
    let (which_cmd, candidates) = ("which", vec!["python3", "python"]);
    #[cfg(target_os = "windows")]
    let (which_cmd, candidates) = ("where", vec!["python", "python3"]);

    for name in &candidates {
        if let Ok(output) = std::process::Command::new(which_cmd).arg(name).output() {
            if output.status.success() {
                // `where` on Windows can return multiple lines; take each one
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let path = line.trim().to_string();
                    if path.is_empty() {
                        continue;
                    }
                    let resolved = std::fs::canonicalize(&path)
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| path.clone());
                    if seen.insert(resolved) {
                        let version = std::process::Command::new(&path)
                            .args(["--version"])
                            .output()
                            .ok()
                            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                            .unwrap_or_default();
                        found.push(PythonEntry {
                            label: if version.is_empty() {
                                name.to_string()
                            } else {
                                version
                            },
                            path,
                        });
                    }
                }
            }
        }
    }

    // Conda environments
    if let Some(home) = home_dir() {
        for base in &["miniconda3", "anaconda3", "miniforge3", ".conda"] {
            let envs_dir = home.join(base).join("envs");
            if let Ok(entries) = std::fs::read_dir(&envs_dir) {
                for entry in entries.flatten() {
                    let env_python = conda_python_path(&entry.path());
                    if env_python.exists() {
                        let path = env_python.display().to_string();
                        let resolved = std::fs::canonicalize(&path)
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| path.clone());
                        if seen.insert(resolved) {
                            let env_name = entry.file_name().to_string_lossy().to_string();
                            found.push(PythonEntry {
                                label: format!("conda: {env_name}"),
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

/// Returns the path to the python executable inside a conda environment.
fn conda_python_path(env_dir: &std::path::Path) -> std::path::PathBuf {
    #[cfg(not(target_os = "windows"))]
    {
        env_dir.join("bin").join("python")
    }
    #[cfg(target_os = "windows")]
    {
        env_dir.join("python.exe")
    }
}

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(std::path::PathBuf::from)
}

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
