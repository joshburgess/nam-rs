use crate::app::{HideConsoleExt, TrainerApp};
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

    show_header(app, ui);
    ui.add_space(SECTION_GAP);
    show_audio_files(app, ui);
    ui.add_space(SECTION_GAP);
    show_configuration(app, ui);
    ui.add_space(SECTION_GAP);
    show_python_environment(app, ui);
    show_install_log(app, ui);
    show_train_controls(app, ui);

    if !app.training_log.is_empty() {
        ui.add_space(4.0);
        super::progress::show(app, ui);
    }
}

// ── Header ─────────────────────────────────────────────────────────────

fn show_header(app: &mut TrainerApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("NAM Trainer").size(20.0).strong());
        ui.colored_label(DIM, egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            show_status_badge(app, ui);
        });
    });
}

// ── Audio Files ────────────────────────────────────────────────────────

fn show_audio_files(app: &mut TrainerApp, ui: &mut egui::Ui) {
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
                if let Some(info) = wav_info(p) {
                    ui.colored_label(DIM, info);
                }
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
                        "https://drive.google.com/file/d/1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9/view?usp=drive_link",
                    );
                }
            });
        });

        // Output audio
        ui.horizontal(|ui| {
            let btn = egui::Button::new("Output Audio...").min_size(egui::vec2(BUTTON_WIDTH, 0.0));
            if ui.add(btn).clicked() {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("WAV files", &["wav"])
                    .pick_files()
                {
                    app.output_paths = paths.iter().map(|p| p.display().to_string()).collect();
                }
            }
            match app.output_paths.len() {
                0 => {
                    ui.colored_label(DIM, "No file(s) selected");
                }
                1 => {
                    ui.colored_label(GREEN, "\u{2713}");
                    ui.label(file_name(&app.output_paths[0]));
                    if let Some(info) = wav_info(&app.output_paths[0]) {
                        ui.colored_label(DIM, info);
                    }
                }
                n => {
                    ui.colored_label(GREEN, "\u{2713}");
                    ui.label(format!("{n} files selected"));
                }
            }
        });

        // Destination
        ui.horizontal(|ui| {
            let btn =
                egui::Button::new("Output Directory...").min_size(egui::vec2(BUTTON_WIDTH, 0.0));
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
}

// ── Configuration ──────────────────────────────────────────────────────

fn show_configuration(app: &mut TrainerApp, ui: &mut egui::Ui) {
    section(ui, "Configuration", |ui| {
        ui.horizontal(|ui| {
            ui.label("Model:");
            ui.add_space(4.0);
            let prev_arch = app.config.architecture;
            for &arch in crate::app::Architecture::all() {
                ui.selectable_value(&mut app.config.architecture, arch, arch.label())
                    .on_hover_text(arch.tooltip());
            }
            if app.config.architecture != prev_arch {
                app.save_config();
            }
        });

        // Device selector — only show when multiple devices available.
        // Clone out of python_status first so we can mutate `app` (e.g. to
        // re-run detection from the warning's Re-check button) below.
        let env_info = if let crate::app::PythonStatus::Ok {
            devices, warnings, ..
        } = &app.python_status
        {
            Some((devices.clone(), warnings.clone()))
        } else {
            None
        };
        if let Some((devices, warnings)) = env_info {
            if devices.len() > 1 {
                ui.horizontal(|ui| {
                    ui.label("Device:");
                    ui.add_space(4.0);
                    for dev in &devices {
                        ui.selectable_value(&mut app.selected_device, dev.id.clone(), &dev.name);
                    }
                });
            }

            // GPU warnings (e.g., NVIDIA hardware detected but PyTorch lacks
            // CUDA). The Install PyTorch button runs pip through the absolute
            // Miniforge python path so PATH is irrelevant. Re-check is a
            // secondary action for users who fixed their install some other way.
            let cuda_install = app.cuda_install.clone();
            let installing = matches!(app.install_state, crate::app::InstallState::Installing(_));
            for warning in &warnings {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(AMBER, format!("\u{26A0} {warning}"));
                });
            }
            if !warnings.is_empty() {
                ui.horizontal(|ui| {
                    if let Some(ref ci) = cuda_install {
                        let btn_text = format!("Install PyTorch with CUDA {}", ci.cuda_version);
                        if ui
                            .add_enabled(!installing, egui::Button::new(btn_text))
                            .on_hover_text(format!(
                                "Runs pip install torch --index-url {}",
                                ci.wheel_index
                            ))
                            .clicked()
                        {
                            app.install_cuda_torch();
                        }
                    }
                    if ui
                        .add_enabled(!installing, egui::Button::new("Re-check"))
                        .on_hover_text("Re-run Python/PyTorch detection")
                        .clicked()
                    {
                        app.python_status = crate::app::PythonStatus::Unknown;
                        app.check_python();
                    }
                });
            }
        }

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
}

// ── Python Environment ─────────────────────────────────────────────────

fn show_python_environment(app: &mut TrainerApp, ui: &mut egui::Ui) {
    section(ui, "Python Environment", |ui| {
        // Auto-discover on first frame (async to avoid blocking UI)
        if app.discovered_pythons.is_none() && app.python_discovery_rx.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            app.python_discovery_rx = Some(rx);
            std::thread::spawn(move || {
                let _ = tx.send(discover_pythons());
            });
        }
        // Check if discovery completed
        if let Some(ref rx) = app.python_discovery_rx {
            if let Ok(result) = rx.try_recv() {
                app.discovered_pythons = Some(result);
                app.python_discovery_rx = None;
            }
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
                        .selectable_value(&mut app.python_path, entry.path.clone(), label)
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
        let not_installing = !matches!(app.install_state, crate::app::InstallState::Installing(_));
        if not_installing {
            let miniforge_dir = home_dir().map(|h| h.join("miniforge3")).unwrap_or_default();
            let has_miniforge = miniforge_dir.exists();
            let has_nam = matches!(app.python_status, crate::app::PythonStatus::Ok { .. });

            if has_miniforge || has_nam {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if has_miniforge
                            && ui
                                .small_button("Uninstall Miniforge")
                                .on_hover_text(format!("Removes {}", miniforge_dir.display()))
                                .clicked()
                        {
                            app.uninstall_miniforge();
                        }
                        if has_nam
                            && ui
                                .small_button("Uninstall NAM")
                                .on_hover_text("Runs: pip uninstall neural-amp-modeler")
                                .clicked()
                        {
                            app.uninstall_nam();
                        }
                    });
                });
            }
        }
    });
}

// ── Install/Uninstall log ──────────────────────────────────────────────

fn show_install_log(app: &mut TrainerApp, ui: &mut egui::Ui) {
    if app.install_log.is_empty() {
        return;
    }

    ui.add_space(SECTION_GAP);
    egui::Frame::group(ui.style())
        .inner_margin(SECTION_MARGIN)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let header = match &app.install_state {
                    crate::app::InstallState::Installing(action) => match action {
                        crate::app::InstallAction::InstallingPython => "Installing Python",
                        crate::app::InstallAction::InstallingNam => "Installing NAM",
                        crate::app::InstallAction::InstallingCudaTorch => {
                            "Installing PyTorch (CUDA)"
                        }
                        crate::app::InstallAction::UninstallingNam => "Uninstalling NAM",
                        crate::app::InstallAction::UninstallingMiniforge => "Removing Miniforge",
                    },
                    _ => "Setup",
                };
                ui.strong(header);
                if matches!(app.install_state, crate::app::InstallState::Installing(_)) {
                    ui.add(egui::Spinner::new().color(AMBER));
                }
                if matches!(
                    app.install_state,
                    crate::app::InstallState::Done | crate::app::InstallState::Failed
                ) {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Dismiss").clicked() {
                            app.install_log.clear();
                            app.install_state = crate::app::InstallState::Idle;
                        }
                    });
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

// ── Train controls ─────────────────────────────────────────────────────

fn show_train_controls(app: &mut TrainerApp, ui: &mut egui::Ui) {
    let env_ready = matches!(app.python_status, crate::app::PythonStatus::Ok { .. });
    let no_active_install = !matches!(app.install_state, crate::app::InstallState::Installing(_))
        && app.install_log.is_empty();

    if env_ready && no_active_install {
        ui.add_space(SECTION_GAP + 2.0);

        match &app.training_state {
            TrainingState::Idle => {
                let can_train = app.can_train();
                let btn_text = egui::RichText::new("Train").size(16.0).strong();
                let btn =
                    egui::Button::new(btn_text).min_size(egui::vec2(ui.available_width(), 34.0));

                // Enter key to start training
                let enter_pressed = can_train
                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.any());

                if ui.add_enabled(can_train, btn).clicked() || enter_pressed {
                    // Validate audio files before starting
                    if let Some(ref input) = app.input_path {
                        let issues =
                            crate::app::validate_audio_files(input, &app.output_paths);
                        if !issues.is_empty() {
                            for issue in &issues {
                                app.training_log.push(format!("Warning: {issue}"));
                            }
                        }
                    }
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
                // Escape key to cancel training
                let escape_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape));

                ui.horizontal(|ui| {
                    let cancel_btn = egui::Button::new(egui::RichText::new("Cancel").color(RED))
                        .min_size(egui::vec2(100.0, 32.0));
                    if ui.add(cancel_btn).clicked() || escape_pressed {
                        if let Some(ref mut w) = app.worker {
                            w.cancel();
                        }
                        app.training_state = TrainingState::Idle;
                        app.worker = None;
                        app.message_rx = None;
                        app.training_log.push("Training cancelled.".into());
                    }
                    // Batch progress indicator
                    if app.total_files > 1 {
                        ui.label(format!(
                            "File {}/{}",
                            app.current_file_index, app.total_files
                        ));
                        ui.separator();
                    }
                    if let Some(last) = app.epoch_history.last() {
                        let mut status = format!(
                            "Epoch {}/{} \u{2014} ESR: {:.6}",
                            last.epoch, app.config.epochs, last.esr
                        );
                        // Show ETA if we have enough data
                        if let Some(avg) = app.avg_epoch_secs {
                            let remaining = app.config.epochs.saturating_sub(last.epoch);
                            let eta_secs = avg * remaining as f64;
                            status.push_str(&format!(" \u{2014} {}", format_eta(eta_secs)));
                        }
                        ui.label(status);
                    }
                });
            }
            TrainingState::Complete => {
                let final_esr = app.epoch_history.last().map(|e| e.esr);
                ui.horizontal(|ui| {
                    let label = if let Some(esr) = final_esr {
                        format!("Training complete! (ESR: {:.6})", esr)
                    } else {
                        "Training complete!".to_string()
                    };
                    ui.colored_label(GREEN, egui::RichText::new(label).size(15.0));
                });
                ui.horizontal(|ui| {
                    if ui.button("Train Again").clicked() {
                        app.training_state = TrainingState::Idle;
                        app.epoch_history.clear();
                        app.training_log.clear();
                        app.model_path = None;
                    }
                    if app.destination_dir.is_some() {
                        if ui
                            .button("Open Output Folder")
                            .on_hover_text("Open the output directory in your file manager")
                            .clicked()
                        {
                            if let Some(ref dir) = app.destination_dir {
                                let _ = open::that(dir);
                            }
                        }
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

    // Hidden demo mode: Ctrl+Shift+D triggers a simulated training run
    if app.training_state == TrainingState::Idle
        && ui.input(|i| i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::D))
    {
        app.start_demo_training();
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
            crate::app::InstallAction::InstallingCudaTorch => "Installing PyTorch (CUDA)...",
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
            crate::app::PythonStatus::Ok {
                version, devices, ..
            } => {
                let best = devices
                    .iter()
                    .find(|d| d.id.starts_with("cuda") || d.id == "mps")
                    .or(devices.first());
                let device_label = best.map(|d| d.name.as_str()).unwrap_or("CPU");
                ui.colored_label(
                    GREEN,
                    format!("Ready \u{2014} Python {version}, {device_label}"),
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
                    RED,
                    format!("Python {version} too old \u{2014} NAM requires {min_maj}.{min_min}+"),
                );
                if clicked {
                    app.install_python();
                }
            }
            crate::app::PythonStatus::NotFound => {
                let clicked = ui
                    .button("Install Python")
                    .on_hover_text(
                        "Downloads and installs Miniforge (Python 3.12+) to ~/miniforge3",
                    )
                    .clicked();
                ui.colored_label(RED, "Python not found");
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
    app.model_path = None;
    app.current_file_index = 0;
    app.total_files = app.output_paths.len();
    app.training_start_time = Some(std::time::Instant::now());
    app.last_epoch_time = None;
    app.avg_epoch_secs = None;
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
        if let Ok(output) = std::process::Command::new(which_cmd)
            .arg(name)
            .hide_console()
            .output()
        {
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
                            .hide_console()
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

fn wav_info(path: &str) -> Option<String> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let samples = reader.len() as f64;
    let channels = spec.channels as f64;
    let rate = spec.sample_rate as f64;
    let duration = samples / channels / rate;

    let rate_khz = spec.sample_rate as f64 / 1000.0;
    let rate_str = if rate_khz == rate_khz.floor() {
        format!("{}kHz", rate_khz as u32)
    } else {
        format!("{:.1}kHz", rate_khz)
    };

    if duration < 60.0 {
        Some(format!("({}, {:.1}s)", rate_str, duration))
    } else {
        let mins = (duration / 60.0).floor() as u32;
        let secs = (duration % 60.0).round() as u32;
        Some(format!("({}, {}m {}s)", rate_str, mins, secs))
    }
}

fn format_eta(secs: f64) -> String {
    let secs = secs.round() as u64;
    if secs < 60 {
        format!("~{secs}s remaining")
    } else if secs < 3600 {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("~{m} min remaining")
        } else {
            format!("~{m}m {s}s remaining")
        }
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        format!("~{h}h {m}m remaining")
    }
}
