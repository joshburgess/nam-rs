use crate::app::{HideConsoleExt, TrainerApp};
use crate::environment_service::home_dir;
use crate::worker::{self, TrainingState};
use std::path::Path;

mod run_history;

// ── Color palette ───────────────────────────────────────────────────────
const GREEN: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);
const AMBER: egui::Color32 = egui::Color32::from_rgb(255, 180, 60);
const RED: egui::Color32 = egui::Color32::from_rgb(255, 100, 100);
const DIM: egui::Color32 = egui::Color32::from_rgb(140, 140, 140);
const SECTION_MARGIN: f32 = 10.0;
const BUTTON_WIDTH: f32 = 130.0;
const SECTION_GAP: f32 = 6.0;

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    show_destructive_confirmation(app, ui.ctx());
    ui.add_space(2.0);

    show_header(app, ui);
    if let Some(error) = app.user_action_error.clone() {
        egui::Frame::group(ui.style())
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(RED, error);
                    if ui.small_button("Dismiss").clicked() {
                        app.user_action_error = None;
                    }
                });
            });
    }
    ui.add_space(SECTION_GAP);
    show_readiness_summary(app, ui);
    ui.add_space(SECTION_GAP);
    show_audio_files(app, ui);
    ui.add_space(SECTION_GAP);
    show_configuration(app, ui);
    ui.add_space(SECTION_GAP);
    show_python_environment(app, ui);
    show_install_log(app, ui);
    show_train_controls(app, ui);
    ui.add_space(SECTION_GAP);
    show_output_conflicts(app, ui);
    ui.add_space(SECTION_GAP);
    egui::CollapsingHeader::new("History And Diagnostics")
        .default_open(false)
        .show(ui, |ui| {
            run_history::show(app, ui);
            ui.add_space(SECTION_GAP);
            show_diagnostics_panel(app, ui);
        });

    if !app.training_log.is_empty() {
        ui.add_space(4.0);
        super::progress::show(app, ui);
    }
}

fn show_readiness_summary(app: &TrainerApp, ui: &mut egui::Ui) {
    let audio_ready = app.input_path.is_some() && !app.output_paths.is_empty();
    let destination_ready = app.destination_dir.is_some();
    let environment_ready = matches!(app.python_status, crate::app::PythonStatus::Ok { .. });
    let ready = audio_ready && destination_ready && environment_ready;

    egui::Frame::group(ui.style())
        .inner_margin(SECTION_MARGIN)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                let (summary, color) = if ready {
                    ("Ready to train", GREEN)
                } else {
                    ("Setup incomplete", AMBER)
                };
                ui.strong(egui::RichText::new(summary).color(color));
                readiness_item(ui, "Audio", audio_ready);
                readiness_item(ui, "Output folder", destination_ready);
                readiness_item(ui, "Python and NAM", environment_ready);
            });
        });
}

fn readiness_item(ui: &mut egui::Ui, label: &str, ready: bool) {
    let (symbol, color) = if ready {
        ("Ready", GREEN)
    } else {
        ("Needed", AMBER)
    };
    ui.separator();
    ui.colored_label(color, format!("{label}: {symbol}"));
}

fn show_destructive_confirmation(app: &mut TrainerApp, ctx: &egui::Context) {
    let Some(action) = app.pending_destructive_action.clone() else {
        return;
    };

    let (title, description, confirm_label) = match action {
        crate::app::InstallAction::UninstallingMiniforge => {
            let path = crate::app::TrainerApp::managed_miniforge_path()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "the managed Miniforge directory".into());
            (
                "Remove managed Miniforge?",
                format!(
                    "This permanently removes {path}, including every environment and package stored there."
                ),
                "Remove Miniforge",
            )
        }
        crate::app::InstallAction::UninstallingNam => (
            "Uninstall NAM?",
            format!(
                "This removes neural-amp-modeler from the selected Python environment:\n{}",
                app.python_path.display()
            ),
            "Uninstall NAM",
        ),
        _ => return,
    };

    let mut open = true;
    let mut confirmed = false;
    let mut cancelled = false;
    egui::Window::new(title)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.set_max_width(460.0);
            ui.label(description);
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    cancelled = true;
                }
                if ui
                    .button(egui::RichText::new(confirm_label).color(RED))
                    .clicked()
                {
                    confirmed = true;
                }
            });
        });

    if confirmed {
        app.pending_destructive_action = None;
        match action {
            crate::app::InstallAction::UninstallingMiniforge => app.uninstall_miniforge(),
            crate::app::InstallAction::UninstallingNam => app.uninstall_nam(),
            _ => {}
        }
    } else if cancelled || !open {
        app.pending_destructive_action = None;
    }
}

fn show_output_conflicts(app: &mut TrainerApp, ui: &mut egui::Ui) {
    let Some(destination) = app.destination_dir.as_deref() else {
        return;
    };
    if app.allow_overwrite_outputs || app.output_paths.is_empty() {
        return;
    }

    let conflicts = crate::training_validation::output_artifact_conflicts(
        destination,
        &app.output_paths,
        non_empty_str(&app.config.output_model_basename),
        non_empty_str(&app.config.batch_name_template),
    );
    if conflicts.is_empty() {
        return;
    }

    section(ui, "Output Conflicts", |ui| {
        ui.colored_label(AMBER, "Existing files will block training.");
        egui::ScrollArea::vertical()
            .max_height(96.0)
            .show(ui, |ui| {
                for conflict in &conflicts {
                    ui.label(egui::RichText::new(conflict.display().to_string()).monospace());
                }
            });
        ui.horizontal(|ui| {
            if ui.button("Allow Overwrite").clicked() {
                app.set_allow_overwrite_outputs(true);
            }
            if ui.button("Choose Folder").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    app.settings.last_destination = Some(dir.clone());
                    app.persist_settings();
                    app.destination_dir = Some(dir);
                }
            }
            if ui.button("Rename Outputs").clicked() {
                app.config.batch_name_template = "{index}-{stem}".into();
                app.save_config();
            }
        });
    });
}

fn show_diagnostics_panel(app: &mut TrainerApp, ui: &mut egui::Ui) {
    section(ui, "Diagnostics", |ui| {
        egui::CollapsingHeader::new("Environment And Request")
            .default_open(false)
            .show(ui, |ui| {
                let mut diagnostics = app.diagnostics_text();
                ui.horizontal(|ui| {
                    if ui.button("Copy Diagnostics").clicked() {
                        ui.ctx().copy_text(diagnostics.clone());
                    }
                    if let Some(ref run) = app.run.data().artifacts {
                        if ui.button("Open Active Log").clicked() {
                            let log_path = run.log_path.clone();
                            if let Err(error) = open::that(log_path) {
                                app.report_user_action_error("Open active log", error);
                            }
                        }
                    }
                });
                ui.add(
                    egui::TextEdit::multiline(&mut diagnostics)
                        .desired_rows(10)
                        .desired_width(ui.available_width())
                        .font(egui::TextStyle::Monospace)
                        .interactive(false),
                );
            });
    });
}

// ── Header ─────────────────────────────────────────────────────────────

fn show_header(app: &mut TrainerApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("NAM Trainer").size(20.0).strong());
        ui.colored_label(
            DIM,
            egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).size(11.0),
        );
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
                    app.settings.last_input_path = Some(path.clone());
                    app.persist_settings();
                    app.input_path = Some(path);
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
                    if let Err(error) = open::that(
                        "https://drive.google.com/file/d/1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9/view?usp=drive_link",
                    ) {
                        app.report_user_action_error("Open standard input download", error);
                    }
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
                    app.output_paths = paths;
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
                    app.settings.last_destination = Some(path.clone());
                    app.persist_settings();
                    app.destination_dir = Some(path);
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

        // Device selector, only show when multiple devices available.
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

        egui::CollapsingHeader::new("Output Naming")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Single model name:");
                    let changed = ui
                        .text_edit_singleline(&mut app.config.output_model_basename)
                        .on_hover_text("Optional basename for single-output training")
                        .changed();
                    if changed {
                        app.save_config();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Batch template:");
                    let changed = ui
                        .text_edit_singleline(&mut app.config.batch_name_template)
                        .on_hover_text(
                            "Use {stem} for the output WAV name and {index} for file number",
                        )
                        .changed();
                    if changed {
                        app.save_config();
                    }
                });
            });

        egui::CollapsingHeader::new("Training Request")
            .default_open(false)
            .show(ui, |ui| {
                let request = worker::build_train_request(app);
                let mut json = match request {
                    Ok(request) => serde_json::to_string_pretty(&request).unwrap_or_else(|error| {
                        format!("Failed to serialize request summary: {error}")
                    }),
                    Err(error) => format!("Failed to build request summary: {error}"),
                };
                ui.add(
                    egui::TextEdit::multiline(&mut json)
                        .desired_rows(12)
                        .desired_width(ui.available_width())
                        .font(egui::TextStyle::Monospace)
                        .interactive(false),
                );
            });
    });
}

fn non_empty_str(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

// ── Python Environment ─────────────────────────────────────────────────

fn show_python_environment(app: &mut TrainerApp, ui: &mut egui::Ui) {
    section(ui, "Python Environment", |ui| {
        // Auto-discover on first frame (async to avoid blocking UI)
        if app.discovered_pythons.is_none() && app.python_discovery_rx.is_none() {
            app.python_discovery_rx = Some(crate::environment_service::spawn_python_discovery());
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

        let current_label = if app.python_path.as_os_str().is_empty() {
            "(select Python)".to_string()
        } else {
            truncate_path(&app.python_path, 55)
        };

        egui::ComboBox::from_id_salt("python_combo")
            .selected_text(current_label)
            .width(full_width)
            .show_ui(ui, |ui| {
                for entry in &discovered {
                    let label = format!("{} ({})", entry.label, entry.path.display());
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
                        app.python_path = path;
                        changed = true;
                    }
                }
            });

        if changed {
            app.settings.python_path = Some(app.python_path.clone());
            app.persist_settings();
            app.python_status = crate::app::PythonStatus::Unknown;
            app.check_python();
        }

        if let crate::app::PythonStatus::Ok {
            version,
            devices,
            report,
            ..
        } = &app.python_status
        {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(DIM, format!("Python {version}"));
                if let Some(nam_version) = &report.nam_version {
                    ui.colored_label(DIM, format!("NAM {nam_version}"));
                }
                if let Some(torch_version) = &report.torch_version {
                    ui.colored_label(DIM, format!("Torch {torch_version}"));
                }
                let device_summary = devices
                    .iter()
                    .map(|device| device.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                ui.colored_label(DIM, format!("Devices: {device_summary}"));
                let packed_status = if report.packed_full_config_supported {
                    "Packed full-config: available"
                } else {
                    "Packed full-config: unavailable"
                };
                ui.colored_label(DIM, packed_status);
            });
        }

        // Management buttons
        let not_installing = !matches!(app.install_state, crate::app::InstallState::Installing(_));
        if not_installing {
            let miniforge_dir = home_dir().map(|h| h.join("miniforge3")).unwrap_or_default();
            let has_miniforge = miniforge_dir.exists();
            let has_managed_miniforge = crate::app::TrainerApp::is_managed_miniforge_install();
            let has_nam = matches!(app.python_status, crate::app::PythonStatus::Ok { .. });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("Refresh Python List")
                        .on_hover_text("Re-scan available Python interpreters")
                        .clicked()
                    {
                        app.discovered_pythons = None;
                        if app.python_discovery_rx.is_none() {
                            app.python_discovery_rx =
                                Some(crate::environment_service::spawn_python_discovery());
                        }
                    }
                });
            });

            if has_miniforge && !has_managed_miniforge {
                ui.add_space(6.0);
                ui.colored_label(
                    AMBER,
                    "This Miniforge installation was not created by NAM Trainer and will not be removed.",
                );
            }

            if has_managed_miniforge || has_nam {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if has_managed_miniforge
                            && ui
                                .small_button("Uninstall Miniforge")
                                .on_hover_text(format!("Removes {}", miniforge_dir.display()))
                                .clicked()
                        {
                            app.pending_destructive_action =
                                Some(crate::app::InstallAction::UninstallingMiniforge);
                        }
                        if has_nam
                            && ui
                                .small_button("Uninstall NAM")
                                .on_hover_text("Runs: pip uninstall neural-amp-modeler")
                                .clicked()
                        {
                            app.pending_destructive_action =
                                Some(crate::app::InstallAction::UninstallingNam);
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

        match app.run.state() {
            TrainingState::Idle => {
                let can_train = app.can_train();
                let btn_text = egui::RichText::new("Train").size(16.0).strong();
                let btn =
                    egui::Button::new(btn_text).min_size(egui::vec2(ui.available_width(), 34.0));

                // Enter key to start training
                let enter_pressed = can_train
                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.any());

                if ui.add_enabled(can_train, btn).clicked() || enter_pressed {
                    if let Some(ref destination) = app.destination_dir {
                        let destination_issues =
                            crate::training_validation::validate_training_artifacts(
                                destination,
                                app.input_path.as_deref(),
                                &app.output_paths,
                                app.config.epochs,
                                app.allow_overwrite_outputs,
                                non_empty_str(&app.config.output_model_basename),
                                non_empty_str(&app.config.batch_name_template),
                            );
                        if !destination_issues.is_empty() {
                            for issue in destination_issues {
                                app.push_training_log(format!("Error: {issue}"));
                            }
                            app.run
                                .finish_error("Fix output directory settings before training.");
                            return;
                        }
                    }

                    let metadata_issues = crate::training_validation::validate_training_metadata(
                        &app.config,
                        &app.metadata,
                    );
                    if !metadata_issues.is_empty() {
                        for issue in metadata_issues {
                            app.push_training_log(format!("Error: {issue}"));
                        }
                        app.run
                            .finish_error("Fix metadata settings before training.");
                        return;
                    }

                    if let Some(message) =
                        crate::training_validation::reconcile_selected_device(app)
                    {
                        app.push_training_log(format!("Warning: {message}"));
                    }

                    // Validate audio files before starting
                    if let Some(ref input) = app.input_path {
                        let issues = crate::app::validate_audio_files(input, &app.output_paths);
                        let has_blocking_error = issues.iter().any(|issue| issue.is_error());
                        for issue in issues {
                            let label = match issue.severity {
                                crate::app::ValidationSeverity::Error => "Error",
                                crate::app::ValidationSeverity::Warning => "Warning",
                            };
                            app.push_training_log(format!("{label}: {}", issue.message));
                        }
                        if has_blocking_error {
                            app.run
                                .finish_error("Fix audio file errors before training.");
                            return;
                        }
                    }
                    app.start_training();
                }
                ui.add_space(4.0);
                let overwrite_changed = ui
                    .checkbox(
                        &mut app.allow_overwrite_outputs,
                        "Allow overwriting outputs",
                    )
                    .on_hover_text(
                        "Permit training to replace existing model, log, and manifest files",
                    )
                    .changed();
                if overwrite_changed {
                    app.set_allow_overwrite_outputs(app.allow_overwrite_outputs);
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
                        app.cancel_training();
                    }
                    // Batch progress indicator
                    if app.run.data().total_files > 1 {
                        let current_name = app
                            .run
                            .data()
                            .current_file_index
                            .checked_sub(1)
                            .and_then(|idx| app.output_paths.get(idx))
                            .map(|path| file_name(path))
                            .unwrap_or_else(|| "pending".into());
                        ui.label(format!(
                            "File {}/{}: {}",
                            app.run.data().current_file_index,
                            app.run.data().total_files,
                            current_name
                        ));
                        ui.separator();
                    }
                    if let Some(last) = app.epoch_history.last() {
                        let mut status = format!(
                            "Epoch {}/{} - ESR: {:.6}",
                            last.epoch, app.config.epochs, last.esr
                        );
                        // Show ETA if we have enough data
                        if let Some(avg) = app.run.data().avg_epoch_secs {
                            let remaining = app.config.epochs.saturating_sub(last.epoch);
                            let eta_secs = avg * remaining as f64;
                            status.push_str(&format!(" - {}", format_eta(eta_secs)));
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
                        app.prepare_train_again();
                    }
                    if app.destination_dir.is_some()
                        && ui
                            .button("Open Output Folder")
                            .on_hover_text("Open the output directory in your file manager")
                            .clicked()
                    {
                        if let Some(ref dir) = app.destination_dir {
                            let dir = dir.clone();
                            if let Err(error) = open::that(dir) {
                                app.report_user_action_error("Open output folder", error);
                            }
                        }
                    }
                    if app.model_path.is_some()
                        && ui
                            .button("Reveal Model")
                            .on_hover_text("Show the generated model in your file manager")
                            .clicked()
                    {
                        if let Some(ref model_path) = app.model_path {
                            let model_path = model_path.clone();
                            if let Err(error) = reveal_path(&model_path) {
                                app.report_user_action_error("Reveal model", error);
                            }
                        }
                    }
                    if ui
                        .button("Copy Diagnostics")
                        .on_hover_text("Copy environment, request, and recent log details")
                        .clicked()
                    {
                        ui.ctx().copy_text(app.diagnostics_text());
                    }
                });
            }
            TrainingState::Error(msg) => {
                ui.colored_label(RED, format!("Error: {}", msg));
                ui.horizontal(|ui| {
                    if ui.button("Reset").clicked() {
                        app.reset_after_error();
                    }
                    if ui
                        .button("Copy Diagnostics")
                        .on_hover_text("Copy environment, request, and recent log details")
                        .clicked()
                    {
                        ui.ctx().copy_text(app.diagnostics_text());
                    }
                });
            }
        }
    }

    // Hidden demo mode: Ctrl+Shift+D triggers a simulated training run
    if app.run.state() == TrainingState::Idle
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
                ui.colored_label(GREEN, format!("Ready - Python {version}, {device_label}"));
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
                    format!("Python {version} too old. NAM requires {min_maj}.{min_min}+"),
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
                        .on_hover_text("Runs: pip install --upgrade neural-amp-modeler")
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

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn format_unix_timestamp(timestamp: u64) -> String {
    format!("unix {timestamp}")
}

fn truncate_path(path: &Path, max_len: usize) -> String {
    let display = path.display().to_string();
    if display.chars().count() <= max_len {
        display
    } else {
        let suffix: String = display
            .chars()
            .rev()
            .take(max_len.saturating_sub(3))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("...{suffix}")
    }
}

fn reveal_path(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .hide_console()
            .status()
            .map(|_| ())
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(format!("/select,{}", path.display()))
            .hide_console()
            .status()
            .map(|_| ())
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let target = path.parent().unwrap_or(path);
        open::that(target)
            .map(|_| ())
            .map_err(std::io::Error::other)
    }
}

fn wav_info(path: &Path) -> Option<String> {
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
