use crate::app::TrainerApp;
use crate::worker::{self, TrainingState};

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    ui.heading("NAM Trainer");
    ui.add_space(8.0);

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

    // Python path
    ui.horizontal(|ui| {
        ui.label("Python:");
        let response = ui.text_edit_singleline(&mut app.python_path);
        if response.lost_focus() {
            app.settings.python_path = Some(app.python_path.clone());
            app.settings.save();
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

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}
