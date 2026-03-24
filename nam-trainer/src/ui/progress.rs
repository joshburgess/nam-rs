use crate::app::TrainerApp;

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    egui::Frame::group(ui.style())
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.strong("Training Progress");
            ui.add_space(4.0);

            // Progress bar
            if !app.epoch_history.is_empty() {
                let current = app.epoch_history.last().map(|e| e.epoch).unwrap_or(0);
                let total = app.config.epochs;
                let fraction = current as f32 / total.max(1) as f32;
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .text(format!("{current}/{total} epochs"))
                        .animate(true),
                );

                // Stats row
                if let Some(last) = app.epoch_history.last() {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        stat_label(ui, "Train loss", last.train_loss);
                        ui.separator();
                        stat_label(ui, "Val loss", last.val_loss);
                        ui.separator();
                        stat_label(ui, "ESR", last.esr);
                    });
                }

                ui.add_space(4.0);
            }

            // Scrollable log
            let max_height = ui.available_height().max(80.0);
            egui::ScrollArea::vertical()
                .max_height(max_height)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &app.training_log {
                        ui.label(
                            egui::RichText::new(line)
                                .monospace()
                                .size(11.0),
                        );
                    }
                });
        });
}

fn stat_label(ui: &mut egui::Ui, name: &str, value: f64) {
    ui.label(
        egui::RichText::new(format!("{name}: {value:.6}"))
            .monospace()
            .size(12.0),
    );
}
