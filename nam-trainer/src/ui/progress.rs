use crate::app::TrainerApp;

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    ui.separator();
    ui.label("Training Log");

    // Progress bar if training
    if !app.epoch_history.is_empty() {
        let current = app.epoch_history.last().map(|e| e.epoch).unwrap_or(0);
        let total = app.config.epochs;
        let fraction = current as f32 / total.max(1) as f32;
        ui.add(
            egui::ProgressBar::new(fraction)
                .text(format!("{}/{} epochs", current, total))
                .animate(true),
        );
        if let Some(last) = app.epoch_history.last() {
            ui.horizontal(|ui| {
                ui.label(format!("Train loss: {:.6}", last.train_loss));
                ui.label(format!("Val loss: {:.6}", last.val_loss));
                ui.label(format!("ESR: {:.6}", last.esr));
            });
        }
    }

    // Scrollable log
    let max_height = ui.available_height().max(100.0);
    egui::ScrollArea::vertical()
        .max_height(max_height)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for line in &app.training_log {
                ui.label(line);
            }
        });
}
