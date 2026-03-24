use crate::app::TrainerApp;
use egui_plot::{Line, Plot, PlotPoints};

const AMBER: egui::Color32 = egui::Color32::from_rgb(255, 180, 60);
const GREEN: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);
const CYAN: egui::Color32 = egui::Color32::from_rgb(80, 180, 255);
const RED: egui::Color32 = egui::Color32::from_rgb(255, 100, 100);

pub fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    egui::Frame::group(ui.style())
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
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
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        stat_label(ui, "Train", last.train_loss, CYAN);
                        ui.separator();
                        stat_label(ui, "Val", last.val_loss, AMBER);
                        ui.separator();
                        stat_label(ui, "ESR", last.esr, GREEN);
                    });
                }

                // Loss curve plot
                if app.epoch_history.len() >= 2 {
                    ui.add_space(6.0);
                    show_loss_plot(app, ui);
                }

                ui.add_space(4.0);
            }

            // Scrollable log
            let max_height = if app.epoch_history.len() >= 2 {
                80.0
            } else {
                150.0
            };
            egui::ScrollArea::vertical()
                .max_height(max_height)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &app.training_log {
                        ui.label(egui::RichText::new(line).monospace().size(11.0));
                    }
                });
        });
}

fn show_loss_plot(app: &TrainerApp, ui: &mut egui::Ui) {
    let train_points: PlotPoints = app
        .epoch_history
        .iter()
        .map(|e| [e.epoch as f64, e.train_loss])
        .collect();
    let val_points: PlotPoints = app
        .epoch_history
        .iter()
        .map(|e| [e.epoch as f64, e.val_loss])
        .collect();
    let esr_points: PlotPoints = app
        .epoch_history
        .iter()
        .map(|e| [e.epoch as f64, e.esr])
        .collect();

    let train_line = Line::new(train_points)
        .name("Train loss")
        .color(CYAN)
        .width(1.5);
    let val_line = Line::new(val_points)
        .name("Val loss")
        .color(AMBER)
        .width(1.5);
    let esr_line = Line::new(esr_points)
        .name("ESR")
        .color(GREEN)
        .width(2.0);

    Plot::new("loss_plot")
        .height(140.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .show_axes([true, true])
        .x_axis_label("Epoch")
        .y_axis_label("Loss")
        .legend(egui_plot::Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(train_line);
            plot_ui.line(val_line);
            plot_ui.line(esr_line);
        });
}

fn stat_label(ui: &mut egui::Ui, name: &str, value: f64, color: egui::Color32) {
    ui.colored_label(
        color,
        egui::RichText::new(format!("{name}: {value:.6}")).monospace().size(12.0),
    );
}
