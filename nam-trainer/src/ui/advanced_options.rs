use crate::app::{Architecture, TrainerApp};

pub fn show(app: &mut TrainerApp, ctx: &egui::Context) {
    let mut open = app.show_advanced;
    egui::Window::new("Advanced Options")
        .open(&mut open)
        .resizable(false)
        .default_width(320.0)
        .show(ctx, |ui| {
            egui::Grid::new("advanced_grid")
                .num_columns(2)
                .spacing([16.0, 8.0])
                .show(ui, |ui| {
                    // Architecture
                    ui.label("Architecture:");
                    egui::ComboBox::from_id_salt("arch_combo")
                        .selected_text(app.config.architecture.label())
                        .width(150.0)
                        .show_ui(ui, |ui| {
                            for &arch in Architecture::all() {
                                ui.selectable_value(
                                    &mut app.config.architecture,
                                    arch,
                                    arch.label(),
                                );
                            }
                        });
                    ui.end_row();

                    // Epochs
                    ui.label("Epochs:")
                        .on_hover_text("Number of training passes over the data");
                    let mut epochs_str = app.config.epochs.to_string();
                    if ui
                        .add(egui::TextEdit::singleline(&mut epochs_str).desired_width(80.0))
                        .changed()
                    {
                        if let Ok(v) = epochs_str.parse::<u32>() {
                            app.config.epochs = v;
                        }
                    }
                    ui.end_row();

                    // Batch size
                    ui.label("Batch size:")
                        .on_hover_text("Samples per training step (reduce if running out of memory)");
                    let mut bs_str = app.config.batch_size.to_string();
                    if ui
                        .add(egui::TextEdit::singleline(&mut bs_str).desired_width(80.0))
                        .changed()
                    {
                        if let Ok(v) = bs_str.parse::<u32>() {
                            app.config.batch_size = v;
                        }
                    }
                    ui.end_row();

                    // Latency
                    ui.label("Reamp latency:")
                        .on_hover_text("Sample delay between input and output. Leave blank for auto-detection.");
                    let mut lat_str = app
                        .config
                        .latency
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut lat_str)
                                .desired_width(80.0)
                                .hint_text("auto"),
                        )
                        .changed()
                    {
                        app.config.latency = if lat_str.trim().is_empty() {
                            None
                        } else {
                            lat_str.parse::<i32>().ok()
                        };
                    }
                    ui.end_row();

                    // Threshold ESR
                    ui.label("Threshold ESR:")
                        .on_hover_text("Stop training early when ESR drops below this value");
                    let mut esr_str = app
                        .config
                        .threshold_esr
                        .map(|v| format!("{v}"))
                        .unwrap_or_default();
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut esr_str)
                                .desired_width(80.0)
                                .hint_text("disabled"),
                        )
                        .changed()
                    {
                        app.config.threshold_esr = if esr_str.trim().is_empty() {
                            None
                        } else {
                            esr_str.parse::<f64>().ok()
                        };
                    }
                    ui.end_row();

                    // Learning rate
                    ui.label("Learning rate:");
                    let mut lr_str = format!("{}", app.config.lr);
                    if ui
                        .add(egui::TextEdit::singleline(&mut lr_str).desired_width(80.0))
                        .changed()
                    {
                        if let Ok(v) = lr_str.parse::<f64>() {
                            app.config.lr = v;
                        }
                    }
                    ui.end_row();

                    // LR decay
                    ui.label("LR decay:");
                    let mut decay_str = format!("{}", app.config.lr_decay);
                    if ui
                        .add(egui::TextEdit::singleline(&mut decay_str).desired_width(80.0))
                        .changed()
                    {
                        if let Ok(v) = decay_str.parse::<f64>() {
                            app.config.lr_decay = v;
                        }
                    }
                    ui.end_row();
                });

            ui.add_space(6.0);
            ui.checkbox(&mut app.config.save_plot, "Save ESR comparison plot");
        });
    app.show_advanced = open;
}
