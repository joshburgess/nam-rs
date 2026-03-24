use crate::app::{Architecture, TrainerApp};

pub fn show(app: &mut TrainerApp, ctx: &egui::Context) {
    let mut open = app.show_advanced;
    egui::Window::new("Advanced Options")
        .open(&mut open)
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("advanced_grid")
                .num_columns(2)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    // Architecture
                    ui.label("Architecture:");
                    egui::ComboBox::from_id_salt("arch_combo")
                        .selected_text(app.config.architecture.label())
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
                    ui.label("Epochs:");
                    let mut epochs_str = app.config.epochs.to_string();
                    if ui.text_edit_singleline(&mut epochs_str).changed() {
                        if let Ok(v) = epochs_str.parse::<u32>() {
                            app.config.epochs = v;
                        }
                    }
                    ui.end_row();

                    // Batch size
                    ui.label("Batch size:");
                    let mut bs_str = app.config.batch_size.to_string();
                    if ui.text_edit_singleline(&mut bs_str).changed() {
                        if let Ok(v) = bs_str.parse::<u32>() {
                            app.config.batch_size = v;
                        }
                    }
                    ui.end_row();

                    // Latency
                    ui.label("Reamp latency:");
                    let mut lat_str = app
                        .config
                        .latency
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut lat_str)
                                .hint_text("auto-detect"),
                        )
                        .changed()
                    {
                        app.config.latency = lat_str.parse::<i32>().ok();
                    }
                    ui.end_row();

                    // Threshold ESR
                    ui.label("Threshold ESR:");
                    let mut esr_str = app
                        .config
                        .threshold_esr
                        .map(|v| format!("{}", v))
                        .unwrap_or_default();
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut esr_str)
                                .hint_text("none (train all epochs)"),
                        )
                        .changed()
                    {
                        app.config.threshold_esr = esr_str.parse::<f64>().ok();
                    }
                    ui.end_row();

                    // Learning rate
                    ui.label("Learning rate:");
                    let mut lr_str = format!("{}", app.config.lr);
                    if ui.text_edit_singleline(&mut lr_str).changed() {
                        if let Ok(v) = lr_str.parse::<f64>() {
                            app.config.lr = v;
                        }
                    }
                    ui.end_row();

                    // LR decay
                    ui.label("LR decay:");
                    let mut decay_str = format!("{}", app.config.lr_decay);
                    if ui.text_edit_singleline(&mut decay_str).changed() {
                        if let Ok(v) = decay_str.parse::<f64>() {
                            app.config.lr_decay = v;
                        }
                    }
                    ui.end_row();
                });

            ui.add_space(4.0);
            ui.checkbox(&mut app.config.save_plot, "Save ESR plot");
        });
    app.show_advanced = open;
}
