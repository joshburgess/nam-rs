use crate::app::{GearType, ToneType, TrainerApp};

pub fn show(app: &mut TrainerApp, ctx: &egui::Context) {
    let mut open = app.show_metadata;
    egui::Window::new("Model Metadata")
        .open(&mut open)
        .resizable(false)
        .default_width(360.0)
        .show(ctx, |ui| {
            ui.label("Optional metadata embedded in the .nam model file.");
            ui.add_space(6.0);

            egui::Grid::new("metadata_grid")
                .num_columns(2)
                .spacing([16.0, 8.0])
                .show(ui, |ui| {
                    ui.label("NAM name:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.name)
                            .desired_width(200.0)
                            .hint_text("e.g. My Fender Twin"),
                    );
                    ui.end_row();

                    ui.label("Modeled by:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.modeled_by)
                            .desired_width(200.0)
                            .hint_text("Your name"),
                    );
                    ui.end_row();

                    ui.label("Gear make:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.gear_make)
                            .desired_width(200.0)
                            .hint_text("e.g. Fender"),
                    );
                    ui.end_row();

                    ui.label("Gear model:");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.gear_model)
                            .desired_width(200.0)
                            .hint_text("e.g. Twin Reverb '65"),
                    );
                    ui.end_row();

                    // Gear type
                    ui.label("Gear type:");
                    let gear_label = app
                        .metadata
                        .gear_type
                        .map(|g| g.label())
                        .unwrap_or("(none)");
                    egui::ComboBox::from_id_salt("gear_type_combo")
                        .selected_text(gear_label)
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut app.metadata.gear_type, None, "(none)");
                            for &gt in GearType::all() {
                                ui.selectable_value(
                                    &mut app.metadata.gear_type,
                                    Some(gt),
                                    gt.label(),
                                );
                            }
                        });
                    ui.end_row();

                    // Tone type
                    ui.label("Tone type:");
                    let tone_label = app
                        .metadata
                        .tone_type
                        .map(|t| t.label())
                        .unwrap_or("(none)");
                    egui::ComboBox::from_id_salt("tone_type_combo")
                        .selected_text(tone_label)
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut app.metadata.tone_type, None, "(none)");
                            for &tt in ToneType::all() {
                                ui.selectable_value(
                                    &mut app.metadata.tone_type,
                                    Some(tt),
                                    tt.label(),
                                );
                            }
                        });
                    ui.end_row();

                    // Level fields
                    ui.label("Input level (dBu):")
                        .on_hover_text("Reamp send level in dBu (for calibration metadata)");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.input_level_dbu)
                            .desired_width(80.0)
                            .hint_text("optional"),
                    );
                    ui.end_row();

                    ui.label("Output level (dBu):")
                        .on_hover_text("Reamp return level in dBu (for calibration metadata)");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.output_level_dbu)
                            .desired_width(80.0)
                            .hint_text("optional"),
                    );
                    ui.end_row();
                });
        });
    app.show_metadata = open;
}
