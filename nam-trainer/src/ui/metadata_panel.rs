use crate::app::{GearType, ToneType, TrainerApp};

pub fn show(app: &mut TrainerApp, ctx: &egui::Context) {
    let mut open = app.show_metadata;
    egui::Window::new("Model Metadata")
        .open(&mut open)
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("metadata_grid")
                .num_columns(2)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    ui.label("NAM name:");
                    ui.text_edit_singleline(&mut app.metadata.name);
                    ui.end_row();

                    ui.label("Modeled by:");
                    ui.text_edit_singleline(&mut app.metadata.modeled_by);
                    ui.end_row();

                    ui.label("Gear make:");
                    ui.text_edit_singleline(&mut app.metadata.gear_make);
                    ui.end_row();

                    ui.label("Gear model:");
                    ui.text_edit_singleline(&mut app.metadata.gear_model);
                    ui.end_row();

                    // Gear type combo
                    ui.label("Gear type:");
                    let gear_label = app
                        .metadata
                        .gear_type
                        .map(|g| g.label())
                        .unwrap_or("(none)");
                    egui::ComboBox::from_id_salt("gear_type_combo")
                        .selected_text(gear_label)
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

                    // Tone type combo
                    ui.label("Tone type:");
                    let tone_label = app
                        .metadata
                        .tone_type
                        .map(|t| t.label())
                        .unwrap_or("(none)");
                    egui::ComboBox::from_id_salt("tone_type_combo")
                        .selected_text(tone_label)
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
                    ui.label("Input level (dBu):");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.input_level_dbu)
                            .hint_text("optional"),
                    );
                    ui.end_row();

                    ui.label("Output level (dBu):");
                    ui.add(
                        egui::TextEdit::singleline(&mut app.metadata.output_level_dbu)
                            .hint_text("optional"),
                    );
                    ui.end_row();
                });
        });
    app.show_metadata = open;
}
