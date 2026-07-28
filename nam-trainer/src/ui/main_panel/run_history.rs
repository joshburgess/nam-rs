use super::{file_name, format_unix_timestamp, reveal_path, section};
use crate::app::TrainerApp;

pub(super) fn show(app: &mut TrainerApp, ui: &mut egui::Ui) {
    if app.settings.recent_runs.is_empty() {
        return;
    }

    section(ui, "Run History", |ui| {
        egui::CollapsingHeader::new("Recent Runs")
            .default_open(false)
            .show(ui, |ui| {
                for run in app.settings.recent_runs.clone().into_iter().take(10) {
                    let esr = run
                        .esr
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "unknown".into());
                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        ui.label(format!(
                            "{} | ESR {esr} | {} | {} | {}",
                            format_unix_timestamp(run.completed_unix_seconds),
                            run.architecture,
                            run.device,
                            file_name(&run.model_path)
                        ));
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Reveal Model").clicked() {
                            if let Err(error) = reveal_path(&run.model_path) {
                                app.report_user_action_error("Reveal model", error);
                            }
                        }
                        if ui.button("Open Manifest").clicked() {
                            if let Err(error) = open::that(&run.manifest_path) {
                                app.report_user_action_error("Open manifest", error);
                            }
                        }
                        if ui.button("Copy Run").clicked() {
                            ui.ctx().copy_text(format!(
                                "model_path: {}\nmanifest_path: {}\nesr: {}\narchitecture: {}\ndevice: {}\ncompleted: {}",
                                run.model_path.display(),
                                run.manifest_path.display(),
                                esr,
                                run.architecture,
                                run.device,
                                format_unix_timestamp(run.completed_unix_seconds)
                            ));
                        }
                    });
                }
            });
    });
}
