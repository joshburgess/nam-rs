#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;
mod settings;
mod ui;
mod worker;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([680.0, 820.0])
            .with_min_inner_size([560.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "NAM Trainer",
        options,
        Box::new(|cc| Ok(Box::new(app::TrainerApp::new(cc)))),
    )
}
