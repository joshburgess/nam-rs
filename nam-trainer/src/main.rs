mod app;
mod settings;
mod ui;
mod worker;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([620.0, 520.0])
            .with_min_inner_size([520.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "NAM Trainer",
        options,
        Box::new(|cc| Ok(Box::new(app::TrainerApp::new(cc)))),
    )
}
