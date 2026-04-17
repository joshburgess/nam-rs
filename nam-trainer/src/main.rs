#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;
mod settings;
mod ui;
mod worker;

fn load_icon() -> Option<egui::IconData> {
    let png_bytes = include_bytes!("../resources/icon.png");
    let img = image::load_from_memory(png_bytes).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    })
}

fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([680.0, 820.0])
        .with_min_inner_size([560.0, 600.0]);

    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "NAM Trainer",
        options,
        Box::new(|cc| Ok(Box::new(app::TrainerApp::new(cc)))),
    )
}
