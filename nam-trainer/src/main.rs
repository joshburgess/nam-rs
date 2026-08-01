#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::path::PathBuf;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let smoke_test_report = smoke_test_report_path();
    if let Some(report_path) = smoke_test_report.clone() {
        nam_trainer::configure_smoke_test(report_path)?;
    }
    if has_argument("--headless-smoke-test") {
        if smoke_test_report.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--headless-smoke-test requires --smoke-test-report",
            )
            .into());
        }
        if load_icon().is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "embedded trainer icon could not be decoded",
            )
            .into());
        }
        return Ok(());
    }

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
        Box::new(|cc| Ok(Box::new(nam_trainer::TrainerApp::new(cc)))),
    )?;
    Ok(())
}

fn smoke_test_report_path() -> Option<PathBuf> {
    let mut arguments = std::env::args_os();
    while let Some(argument) = arguments.next() {
        if argument == "--smoke-test-report" {
            return arguments.next().map(PathBuf::from);
        }
    }
    None
}

fn has_argument(expected: &str) -> bool {
    std::env::args_os().any(|argument| argument == expected)
}
