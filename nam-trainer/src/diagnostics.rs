use crate::app::{PythonStatus, TrainerApp};

pub(crate) fn build_diagnostics_summary(app: &TrainerApp) -> String {
    let request_json = match crate::worker::build_train_request(app) {
        Ok(request) => serde_json::to_string_pretty(&request)
            .unwrap_or_else(|error| format!("failed to serialize request: {error}")),
        Err(error) => format!("failed to build request: {error}"),
    };
    let (python_status, nam_version, torch_version, devices) = match &app.python_status {
        PythonStatus::Ok {
            version,
            devices,
            report,
            ..
        } => (
            format!("ok ({version})"),
            report.nam_version.as_deref().unwrap_or("unknown"),
            report.torch_version.as_deref().unwrap_or("unknown"),
            devices
                .iter()
                .map(|device| format!("{} ({})", device.name, device.id))
                .collect::<Vec<_>>()
                .join(", "),
        ),
        PythonStatus::VersionTooOld { version } => (
            format!("too old ({version})"),
            "unknown",
            "unknown",
            String::new(),
        ),
        PythonStatus::NotFound => ("not found".into(), "unknown", "unknown", String::new()),
        PythonStatus::Error(message) => (
            format!("error ({message})"),
            "unknown",
            "unknown",
            String::new(),
        ),
        PythonStatus::Unknown => ("unknown".into(), "unknown", "unknown", String::new()),
    };
    let recent_log = app
        .training_log
        .iter()
        .rev()
        .take(40)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    let active_run = app
        .run
        .data()
        .artifacts
        .as_ref()
        .map(|run| {
            format!(
                "run_id: {}\nlog_path: {}\nmanifest_path: {}",
                run.id,
                run.log_path.display(),
                run.manifest_path.display()
            )
        })
        .unwrap_or_else(|| "none".into());

    format!(
        "NAM Trainer Diagnostics\n\
         python_path: {}\n\
         python_status: {python_status}\n\
         nam_version: {nam_version}\n\
         torch_version: {torch_version}\n\
         selected_device: {}\n\
         detected_devices: {devices}\n\
         active_run: {active_run}\n\n\
         request:\n{request_json}\n\n\
         recent_log:\n{recent_log}\n",
        app.python_path.display(),
        app.selected_device
    )
}
