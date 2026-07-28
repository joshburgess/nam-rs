use crate::app::{ModelMetadata, PythonStatus, TrainerApp, TrainingConfig};
use crate::artifacts::TrainingRunArtifacts;
use std::path::{Path, PathBuf};

const MIN_FREE_DESTINATION_BYTES: u64 = 1_000_000_000;
const TRAINING_ARTIFACT_BYTES_PER_OUTPUT: u64 = 250_000_000;

pub fn validate_training_metadata(
    config: &TrainingConfig,
    metadata: &ModelMetadata,
) -> Vec<String> {
    let mut issues = Vec::new();
    for (label, value) in [
        ("Input level dBu", metadata.input_level_dbu.trim()),
        ("Output level dBu", metadata.output_level_dbu.trim()),
    ] {
        if !value.is_empty() && value.parse::<f64>().is_err() {
            issues.push(format!("{label} must be a number"));
        }
    }
    if config.use_full_config_trainer && metadata_has_values(metadata) {
        issues.push(
            "Upstream full-config trainer mode does not export GUI metadata. Disable full-config trainer mode or clear metadata fields.".into(),
        );
    }
    issues
}

pub fn validate_destination_dir(destination: &Path) -> Vec<String> {
    let mut issues = Vec::new();
    let path = destination;
    if !path.exists() {
        issues.push("Output directory does not exist".into());
        return issues;
    }
    if !path.is_dir() {
        issues.push("Output path is not a directory".into());
        return issues;
    }

    let probe_path = path.join(format!(".nam-rs-write-test-{}", std::process::id()));
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&probe_path)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe_path);
        }
        Err(error) => issues.push(format!("Output directory is not writable: {error}")),
    }
    match available_space_bytes(path) {
        Ok(Some(bytes)) if bytes < MIN_FREE_DESTINATION_BYTES => issues.push(format!(
            "Output directory has less than {} MB free",
            MIN_FREE_DESTINATION_BYTES / 1_000_000
        )),
        Ok(_) => {}
        Err(error) => issues.push(format!(
            "Could not check output directory free space: {error}"
        )),
    }
    issues
}

pub fn validate_training_artifacts(
    destination: &Path,
    input_path: Option<&Path>,
    output_paths: &[PathBuf],
    epochs: u32,
    allow_overwrite: bool,
    output_model_basename: Option<&str>,
    batch_name_template: Option<&str>,
) -> Vec<String> {
    let mut issues = validate_destination_dir(destination);
    if !issues.is_empty() {
        return issues;
    }
    let artifacts = TrainingRunArtifacts::new(destination, output_paths).with_naming(
        output_model_basename.map(str::to_string),
        batch_name_template.map(str::to_string),
    );
    issues.extend(
        artifacts
            .duplicate_artifact_paths()
            .into_iter()
            .map(|path| {
                format!(
                    "Multiple output files would write the same artifact: {}. Include {{index}} in the batch naming template.",
                    path.display()
                )
            }),
    );
    if !allow_overwrite {
        issues.extend(
            output_artifact_conflicts(
                destination,
                output_paths,
                output_model_basename,
                batch_name_template,
            )
            .into_iter()
            .map(|path| format!("Output artifact already exists: {}", path.display())),
        );
    }
    issues.extend(validate_estimated_training_space(
        destination,
        input_path,
        output_paths,
        epochs,
    ));
    issues
}

pub fn output_artifact_conflicts(
    destination: &Path,
    output_paths: &[PathBuf],
    output_model_basename: Option<&str>,
    batch_name_template: Option<&str>,
) -> Vec<std::path::PathBuf> {
    TrainingRunArtifacts::new(destination, output_paths)
        .with_naming(
            output_model_basename.map(str::to_string),
            batch_name_template.map(str::to_string),
        )
        .conflicting_existing_artifacts()
}

pub fn reconcile_selected_device(app: &mut TrainerApp) -> Option<String> {
    let devices = match &app.python_status {
        PythonStatus::Ok { devices, .. } => devices,
        _ => return None,
    };
    if devices
        .iter()
        .any(|device| device.id == app.selected_device)
    {
        return None;
    }
    let fallback = devices
        .iter()
        .find(|device| device.id == "cpu")
        .or_else(|| devices.first())?;
    let previous = std::mem::replace(&mut app.selected_device, fallback.id.clone());
    Some(format!(
        "Selected device {previous} is unavailable. Falling back to {}.",
        fallback.name
    ))
}

fn validate_estimated_training_space(
    destination: &Path,
    input_path: Option<&Path>,
    output_paths: &[PathBuf],
    epochs: u32,
) -> Vec<String> {
    let mut required = output_paths
        .len()
        .max(1)
        .saturating_mul(TRAINING_ARTIFACT_BYTES_PER_OUTPUT as usize) as u64;
    required = required.saturating_add((epochs as u64).saturating_mul(1_000_000));
    if let Some(path) = input_path {
        required = required.saturating_add(file_len(path).unwrap_or(0));
    }
    for path in output_paths {
        required = required.saturating_add(file_len(path).unwrap_or(0));
    }
    required = required.max(MIN_FREE_DESTINATION_BYTES);
    match available_space_bytes(destination) {
        Ok(Some(bytes)) if bytes < required => vec![format!(
            "Output directory has less than the estimated {} MB required for this run",
            required / 1_000_000
        )],
        Ok(_) => Vec::new(),
        Err(error) => vec![format!(
            "Could not estimate output directory free space: {error}"
        )],
    }
}

fn metadata_has_values(metadata: &ModelMetadata) -> bool {
    !metadata.name.trim().is_empty()
        || !metadata.modeled_by.trim().is_empty()
        || !metadata.gear_make.trim().is_empty()
        || !metadata.gear_model.trim().is_empty()
        || metadata.gear_type.is_some()
        || metadata.tone_type.is_some()
        || !metadata.input_level_dbu.trim().is_empty()
        || !metadata.output_level_dbu.trim().is_empty()
}

fn file_len(path: &Path) -> std::io::Result<u64> {
    std::fs::metadata(path).map(|metadata| metadata.len())
}

#[cfg(unix)]
fn available_space_bytes(path: &Path) -> std::io::Result<Option<u64>> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path contains interior nul byte",
        )
    })?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is nul-terminated and `stat` points to writable storage.
    let result = unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `statvfs` returned success and initialized `stat`.
    let stat = unsafe { stat.assume_init() };
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    let available_blocks = u64::from(stat.f_bavail);
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let available_blocks = stat.f_bavail;
    Ok(Some(available_blocks.saturating_mul(stat.f_frsize)))
}

#[cfg(not(unix))]
fn available_space_bytes(_path: &Path) -> std::io::Result<Option<u64>> {
    Ok(None)
}
