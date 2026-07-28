use std::fmt;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::app::{PythonStatus, TrainerApp};
use crate::artifacts::TrainingRunArtifacts;

static RUN_ID_COUNTER: AtomicU64 = AtomicU64::new(0);
const TRAINING_MANIFEST_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub(crate) struct RunId(String);

impl RunId {
    pub(crate) fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ManifestStatus {
    Running,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct CleanupFailure {
    pub(crate) operation: String,
    pub(crate) path: PathBuf,
    pub(crate) message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct CleanupReport {
    pub(crate) failures: Vec<CleanupFailure>,
}

impl CleanupReport {
    pub(crate) fn is_complete(&self) -> bool {
        self.failures.is_empty()
    }

    fn push(&mut self, operation: &str, path: &Path, error: impl fmt::Display) {
        self.failures.push(CleanupFailure {
            operation: operation.to_string(),
            path: path.to_path_buf(),
            message: error.to_string(),
        });
    }
}

#[derive(Debug)]
struct RunLog {
    path: PathBuf,
    writer: Mutex<Option<BufWriter<std::fs::File>>>,
    first_error: Mutex<Option<String>>,
}

impl RunLog {
    fn new(path: PathBuf, writer: Option<BufWriter<std::fs::File>>) -> Self {
        Self {
            path,
            writer: Mutex::new(writer),
            first_error: Mutex::new(None),
        }
    }

    fn append(&self, line: &str) -> std::io::Result<()> {
        if let Some(message) = self
            .first_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            return Err(std::io::Error::other(message));
        }

        let result = (|| {
            let mut writer = self
                .writer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if writer.is_none() {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)?;
                *writer = Some(BufWriter::new(file));
            }
            if let Some(writer) = writer.as_mut() {
                writeln!(writer, "{line}")?;
                writer.flush()?;
            }
            Ok(())
        })();
        self.remember_error(&result);
        result
    }

    fn flush(&self) -> std::io::Result<()> {
        let result = self
            .writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
            .map_or(Ok(()), Write::flush);
        self.remember_error(&result);
        result
    }

    fn remember_error(&self, result: &std::io::Result<()>) {
        if let Err(error) = result {
            let mut first_error = self
                .first_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if first_error.is_none() {
                *first_error = Some(error.to_string());
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ActiveTrainingRun {
    pub(crate) id: RunId,
    pub(crate) log_path: PathBuf,
    pub(crate) manifest_path: PathBuf,
    staging_dir: PathBuf,
    reserved_paths: Vec<PathBuf>,
    log: Arc<RunLog>,
    snapshot: Option<RunSnapshot>,
}

impl ActiveTrainingRun {
    pub(crate) fn staging_dir(&self) -> &Path {
        &self.staging_dir
    }

    fn reservation_marker(&self) -> String {
        format!("NAM Trainer reservation for {}\n", self.id)
    }

    pub(crate) fn reserved_paths(&self) -> &[PathBuf] {
        &self.reserved_paths
    }

    pub(crate) fn append_log(&self, line: &str) -> std::io::Result<()> {
        self.log.append(line)
    }

    pub(crate) fn flush_log(&self) -> std::io::Result<()> {
        self.log.flush()
    }

    pub(crate) fn request(&self) -> Option<&crate::worker::protocol::TrainRequest> {
        self.snapshot.as_ref().map(|snapshot| &snapshot.request)
    }

    #[cfg(test)]
    pub(crate) fn capture_test_snapshot(&mut self, app: &TrainerApp) -> Result<(), String> {
        self.snapshot = Some(RunSnapshot::capture(
            app,
            crate::worker::build_train_request_for_run(
                app,
                self.id.to_string(),
                self.staging_dir.clone(),
            )
            .map_err(|error| error.to_string())?,
            unix_timestamp_secs(),
        ));
        Ok(())
    }

    pub(crate) fn from_existing(
        id: impl Into<String>,
        log_path: PathBuf,
        manifest_path: PathBuf,
    ) -> Self {
        let id = RunId::new(id);
        let staging_dir = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!(".nam-trainer-{id}-staging"));
        Self {
            id,
            log: Arc::new(RunLog::new(log_path.clone(), None)),
            log_path,
            manifest_path,
            staging_dir,
            reserved_paths: Vec::new(),
            snapshot: None,
        }
    }

    pub(crate) fn with_recovery_inventory(
        mut self,
        staging_dir: PathBuf,
        reserved_paths: Vec<PathBuf>,
    ) -> Self {
        self.staging_dir = staging_dir;
        self.reserved_paths = reserved_paths;
        self
    }
}

#[derive(Clone, Debug)]
struct RunSnapshot {
    started_unix_seconds: u64,
    app_version: String,
    package_name: String,
    os: String,
    arch: String,
    python_path: PathBuf,
    selected_device: String,
    nam_version: Option<String>,
    torch_version: Option<String>,
    packed_full_config_supported: bool,
    input_path: Option<PathBuf>,
    output_paths: Vec<PathBuf>,
    input_wav: Option<WavMetadata>,
    output_wavs: Vec<WavMetadata>,
    destination_dir: Option<PathBuf>,
    request: crate::worker::protocol::TrainRequest,
}

impl RunSnapshot {
    fn capture(
        app: &TrainerApp,
        request: crate::worker::protocol::TrainRequest,
        started_unix_seconds: u64,
    ) -> Self {
        let (nam_version, torch_version, packed_full_config_supported) = match &app.python_status {
            PythonStatus::Ok { report, .. } => (
                report.nam_version.clone(),
                report.torch_version.clone(),
                report.packed_full_config_supported,
            ),
            _ => (None, None, false),
        };
        Self {
            started_unix_seconds,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            package_name: env!("CARGO_PKG_NAME").to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            python_path: app.python_path.clone(),
            selected_device: app.selected_device.to_string(),
            nam_version,
            torch_version,
            packed_full_config_supported,
            input_path: app.input_path.clone(),
            output_paths: app.output_paths.clone(),
            input_wav: app.input_path.as_deref().and_then(read_wav_metadata),
            output_wavs: app
                .output_paths
                .iter()
                .filter_map(|path| read_wav_metadata(path))
                .collect(),
            destination_dir: app.destination_dir.clone(),
            request,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct TrainingRunManifest {
    #[serde(default = "manifest_schema_version")]
    schema_version: u32,
    run_id: RunId,
    status: ManifestStatus,
    #[serde(alias = "created_unix_seconds")]
    started_unix_seconds: u64,
    #[serde(default)]
    completed_unix_seconds: Option<u64>,
    app_version: String,
    package_name: String,
    os: String,
    arch: String,
    python_path: PathBuf,
    selected_device: String,
    nam_version: Option<String>,
    torch_version: Option<String>,
    packed_full_config_supported: bool,
    input_path: Option<PathBuf>,
    output_paths: Vec<PathBuf>,
    input_wav: Option<WavMetadata>,
    output_wavs: Vec<WavMetadata>,
    destination_dir: Option<PathBuf>,
    request: crate::worker::protocol::TrainRequest,
    model_path: Option<PathBuf>,
    staging_dir: PathBuf,
    reserved_paths: Vec<PathBuf>,
    #[serde(default)]
    cleanup_failures: Vec<CleanupFailure>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WavMetadata {
    path: PathBuf,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    duration_seconds: f64,
}

#[cfg(test)]
pub(crate) fn save_training_log(
    model_path: &Path,
    active_run: Option<&ActiveTrainingRun>,
    lines: &[String],
) -> std::io::Result<PathBuf> {
    let log_path = model_path.with_extension("training.log");
    if let Some(run) = active_run {
        verify_reservation(run, &log_path)?;
        let contents = std::fs::read(&run.log_path)?;
        crate::persistence::atomic_write(&log_path, &contents)?;
    } else {
        crate::persistence::atomic_write(&log_path, lines.join("\n").as_bytes())?;
    }
    Ok(log_path)
}

pub(crate) fn prepare_training_run(app: &TrainerApp) -> std::io::Result<ActiveTrainingRun> {
    let destination = app.destination_dir.as_deref().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing destination")
    })?;
    let run_id = RunId::new(make_run_id());
    let log_path = destination.join(format!("{run_id}.training.log"));
    let manifest_path = destination.join(format!("{run_id}.training-manifest.json"));
    let staging_dir = destination.join(format!(".nam-trainer-{run_id}-staging"));
    let request =
        crate::worker::build_train_request_for_run(app, run_id.to_string(), staging_dir.clone())
            .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let snapshot = RunSnapshot::capture(app, request, unix_timestamp_secs());
    let log_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&log_path)?;
    let mut run = ActiveTrainingRun {
        id: run_id,
        log: Arc::new(RunLog::new(
            log_path.clone(),
            Some(BufWriter::new(log_file)),
        )),
        log_path,
        manifest_path,
        staging_dir,
        reserved_paths: Vec::new(),
        snapshot: Some(snapshot),
    };
    run.reserved_paths = match reserve_output_artifacts(app, &run) {
        Ok(paths) => paths,
        Err(error) => {
            let _ = std::fs::remove_file(&run.log_path);
            return Err(error);
        }
    };
    if let Err(error) = std::fs::create_dir(run.staging_dir()) {
        let _ = std::fs::remove_file(&run.log_path);
        let _ = cleanup_run_resources(Some(&run));
        return Err(error);
    }
    if let Err(error) = save_training_manifest(
        &run.manifest_path,
        &run,
        ManifestStatus::Running,
        None,
        &CleanupReport::default(),
    ) {
        let _ = std::fs::remove_file(&run.log_path);
        let _ = cleanup_run_resources(Some(&run));
        return Err(error);
    }
    Ok(run)
}

fn reserve_output_artifacts(
    app: &TrainerApp,
    run: &ActiveTrainingRun,
) -> std::io::Result<Vec<PathBuf>> {
    if app.allow_overwrite_outputs {
        return Ok(Vec::new());
    }
    let destination = app.destination_dir.as_deref().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing destination")
    })?;
    let artifacts = TrainingRunArtifacts::new(destination, &app.output_paths).with_naming(
        non_empty(&app.config.output_model_basename),
        non_empty(&app.config.batch_name_template),
    );
    let paths = artifacts
        .predicted_model_paths()
        .into_iter()
        .flat_map(|model_path| {
            [
                model_path.clone(),
                TrainingRunArtifacts::log_path_for_model(&model_path),
                TrainingRunArtifacts::manifest_path_for_model(&model_path),
            ]
        });
    let marker = run.reservation_marker();
    let mut reserved = Vec::new();
    for path in paths {
        let result = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .and_then(|mut file| {
                file.write_all(marker.as_bytes())?;
                file.sync_all()
            });
        if let Err(error) = result {
            let _ = cleanup_reserved_paths(&reserved, &marker);
            return Err(std::io::Error::new(
                error.kind(),
                format!(
                    "could not reserve output artifact {}: {error}",
                    path.display()
                ),
            ));
        }
        reserved.push(path);
    }
    Ok(reserved)
}

#[cfg(test)]
pub(crate) fn promote_staged_model(
    active_run: Option<&ActiveTrainingRun>,
    destination: Option<&Path>,
    model_path: &Path,
) -> std::io::Result<PathBuf> {
    let source = model_path;
    let Some(run) = active_run else {
        return Ok(source.to_path_buf());
    };
    if !source.starts_with(run.staging_dir()) {
        return Ok(source.to_path_buf());
    }
    let destination = destination.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing destination")
    })?;
    let file_name = source.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("trained model path has no file name: {}", source.display()),
        )
    })?;
    let published = destination.join(file_name);
    verify_reservation(run, &published)?;
    crate::persistence::atomic_promote(source, &published)?;
    Ok(published)
}

fn verify_reservation(run: &ActiveTrainingRun, path: &Path) -> std::io::Result<()> {
    if run.reserved_paths.iter().any(|reserved| reserved == path) {
        let contents = std::fs::read_to_string(path)?;
        if contents != run.reservation_marker() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "reserved output was modified by another process: {}",
                    path.display()
                ),
            ));
        }
    }
    Ok(())
}

fn cleanup_reserved_paths(paths: &[PathBuf], marker: &str) -> CleanupReport {
    let mut report = CleanupReport::default();
    for path in paths {
        match std::fs::read(path) {
            Ok(contents) if contents == marker.as_bytes() => {
                if let Err(error) = std::fs::remove_file(path) {
                    report.push("remove reservation", path, error);
                }
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => report.push("inspect reservation", path, error),
        }
    }
    report
}

pub(crate) fn cleanup_run_resources(active_run: Option<&ActiveTrainingRun>) -> CleanupReport {
    let mut report = CleanupReport::default();
    if let Some(run) = active_run {
        if let Err(error) = std::fs::remove_dir_all(run.staging_dir()) {
            if error.kind() != std::io::ErrorKind::NotFound {
                report.push("remove staging directory", run.staging_dir(), error);
            }
        }
        report.failures.extend(
            cleanup_reserved_paths(&run.reserved_paths, &run.reservation_marker()).failures,
        );
    }
    report
}

pub(crate) fn publish_artifact_bundle(
    staged_model_path: &Path,
    run: &ActiveTrainingRun,
    destination: &Path,
) -> std::io::Result<(PathBuf, PathBuf, PathBuf)> {
    let file_name = staged_model_path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "trained model path has no file name: {}",
                staged_model_path.display()
            ),
        )
    })?;
    let model_path = if staged_model_path.starts_with(run.staging_dir()) {
        destination.join(file_name)
    } else {
        staged_model_path.to_path_buf()
    };
    let log_path = TrainingRunArtifacts::log_path_for_model(&model_path);
    let manifest_path = TrainingRunArtifacts::manifest_path_for_model(&model_path);
    let staged_log = run
        .staging_dir()
        .join(log_path.file_name().unwrap_or_default());
    let staged_manifest = run
        .staging_dir()
        .join(manifest_path.file_name().unwrap_or_default());

    for path in [&model_path, &log_path, &manifest_path] {
        verify_reservation(run, path)?;
    }
    run.flush_log()?;
    crate::persistence::atomic_write(&staged_log, &std::fs::read(&run.log_path)?)?;
    save_training_manifest(
        &staged_manifest,
        run,
        ManifestStatus::Complete,
        Some(&model_path),
        &CleanupReport::default(),
    )?;

    if staged_model_path != model_path {
        crate::persistence::atomic_promote(staged_model_path, &model_path)?;
    }
    crate::persistence::atomic_promote(&staged_log, &log_path)?;
    crate::persistence::atomic_promote(&staged_manifest, &manifest_path)?;
    Ok((model_path, log_path, manifest_path))
}

pub(crate) fn save_training_manifest(
    path: &Path,
    run: &ActiveTrainingRun,
    status: ManifestStatus,
    model_path: Option<&Path>,
    cleanup_report: &CleanupReport,
) -> std::io::Result<()> {
    let snapshot = run.snapshot.as_ref().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "training run has no immutable launch snapshot",
        )
    })?;
    let manifest = TrainingRunManifest {
        schema_version: TRAINING_MANIFEST_SCHEMA_VERSION,
        run_id: run.id.clone(),
        status,
        started_unix_seconds: snapshot.started_unix_seconds,
        completed_unix_seconds: (status != ManifestStatus::Running).then(unix_timestamp_secs),
        app_version: snapshot.app_version.clone(),
        package_name: snapshot.package_name.clone(),
        os: snapshot.os.clone(),
        arch: snapshot.arch.clone(),
        python_path: snapshot.python_path.clone(),
        selected_device: snapshot.selected_device.clone(),
        nam_version: snapshot.nam_version.clone(),
        torch_version: snapshot.torch_version.clone(),
        packed_full_config_supported: snapshot.packed_full_config_supported,
        input_path: snapshot.input_path.clone(),
        output_paths: snapshot.output_paths.clone(),
        input_wav: snapshot.input_wav.clone(),
        output_wavs: snapshot.output_wavs.clone(),
        destination_dir: snapshot.destination_dir.clone(),
        request: snapshot.request.clone(),
        model_path: model_path.map(Path::to_path_buf),
        staging_dir: run.staging_dir.clone(),
        reserved_paths: run.reserved_paths.clone(),
        cleanup_failures: cleanup_report.failures.clone(),
    };
    let json = serde_json::to_string_pretty(&manifest).map_err(std::io::Error::other)?;
    crate::persistence::atomic_write(path, json.as_bytes())
}

const fn manifest_schema_version() -> u32 {
    TRAINING_MANIFEST_SCHEMA_VERSION
}

pub fn validate_training_manifest_json(json: &str) -> serde_json::Result<()> {
    serde_json::from_str::<TrainingRunManifest>(json).map(|_| ())
}

pub(crate) fn make_run_id() -> String {
    let counter = RUN_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "nam-training-run-{}-{}-{counter}",
        unix_timestamp_nanos(),
        std::process::id(),
    )
}

pub(crate) fn unix_timestamp_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn read_wav_metadata(path: &Path) -> Option<WavMetadata> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let samples = reader.len() as f64;
    let channels = spec.channels.max(1) as f64;
    let sample_rate = spec.sample_rate.max(1) as f64;
    Some(WavMetadata {
        path: path.to_path_buf(),
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: spec.bits_per_sample,
        duration_seconds: samples / channels / sample_rate,
    })
}

pub(crate) fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn non_empty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::{cleanup_run_resources, promote_staged_model, ActiveTrainingRun};

    #[test]
    fn staged_model_replaces_only_its_owned_reservation() {
        let (_temp, directory) = unique_test_dir("promotion");
        std::fs::create_dir_all(&directory).unwrap();
        let mut run = ActiveTrainingRun::from_existing(
            "run",
            directory.join("run.log"),
            directory.join("run.json"),
        );
        std::fs::create_dir(run.staging_dir()).unwrap();
        let source = run.staging_dir().join("model.nam");
        let destination = directory.join("model.nam");
        std::fs::write(&source, b"model").unwrap();
        std::fs::write(&destination, run.reservation_marker()).unwrap();
        run.reserved_paths.push(destination.clone());

        let published = promote_staged_model(Some(&run), Some(&directory), &source).unwrap();

        assert_eq!(published, destination);
        assert_eq!(std::fs::read(&published).unwrap(), b"model");
        assert!(!source.exists());
        assert!(cleanup_run_resources(Some(&run)).is_complete());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn modified_reservation_blocks_model_publication() {
        let (_temp, directory) = unique_test_dir("tamper");
        std::fs::create_dir_all(&directory).unwrap();
        let mut run = ActiveTrainingRun::from_existing(
            "run",
            directory.join("run.log"),
            directory.join("run.json"),
        );
        std::fs::create_dir(run.staging_dir()).unwrap();
        let source = run.staging_dir().join("model.nam");
        let destination = directory.join("model.nam");
        std::fs::write(&source, b"model").unwrap();
        std::fs::write(&destination, b"other process").unwrap();
        run.reserved_paths.push(destination.clone());

        let error = promote_staged_model(Some(&run), Some(&directory), &source).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&destination).unwrap(), b"other process");
        assert!(source.exists());
        assert!(cleanup_run_resources(Some(&run)).is_complete());
        assert_eq!(std::fs::read(&destination).unwrap(), b"other process");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cleanup_report_retains_failed_paths() {
        let (_temp, directory) = unique_test_dir("cleanup_report");
        std::fs::create_dir_all(&directory).unwrap();
        let run = ActiveTrainingRun::from_existing(
            "run",
            directory.join("run.log"),
            directory.join("run.json"),
        );
        std::fs::write(run.staging_dir(), b"not a directory").unwrap();

        let report = cleanup_run_resources(Some(&run));

        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].operation, "remove staging directory");
        assert_eq!(report.failures[0].path, run.staging_dir());
        assert!(run.staging_dir().exists());

        std::fs::remove_dir_all(directory).unwrap();
    }

    fn unique_test_dir(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::Builder::new()
            .prefix(&format!("nam-trainer-run-manifest-{name}-"))
            .tempdir()
            .unwrap();
        let path = temp.path().to_path_buf();
        (temp, path)
    }
}
