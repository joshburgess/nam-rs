mod channel;
pub mod protocol;
mod publication;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::app::{HideConsoleExt, TrainerApp};
use crate::background::ManagedTask;
pub use channel::WorkerMessageReceiver;
pub(crate) use channel::{send_worker_message, worker_message_channel, WorkerMessageSender};
use publication::{prepare_worker_message, PublicationContext};

const MAX_STDERR_LOG_LINES: usize = 500;
const WORKER_LOG_CAPACITY: usize = 512;
const PROTOCOL_INPUT_CAPACITY: usize = 256;
const MAX_PROTOCOL_REORDER_WINDOW: u64 = 64;
const COOPERATIVE_CANCEL_GRACE: Duration = Duration::from_secs(3);

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
enum ProtocolSequenceError {
    #[error("worker event run ID {actual} does not match active run {expected}")]
    RunIdMismatch { actual: String, expected: String },
    #[error("worker emitted an event after the terminal event")]
    EventAfterTerminal,
    #[error("worker event sequence {actual} is not greater than the previous sequence {previous}")]
    NonIncreasingSequence { actual: u64, previous: u64 },
    #[error(
        "worker event sequence {actual} exceeds the maximum accepted sequence {maximum} while waiting for sequence {expected}"
    )]
    SequenceJump {
        actual: u64,
        expected: u64,
        maximum: u64,
    },
    #[error("worker output ended while waiting for sequence {expected}; next received sequence was {next}")]
    MissingSequence { expected: u64, next: u64 },
    #[error("{event} event is missing its file index")]
    MissingFileIndex { event: &'static str },
    #[error("worker started a file before completing the active file")]
    FileAlreadyActive,
    #[error("{event} file index {actual} does not match active file {active}")]
    FileIndexMismatch {
        event: &'static str,
        actual: usize,
        active: usize,
    },
    #[error("{event} arrived before training_start")]
    EventBeforeTrainingStart { event: &'static str },
    #[error("all_complete arrived while a file was still active")]
    AllCompleteWhileFileActive,
    #[error("all_complete must not identify a file")]
    AllCompleteHasFileIndex,
    #[error("log event file index does not match the active file")]
    LogFileIndexMismatch,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerRequestError {
    #[error("an input WAV is required")]
    MissingInputWav,
    #[error("at least one output WAV is required")]
    MissingOutputWavs,
    #[error("a training destination is required")]
    MissingDestination,
    #[error("{label} path cannot be represented as Unicode for the Python worker: {}", path.display())]
    NonUnicodePath { label: &'static str, path: PathBuf },
}

impl WorkerRequestError {
    fn kind(&self) -> WorkerFailureKind {
        match self {
            Self::MissingInputWav | Self::MissingOutputWavs | Self::MissingDestination => {
                WorkerFailureKind::MissingArtifact
            }
            Self::NonUnicodePath { .. } => WorkerFailureKind::UnsupportedPath,
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum WorkerStartupError {
    #[error(transparent)]
    InvalidRequest(#[from] WorkerRequestError),
    #[error("failed to serialize training request: {source}")]
    SerializeRequest {
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to write worker script to {}: {source}", path.display())]
    WriteScript {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to start Python worker at {}: {source}", python.display())]
    Launch {
        python: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Python worker stdin was not available")]
    MissingStdin,
    #[error("failed to send request to Python worker: {source}")]
    SendRequest {
        #[source]
        source: std::io::Error,
    },
}

impl WorkerStartupError {
    fn kind(&self) -> WorkerFailureKind {
        match self {
            Self::InvalidRequest(error) => error.kind(),
            Self::SerializeRequest { .. } => WorkerFailureKind::Serialization,
            Self::WriteScript { .. } | Self::MissingStdin | Self::SendRequest { .. } => {
                WorkerFailureKind::Subprocess
            }
            Self::Launch { .. } => WorkerFailureKind::Launch,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkerFailureKind {
    Reported(protocol::WorkerErrorKind),
    MissingArtifact,
    UnsupportedPath,
    Launch,
    Serialization,
    Subprocess,
    Protocol,
    ProtocolSequence,
}

impl From<protocol::WorkerErrorKind> for WorkerFailureKind {
    fn from(kind: protocol::WorkerErrorKind) -> Self {
        Self::Reported(kind)
    }
}

#[derive(Debug)]
struct ProtocolSequenceValidator {
    run_id: String,
    last_sequence: Option<u64>,
    active_file_index: Option<usize>,
    terminal: bool,
}

impl ProtocolSequenceValidator {
    fn new(run_id: String) -> Self {
        Self {
            run_id,
            last_sequence: None,
            active_file_index: None,
            terminal: false,
        }
    }

    fn validate(&mut self, event: &protocol::WorkerEvent) -> Result<(), ProtocolSequenceError> {
        if event.run_id() != self.run_id {
            return Err(ProtocolSequenceError::RunIdMismatch {
                actual: event.run_id().to_string(),
                expected: self.run_id.clone(),
            });
        }
        if self.terminal {
            return Err(ProtocolSequenceError::EventAfterTerminal);
        }
        if let Some(previous) = self
            .last_sequence
            .filter(|previous| event.sequence() <= *previous)
        {
            return Err(ProtocolSequenceError::NonIncreasingSequence {
                actual: event.sequence(),
                previous,
            });
        }

        match event {
            protocol::WorkerEvent::TrainingStart { .. } => {
                let file_index =
                    event
                        .file_index()
                        .ok_or(ProtocolSequenceError::MissingFileIndex {
                            event: "training_start",
                        })?;
                if self.active_file_index.replace(file_index).is_some() {
                    return Err(ProtocolSequenceError::FileAlreadyActive);
                }
            }
            protocol::WorkerEvent::EpochEnd { .. } => {
                self.require_active_file(event, "epoch_end")?;
            }
            protocol::WorkerEvent::TrainingComplete { .. }
            | protocol::WorkerEvent::TrainingFailed { .. } => {
                self.require_active_file(event, "file completion")?;
                self.active_file_index = None;
            }
            protocol::WorkerEvent::AllComplete { .. } => {
                if self.active_file_index.is_some() {
                    return Err(ProtocolSequenceError::AllCompleteWhileFileActive);
                }
                if event.file_index().is_some() {
                    return Err(ProtocolSequenceError::AllCompleteHasFileIndex);
                }
                self.terminal = true;
            }
            protocol::WorkerEvent::Error { .. } => {
                self.terminal = true;
            }
            protocol::WorkerEvent::Log { .. } => {
                if let Some(file_index) = event.file_index() {
                    if self.active_file_index != Some(file_index) {
                        return Err(ProtocolSequenceError::LogFileIndexMismatch);
                    }
                }
            }
        }
        self.last_sequence = Some(event.sequence());
        Ok(())
    }

    fn require_active_file(
        &self,
        event: &protocol::WorkerEvent,
        event_name: &'static str,
    ) -> Result<(), ProtocolSequenceError> {
        let file_index = event
            .file_index()
            .ok_or(ProtocolSequenceError::MissingFileIndex { event: event_name })?;
        match self.active_file_index {
            Some(active) if active == file_index => Ok(()),
            Some(active) => Err(ProtocolSequenceError::FileIndexMismatch {
                event: event_name,
                actual: file_index,
                active,
            }),
            None => Err(ProtocolSequenceError::EventBeforeTrainingStart { event: event_name }),
        }
    }
}

enum ProtocolInput {
    StdoutEvent(protocol::WorkerEvent),
    StderrEvent(protocol::WorkerEvent),
    StdoutLog(String),
    StderrLog(String),
}

struct ProtocolDispatcher {
    validator: ProtocolSequenceValidator,
    pending: BTreeMap<u64, protocol::WorkerEvent>,
    next_sequence: u64,
    stderr_log_lines: usize,
    stderr_suppression_sent: bool,
}

impl ProtocolDispatcher {
    fn new(run_id: String) -> Self {
        Self {
            validator: ProtocolSequenceValidator::new(run_id),
            pending: BTreeMap::new(),
            next_sequence: 1,
            stderr_log_lines: 0,
            stderr_suppression_sent: false,
        }
    }

    fn accept(&mut self, input: ProtocolInput) -> Vec<WorkerMessage> {
        match input {
            ProtocolInput::StdoutEvent(event) | ProtocolInput::StderrEvent(event) => {
                self.accept_event(event)
            }
            ProtocolInput::StdoutLog(line) => vec![WorkerMessage::Log(line)],
            ProtocolInput::StderrLog(line) => self.accept_stderr_log(line),
        }
    }

    fn finish(&mut self) -> Vec<WorkerMessage> {
        let Some(next) = self
            .pending
            .first_key_value()
            .map(|(sequence, _)| *sequence)
        else {
            return Vec::new();
        };
        self.pending.clear();
        vec![Self::sequence_error(
            ProtocolSequenceError::MissingSequence {
                expected: self.next_sequence,
                next,
            },
        )]
    }

    fn accept_event(&mut self, event: protocol::WorkerEvent) -> Vec<WorkerMessage> {
        let sequence = event.sequence();
        if sequence < self.next_sequence || self.pending.contains_key(&sequence) {
            let previous = self.next_sequence.saturating_sub(1);
            return vec![WorkerMessage::Error {
                kind: WorkerFailureKind::ProtocolSequence,
                message: ProtocolSequenceError::NonIncreasingSequence {
                    actual: sequence,
                    previous,
                }
                .to_string(),
            }];
        }
        let maximum = self
            .next_sequence
            .saturating_add(MAX_PROTOCOL_REORDER_WINDOW);
        if sequence > maximum {
            return vec![Self::sequence_error(ProtocolSequenceError::SequenceJump {
                actual: sequence,
                expected: self.next_sequence,
                maximum,
            })];
        }
        self.pending.insert(sequence, event);

        let mut messages = Vec::new();
        while let Some(event) = self.pending.remove(&self.next_sequence) {
            messages.push(self.validate_event(event));
            self.next_sequence = self.next_sequence.saturating_add(1);
        }
        messages
    }

    fn sequence_error(error: ProtocolSequenceError) -> WorkerMessage {
        WorkerMessage::Error {
            kind: WorkerFailureKind::ProtocolSequence,
            message: error.to_string(),
        }
    }

    fn accept_stderr_log(&mut self, line: String) -> Vec<WorkerMessage> {
        self.stderr_log_lines = self.stderr_log_lines.saturating_add(1);
        if self.stderr_log_lines <= MAX_STDERR_LOG_LINES {
            return vec![WorkerMessage::Log(line)];
        }
        if !self.stderr_suppression_sent {
            self.stderr_suppression_sent = true;
            return vec![WorkerMessage::Log(format!(
                "Additional stderr output suppressed after {MAX_STDERR_LOG_LINES} lines."
            ))];
        }
        Vec::new()
    }

    fn validate_event(&mut self, event: protocol::WorkerEvent) -> WorkerMessage {
        if let Err(error) = self.validator.validate(&event) {
            return WorkerMessage::Error {
                kind: WorkerFailureKind::ProtocolSequence,
                message: error.to_string(),
            };
        }
        event_to_message(event)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TrainingState {
    Idle,
    Training,
    Complete,
    Error(String),
}

#[derive(Debug, PartialEq)]
pub enum WorkerMessage {
    Log(String),
    TrainingStart {
        file: String,
        total_epochs: u32,
    },
    EpochEnd {
        epoch: u32,
        train_loss: f64,
        val_loss: f64,
        esr: f64,
    },
    FileCompleted {
        file: String,
        validation_esr: f64,
        model_path: String,
    },
    FilePublished {
        file: String,
        validation_esr: f64,
        model_path: PathBuf,
        log_path: PathBuf,
        manifest_path: PathBuf,
    },
    FilePublicationFailed {
        file: String,
        error: String,
    },
    FileFailed {
        file: String,
        kind: protocol::WorkerErrorKind,
        error: String,
    },
    RunCompleted,
    Error {
        kind: WorkerFailureKind,
        message: String,
    },
    WorkerExited {
        exit_code: Option<i32>,
    },
}

pub struct WorkerHandle {
    cancelled: Arc<AtomicBool>,
    task: Option<ManagedTask>,
}

impl WorkerHandle {
    pub(crate) fn from_join(cancelled: Arc<AtomicBool>, join: thread::JoinHandle<()>) -> Self {
        Self {
            cancelled: Arc::clone(&cancelled),
            task: Some(ManagedTask::from_join(cancelled, join)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(task) = &self.task {
            task.cancel();
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn send_worker_startup_error(tx: &WorkerMessageSender, error: WorkerStartupError) {
    let _ = send_worker_message(
        tx,
        WorkerMessage::Error {
            kind: error.kind(),
            message: error.to_string(),
        },
    );
}

/// Spawn the Python worker subprocess and return a handle + message receiver.
pub fn spawn(app: &TrainerApp) -> (WorkerHandle, WorkerMessageReceiver) {
    let (tx, rx) = worker_message_channel(WORKER_LOG_CAPACITY);
    let cancelled = Arc::clone(&tx.cancelled);
    let handle = WorkerHandle {
        cancelled: Arc::clone(&cancelled),
        task: None,
    };

    let request = app
        .run
        .data()
        .artifacts
        .as_ref()
        .and_then(crate::run_manifest::ActiveTrainingRun::request)
        .cloned();
    let request = match request.map(Ok).unwrap_or_else(|| build_train_request(app)) {
        Ok(request) => request,
        Err(error) => {
            send_worker_startup_error(&tx, error.into());
            let _ = send_worker_message(&tx, WorkerMessage::WorkerExited { exit_code: None });
            return (handle, rx);
        }
    };
    let protocol_run_id = request.run_id.clone();

    let request_json = match serde_json::to_string(&request) {
        Ok(json) => json,
        Err(source) => {
            send_worker_startup_error(&tx, WorkerStartupError::SerializeRequest { source });
            let _ = send_worker_message(&tx, WorkerMessage::WorkerExited { exit_code: None });
            return (handle, rx);
        }
    };
    let python_path = app.python_path.clone();
    let publication_context = app
        .run
        .data()
        .artifacts
        .clone()
        .zip(app.destination_dir.clone())
        .map(|(run, destination)| Arc::new(PublicationContext { run, destination }));

    let worker_cancelled = Arc::clone(&cancelled);
    let join = thread::spawn(move || {
        // Write the embedded worker script to a temp file so we can run it
        // as `python /path/to/script.py` rather than `python -c "..."`.
        // The -c form can have subtle encoding and import issues on Windows.
        let script_path = match create_worker_script() {
            Ok(path) => path,
            Err((path, source)) => {
                send_worker_startup_error(&tx, WorkerStartupError::WriteScript { path, source });
                let _ = send_worker_message(&tx, WorkerMessage::WorkerExited { exit_code: None });
                return;
            }
        };

        let result = Command::new(&python_path)
            .arg(script_path.as_os_str())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .hide_console()
            .spawn();

        let mut child = match result {
            Ok(c) => c,
            Err(source) => {
                send_worker_startup_error(
                    &tx,
                    WorkerStartupError::Launch {
                        python: python_path,
                        source,
                    },
                );
                let _ = send_worker_message(&tx, WorkerMessage::WorkerExited { exit_code: None });
                return;
            }
        };

        let mut child_stdin = child.stdin.take();
        let request_write = child_stdin
            .as_mut()
            .ok_or(WorkerStartupError::MissingStdin)
            .and_then(|stdin| {
                writeln!(stdin, "{request_json}")
                    .and_then(|_| stdin.flush())
                    .map_err(|source| WorkerStartupError::SendRequest { source })
            });
        if let Err(error) = request_write {
            send_worker_startup_error(&tx, error);
            let _ = child.kill();
            let exit_code = child.wait().ok().and_then(|status| status.code());
            let _ = send_worker_message(&tx, WorkerMessage::WorkerExited { exit_code });
            return;
        }

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let (protocol_tx, protocol_rx) =
            crossbeam_channel::bounded::<ProtocolInput>(PROTOCOL_INPUT_CAPACITY);
        let dispatcher_tx = tx.clone();
        let dispatcher_publication_context = publication_context.clone();
        let dispatcher_thread = thread::spawn(move || {
            let mut dispatcher = ProtocolDispatcher::new(protocol_run_id);
            for input in protocol_rx {
                for message in dispatcher.accept(input) {
                    let message =
                        prepare_worker_message(message, dispatcher_publication_context.as_deref());
                    if !send_worker_message(&dispatcher_tx, message) {
                        return;
                    }
                }
            }
            for message in dispatcher.finish() {
                let message =
                    prepare_worker_message(message, dispatcher_publication_context.as_deref());
                if !send_worker_message(&dispatcher_tx, message) {
                    return;
                }
            }
        });

        // Drain stderr in a background thread to prevent pipe buffer
        // deadlock. PyTorch/Lightning write progress and warnings to
        // stderr; on Windows the pipe buffer is 64KB and fills up over
        // long training runs, blocking the Python process. Also collect
        // the last few lines for diagnostics if the worker crashes.
        let last_stderr: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let last_stderr_writer = Arc::clone(&last_stderr);
        let protocol_stderr = protocol_tx.clone();
        let stderr_thread = stderr.map(|stderr| {
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    if line.trim().is_empty() {
                        continue;
                    }
                    // Try to parse as a JSON protocol event first. The
                    // worker falls back to stderr when stdout is broken
                    // (e.g. after a CUDA crash), so training_failed
                    // events may arrive here instead of stdout.
                    let input = match serde_json::from_str::<protocol::WorkerEvent>(&line) {
                        Ok(event) => ProtocolInput::StderrEvent(event),
                        Err(_) => ProtocolInput::StderrLog(line.clone()),
                    };
                    if protocol_stderr.send(input).is_err() {
                        break;
                    }
                    let mut buf = last_stderr_writer.lock();
                    buf.push(line);
                    if buf.len() > 20 {
                        buf.remove(0);
                    }
                }
            })
        });

        let protocol_stdout = protocol_tx.clone();
        let stdout_thread = stdout.map(|stdout| {
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let input = match serde_json::from_str::<protocol::WorkerEvent>(&line) {
                        Ok(event) => ProtocolInput::StdoutEvent(event),
                        Err(_) => ProtocolInput::StdoutLog(line),
                    };
                    if protocol_stdout.send(input).is_err() {
                        break;
                    }
                }
            })
        });
        drop(protocol_tx);

        // This thread is the sole owner of Child. Cancellation is queued,
        // including when it arrives before process startup completes.
        let exit_code = wait_for_child(&mut child, &worker_cancelled, child_stdin);

        if let Some(thread) = stdout_thread {
            let _ = thread.join();
        }
        if let Some(thread) = stderr_thread {
            let _ = thread.join();
        }
        let _ = dispatcher_thread.join();

        // If exit was non-zero and we never sent a completion/error event,
        // include the last stderr lines to help diagnose the crash.
        if exit_code.unwrap_or(0) != 0 {
            let buf = last_stderr.lock();
            if !buf.is_empty() {
                let tail = buf.join("\n");
                let _ = send_worker_message(
                    &tx,
                    WorkerMessage::Error {
                        kind: WorkerFailureKind::Subprocess,
                        message: format!(
                            "Python exited with code {}. Last output:\n{tail}",
                            exit_code.unwrap_or(-1)
                        ),
                    },
                );
            }
        }

        let _ = send_worker_message(&tx, WorkerMessage::WorkerExited { exit_code });
    });

    (WorkerHandle::from_join(cancelled, join), rx)
}

fn wait_for_child(
    child: &mut Child,
    cancelled: &AtomicBool,
    mut child_stdin: Option<std::process::ChildStdin>,
) -> Option<i32> {
    let mut force_cancel_at = None;
    loop {
        if cancelled.load(Ordering::Acquire) && force_cancel_at.is_none() {
            if let Some(mut stdin) = child_stdin.take() {
                let cooperative_request =
                    writeln!(stdin, r#"{{"command":"cancel"}}"#).and_then(|_| stdin.flush());
                if cooperative_request.is_ok() {
                    force_cancel_at = Some(Instant::now() + COOPERATIVE_CANCEL_GRACE);
                } else {
                    force_cancel_at = Some(Instant::now());
                }
            } else {
                force_cancel_at = Some(Instant::now());
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => return status.code(),
            Ok(None) => {
                if force_cancel_at.is_some_and(|deadline| Instant::now() >= deadline) {
                    let _ = child.kill();
                    return child.wait().ok().and_then(|status| status.code());
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => return child.wait().ok().and_then(|status| status.code()),
        }
    }
}

#[cfg(test)]
fn parse_worker_line(line: String) -> WorkerMessage {
    match serde_json::from_str::<protocol::WorkerEvent>(&line) {
        Ok(event) => event_to_message(event),
        Err(_) => WorkerMessage::Log(line),
    }
}

pub fn build_train_request(app: &TrainerApp) -> Result<protocol::TrainRequest, WorkerRequestError> {
    let run_id = app
        .run
        .data()
        .artifacts
        .as_ref()
        .map(|run| run.id.to_string())
        .unwrap_or_else(crate::run_manifest::make_run_id);
    let destination = app
        .run
        .data()
        .artifacts
        .as_ref()
        .map(|run| run.staging_dir().to_path_buf())
        .or_else(|| app.destination_dir.clone())
        .ok_or(WorkerRequestError::MissingDestination)?;
    build_train_request_for_run(app, run_id, destination)
}

pub(crate) fn build_train_request_for_run(
    app: &TrainerApp,
    run_id: String,
    destination: PathBuf,
) -> Result<protocol::TrainRequest, WorkerRequestError> {
    let input_path = app
        .input_path
        .as_deref()
        .ok_or(WorkerRequestError::MissingInputWav)
        .and_then(|path| encode_worker_path(path, "input WAV"))?;
    if app.output_paths.is_empty() {
        return Err(WorkerRequestError::MissingOutputWavs);
    }
    let output_paths = app
        .output_paths
        .iter()
        .map(|path| encode_worker_path(path, "output WAV"))
        .collect::<Result<Vec<_>, _>>()?;
    if destination.as_os_str().is_empty() {
        return Err(WorkerRequestError::MissingDestination);
    }
    let destination = encode_worker_path(&destination, "training destination")?;
    Ok(protocol::TrainRequest {
        protocol_version: protocol::PROTOCOL_VERSION,
        run_id,
        input_path,
        output_paths,
        destination,
        output_model_basename: non_empty(&app.config.output_model_basename),
        batch_name_template: non_empty(&app.config.batch_name_template),
        architecture: app.config.architecture.as_str().to_string(),
        packed: app.config.architecture == crate::app::Architecture::Packed,
        epochs: app.config.epochs,
        batch_size: app.config.batch_size,
        lr: app.config.lr,
        lr_decay: app.config.lr_decay,
        latency: app.config.latency,
        threshold_esr: app.config.threshold_esr,
        save_plot: app.config.save_plot,
        fit_mrstft: app.config.fit_mrstft,
        ignore_checks: app.config.ignore_checks,
        num_output_samples_per_datum: app.config.num_output_samples_per_datum,
        use_full_config_trainer: app.config.use_full_config_trainer,
        device: app.selected_device.to_string(),
        metadata: protocol::MetadataRequest {
            name: non_empty(&app.metadata.name),
            modeled_by: non_empty(&app.metadata.modeled_by),
            gear_make: non_empty(&app.metadata.gear_make),
            gear_model: non_empty(&app.metadata.gear_model),
            gear_type: app.metadata.gear_type.map(|g| g.as_str().to_string()),
            tone_type: app.metadata.tone_type.map(|t| t.as_str().to_string()),
            input_level_dbu: app.metadata.input_level_dbu.parse::<f64>().ok(),
            output_level_dbu: app.metadata.output_level_dbu.parse::<f64>().ok(),
        },
    })
}

fn encode_worker_path(path: &Path, label: &'static str) -> Result<String, WorkerRequestError> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| WorkerRequestError::NonUnicodePath {
            label,
            path: path.to_path_buf(),
        })
}

pub(crate) fn event_to_message(event: protocol::WorkerEvent) -> WorkerMessage {
    if event.protocol_version() != protocol::PROTOCOL_VERSION {
        return WorkerMessage::Error {
            kind: WorkerFailureKind::Protocol,
            message: format!(
                "Unsupported worker protocol version {}; expected {}",
                event.protocol_version(),
                protocol::PROTOCOL_VERSION
            ),
        };
    }

    match event {
        protocol::WorkerEvent::EpochEnd {
            epoch,
            train_loss,
            val_loss,
            esr,
            ..
        } => WorkerMessage::EpochEnd {
            epoch,
            train_loss,
            val_loss,
            esr,
        },
        protocol::WorkerEvent::TrainingComplete {
            file,
            validation_esr,
            model_path,
            ..
        } => WorkerMessage::FileCompleted {
            file,
            validation_esr,
            model_path,
        },
        protocol::WorkerEvent::TrainingFailed {
            file,
            error_kind,
            error,
            ..
        } => WorkerMessage::FileFailed {
            file,
            kind: error_kind,
            error,
        },
        protocol::WorkerEvent::TrainingStart {
            file, total_epochs, ..
        } => WorkerMessage::TrainingStart { file, total_epochs },
        protocol::WorkerEvent::AllComplete { .. } => WorkerMessage::RunCompleted,
        protocol::WorkerEvent::Error {
            error_kind,
            message,
            ..
        } => WorkerMessage::Error {
            kind: error_kind.into(),
            message,
        },
        protocol::WorkerEvent::Log { message, .. } => WorkerMessage::Log(message),
    }
}

fn non_empty(s: &str) -> Option<String> {
    if s.trim().is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn create_worker_script() -> Result<tempfile::TempPath, (PathBuf, std::io::Error)> {
    let file = tempfile::Builder::new()
        .prefix("nam_worker_")
        .suffix(".py")
        .tempfile()
        .map_err(|error| (std::env::temp_dir(), error))?;
    let path = file.into_temp_path();
    std::fs::write(&path, include_str!("../../python/nam_worker.py"))
        .map_err(|error| (path.to_path_buf(), error))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::{
        create_worker_script, event_to_message, parse_worker_line, protocol, send_worker_message,
        wait_for_child, worker_message_channel, ProtocolDispatcher, ProtocolInput,
        ProtocolSequenceError, ProtocolSequenceValidator, WorkerFailureKind, WorkerHandle,
        WorkerMessage, WorkerStartupError, MAX_PROTOCOL_REORDER_WINDOW,
    };
    #[cfg(unix)]
    use super::{encode_worker_path, WorkerRequestError};
    use proptest::prelude::*;
    use serde_json::json;
    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn worker_script_paths_are_unique() {
        let a = create_worker_script().unwrap();
        let b = create_worker_script().unwrap();
        assert_ne!(&*a, &*b);
        assert_eq!(a.extension().and_then(|ext| ext.to_str()), Some("py"));
        assert_eq!(b.extension().and_then(|ext| ext.to_str()), Some("py"));
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_worker_paths_return_a_structured_error() {
        use std::os::unix::ffi::OsStringExt;

        let path = std::path::PathBuf::from(std::ffi::OsString::from_vec(vec![0xff]));
        let error = encode_worker_path(&path, "input").unwrap_err();

        assert!(matches!(
            error,
            WorkerRequestError::NonUnicodePath {
                label: "input",
                path: error_path,
            } if error_path == path
        ));
    }

    #[test]
    fn startup_errors_distinguish_serialization_and_launch_failures() {
        let serialization = WorkerStartupError::SerializeRequest {
            source: serde_json::from_str::<serde_json::Value>("{").unwrap_err(),
        };
        assert_eq!(serialization.kind(), WorkerFailureKind::Serialization);
        assert!(serialization.to_string().contains("serialize"));

        let launch = WorkerStartupError::Launch {
            python: "missing-python".into(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        };
        assert_eq!(launch.kind(), WorkerFailureKind::Launch);
        assert!(launch.to_string().contains("start Python worker"));
    }

    #[test]
    fn maps_worker_events_to_messages() {
        assert_eq!(
            event_to_message(protocol::WorkerEvent::TrainingStart {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "run".into(),
                file_index: Some(0),
                sequence: 1,
                file: "output.wav".into(),
                total_epochs: 3,
            }),
            WorkerMessage::TrainingStart {
                file: "output.wav".into(),
                total_epochs: 3,
            }
        );
        assert_eq!(
            event_to_message(protocol::WorkerEvent::EpochEnd {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "run".into(),
                file_index: Some(0),
                sequence: 2,
                epoch: 2,
                train_loss: 0.1,
                val_loss: 0.2,
                esr: 0.3,
            }),
            WorkerMessage::EpochEnd {
                epoch: 2,
                train_loss: 0.1,
                val_loss: 0.2,
                esr: 0.3,
            }
        );
        assert_eq!(
            event_to_message(protocol::WorkerEvent::TrainingComplete {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "run".into(),
                file_index: Some(0),
                sequence: 3,
                file: "output.wav".into(),
                validation_esr: 0.01,
                model_path: "model.nam".into(),
            }),
            WorkerMessage::FileCompleted {
                file: "output.wav".into(),
                validation_esr: 0.01,
                model_path: "model.nam".into(),
            }
        );
        assert_eq!(
            event_to_message(protocol::WorkerEvent::TrainingFailed {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "run".into(),
                file_index: Some(0),
                sequence: 3,
                file: "output.wav".into(),
                error_kind: protocol::WorkerErrorKind::Training,
                error: "failed".into(),
            }),
            WorkerMessage::FileFailed {
                file: "output.wav".into(),
                kind: protocol::WorkerErrorKind::Training,
                error: "failed".into(),
            }
        );
        assert_eq!(
            event_to_message(protocol::WorkerEvent::AllComplete {
                protocol_version: protocol::PROTOCOL_VERSION,
                run_id: "run".into(),
                file_index: None,
                sequence: 4,
            }),
            WorkerMessage::RunCompleted
        );
    }

    #[test]
    fn protocol_mismatch_becomes_a_structured_error() {
        let message = event_to_message(protocol::WorkerEvent::Log {
            protocol_version: protocol::PROTOCOL_VERSION + 1,
            run_id: "run".into(),
            file_index: None,
            sequence: 1,
            message: "incompatible".into(),
        });

        assert!(matches!(
            message,
            WorkerMessage::Error {
                kind: WorkerFailureKind::Protocol,
                message,
            } if message.contains("Unsupported worker protocol version")
        ));
    }

    #[test]
    fn protocol_sequence_rejects_invalid_state_transitions() {
        let epoch = protocol_event(json!({
            "type": "epoch_end",
            "protocol_version": protocol::PROTOCOL_VERSION,
            "run_id": "run",
            "file_index": 0,
            "sequence": 1,
            "epoch": 1,
            "train_loss": 0.1,
            "val_loss": 0.2,
            "esr": 0.3
        }));
        assert_eq!(
            ProtocolSequenceValidator::new("run".into())
                .validate(&epoch)
                .unwrap_err(),
            ProtocolSequenceError::EventBeforeTrainingStart { event: "epoch_end" }
        );

        let mut validator = ProtocolSequenceValidator::new("run".into());
        validator.validate(&start_event(0, 1)).unwrap();
        assert_eq!(
            validator.validate(&complete_event(1, 2)).unwrap_err(),
            ProtocolSequenceError::FileIndexMismatch {
                event: "file completion",
                actual: 1,
                active: 0,
            }
        );

        let mut validator = ProtocolSequenceValidator::new("run".into());
        validator.validate(&start_event(0, 1)).unwrap();
        validator.validate(&complete_event(0, 2)).unwrap();
        assert_eq!(
            validator.validate(&complete_event(0, 3)).unwrap_err(),
            ProtocolSequenceError::EventBeforeTrainingStart {
                event: "file completion",
            }
        );

        let mut validator = ProtocolSequenceValidator::new("run".into());
        validator.validate(&start_event(0, 1)).unwrap();
        validator.validate(&complete_event(0, 2)).unwrap();
        validator
            .validate(&protocol_event(json!({
                "type": "all_complete",
                "protocol_version": protocol::PROTOCOL_VERSION,
                "run_id": "run",
                "file_index": null,
                "sequence": 3
            })))
            .unwrap();
        assert_eq!(
            validator
                .validate(&protocol_event(json!({
                    "type": "log",
                    "protocol_version": protocol::PROTOCOL_VERSION,
                    "run_id": "run",
                    "file_index": null,
                    "sequence": 4,
                    "message": "late"
                })))
                .unwrap_err(),
            ProtocolSequenceError::EventAfterTerminal
        );
    }

    #[test]
    fn protocol_dispatcher_orders_reversed_stdout_and_stderr_events() {
        let mut dispatcher = ProtocolDispatcher::new("run".into());
        let deferred = dispatcher.accept(ProtocolInput::StderrEvent(complete_event(0, 2)));
        assert!(deferred.is_empty());

        let messages = dispatcher.accept(ProtocolInput::StdoutEvent(start_event(0, 1)));
        assert_eq!(messages.len(), 2);
        assert!(matches!(
            messages.first(),
            Some(WorkerMessage::TrainingStart { .. })
        ));
        assert!(matches!(
            messages.get(1),
            Some(WorkerMessage::FileCompleted { .. })
        ));
    }

    proptest! {
        #[test]
        fn protocol_dispatcher_orders_shuffled_events_from_both_streams(
            priorities in proptest::collection::vec(any::<u8>(), 1..32),
        ) {
            let mut delivery: Vec<_> = priorities
                .into_iter()
                .enumerate()
                .map(|(index, priority)| (priority, index as u64 + 1))
                .collect();
            delivery.sort_by_key(|(priority, _)| *priority);

            let expected_count = delivery.len();
            let mut dispatcher = ProtocolDispatcher::new("run".into());
            let mut messages = Vec::new();
            for (priority, sequence) in delivery {
                let event = log_event(sequence);
                let input = if priority % 2 == 0 {
                    ProtocolInput::StdoutEvent(event)
                } else {
                    ProtocolInput::StderrEvent(event)
                };
                messages.extend(dispatcher.accept(input));
            }
            messages.extend(dispatcher.finish());

            prop_assert_eq!(messages.len(), expected_count);
            for (index, message) in messages.into_iter().enumerate() {
                prop_assert_eq!(message, WorkerMessage::Log(format!("event {}", index + 1)));
            }
        }

        #[test]
        fn protocol_dispatcher_reports_any_missing_sequence(
            count in 2u64..32,
            missing_index in 0usize..31,
        ) {
            let missing = missing_index as u64 % (count - 1) + 1;
            let mut dispatcher = ProtocolDispatcher::new("run".into());
            for sequence in 1..=count {
                if sequence != missing {
                    dispatcher.accept(ProtocolInput::StdoutEvent(log_event(sequence)));
                }
            }
            let finished = dispatcher.finish();
            let is_sequence_error = matches!(
                finished.as_slice(),
                [WorkerMessage::Error {
                    kind: WorkerFailureKind::ProtocolSequence,
                    ..
                }]
            );
            prop_assert!(is_sequence_error);
        }

        #[test]
        fn protocol_dispatcher_rejects_duplicates(sequence in 1u64..32) {
            let mut dispatcher = ProtocolDispatcher::new("run".into());
            dispatcher.accept(ProtocolInput::StdoutEvent(log_event(sequence)));
            let duplicate = dispatcher.accept(ProtocolInput::StderrEvent(log_event(sequence)));
            let is_sequence_error = matches!(
                duplicate.as_slice(),
                [WorkerMessage::Error {
                    kind: WorkerFailureKind::ProtocolSequence,
                    ..
                }]
            );
            prop_assert!(is_sequence_error);
        }

        #[test]
        fn protocol_dispatcher_rejects_large_sequence_jumps(extra in 1u64..10_000) {
            let mut dispatcher = ProtocolDispatcher::new("run".into());
            let sequence = 1u64
                .saturating_add(MAX_PROTOCOL_REORDER_WINDOW)
                .saturating_add(extra);
            let messages =
                dispatcher.accept(ProtocolInput::StdoutEvent(log_event(sequence)));
            let is_sequence_error = matches!(
                messages.as_slice(),
                [WorkerMessage::Error {
                    kind: WorkerFailureKind::ProtocolSequence,
                    ..
                }]
            );
            prop_assert!(is_sequence_error);
            prop_assert_eq!(dispatcher.pending.len(), 0);
        }
    }

    fn start_event(file_index: usize, sequence: u64) -> protocol::WorkerEvent {
        protocol_event(json!({
            "type": "training_start",
            "protocol_version": protocol::PROTOCOL_VERSION,
            "run_id": "run",
            "file_index": file_index,
            "sequence": sequence,
            "file": "output.wav",
            "total_epochs": 1
        }))
    }

    fn complete_event(file_index: usize, sequence: u64) -> protocol::WorkerEvent {
        protocol_event(json!({
            "type": "training_complete",
            "protocol_version": protocol::PROTOCOL_VERSION,
            "run_id": "run",
            "file_index": file_index,
            "sequence": sequence,
            "file": "output.wav",
            "validation_esr": 0.1,
            "model_path": "model.nam"
        }))
    }

    fn log_event(sequence: u64) -> protocol::WorkerEvent {
        protocol_event(json!({
            "type": "log",
            "protocol_version": protocol::PROTOCOL_VERSION,
            "run_id": "run",
            "file_index": null,
            "sequence": sequence,
            "message": format!("event {sequence}")
        }))
    }

    fn protocol_event(value: serde_json::Value) -> protocol::WorkerEvent {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn cancellation_is_remembered_before_a_child_is_available() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let handle = WorkerHandle {
            cancelled: Arc::clone(&cancelled),
            task: None,
        };

        handle.cancel();

        assert!(cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn dropping_handle_queues_cancellation() {
        let cancelled = Arc::new(AtomicBool::new(false));
        drop(WorkerHandle {
            cancelled: Arc::clone(&cancelled),
            task: None,
        });

        assert!(cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn queued_cancellation_terminates_a_newly_started_child() {
        let Some(python) = python_for_tests() else {
            return;
        };
        let mut child = Command::new(python)
            .args(["-c", "import time; time.sleep(30)"])
            .spawn()
            .unwrap();
        let cancelled = AtomicBool::new(true);
        let started = std::time::Instant::now();

        let _ = wait_for_child(&mut child, &cancelled, None);

        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn cancellation_allows_a_cooperative_child_to_exit() {
        let Some(python) = python_for_tests() else {
            return;
        };
        let mut child = Command::new(python)
            .args([
                "-c",
                "import json, sys; command = json.loads(sys.stdin.readline()); sys.exit(0 if command.get('command') == 'cancel' else 2)",
            ])
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let child_stdin = child.stdin.take();
        let cancelled = AtomicBool::new(true);

        let exit_code = wait_for_child(&mut child, &cancelled, child_stdin);

        assert_eq!(exit_code, Some(0));
    }

    proptest! {
        #[test]
        fn malformed_worker_output_is_preserved_as_log(line in any::<String>()) {
            prop_assume!(serde_json::from_str::<protocol::WorkerEvent>(&line).is_err());
            prop_assert_eq!(parse_worker_line(line.clone()), WorkerMessage::Log(line));
        }

        #[test]
        fn arbitrary_log_events_decode_without_losing_the_message(
            message in any::<String>(),
            protocol_version in any::<u32>(),
        ) {
            let line = serde_json::json!({
                "type": "log",
                "protocol_version": protocol_version,
                "run_id": "run",
                "file_index": null,
                "sequence": 1,
                "message": message,
            })
            .to_string();
            let decoded = parse_worker_line(line);
            if protocol_version == protocol::PROTOCOL_VERSION {
                prop_assert_eq!(decoded, WorkerMessage::Log(message));
            } else {
                let is_protocol_error = matches!(
                    decoded,
                    WorkerMessage::Error {
                        kind: WorkerFailureKind::Protocol,
                        ..
                    }
                );
                prop_assert!(is_protocol_error);
            }
        }
    }

    #[test]
    fn saturated_channel_drops_only_lossy_progress_messages() {
        let (tx, rx) = worker_message_channel(1);
        assert!(send_worker_message(&tx, WorkerMessage::Log("first".into())));
        assert!(send_worker_message(
            &tx,
            WorkerMessage::Log("dropped".into())
        ));

        assert_eq!(rx.try_recv().unwrap(), WorkerMessage::Log("first".into()));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn epoch_backpressure_retains_the_latest_update() {
        let (tx, rx) = worker_message_channel(1);
        for epoch in 1..=3 {
            assert!(send_worker_message(
                &tx,
                WorkerMessage::EpochEnd {
                    epoch,
                    train_loss: f64::from(epoch),
                    val_loss: 0.0,
                    esr: 0.0,
                }
            ));
        }

        assert!(matches!(
            rx.try_recv(),
            Ok(WorkerMessage::EpochEnd { epoch: 3, .. })
        ));
    }

    #[test]
    fn critical_messages_bypass_saturated_logs() {
        let (tx, rx) = worker_message_channel(1);
        assert!(send_worker_message(&tx, WorkerMessage::Log("first".into())));
        assert!(send_worker_message(
            &tx,
            WorkerMessage::Log("dropped".into())
        ));
        assert!(send_worker_message(&tx, WorkerMessage::RunCompleted));

        assert_eq!(rx.try_recv(), Ok(WorkerMessage::RunCompleted));
        assert_eq!(rx.try_recv(), Ok(WorkerMessage::Log("first".into())));
    }

    #[test]
    fn critical_queue_is_bounded_and_cancellation_aware() {
        let (tx, rx) = worker_message_channel(1);
        for index in 0..super::channel::WORKER_CRITICAL_CAPACITY {
            assert!(send_worker_message(
                &tx,
                WorkerMessage::TrainingStart {
                    file: format!("{index}.wav"),
                    total_epochs: 1,
                }
            ));
        }

        tx.cancelled.store(true, Ordering::Release);
        assert!(!send_worker_message(
            &tx,
            WorkerMessage::TrainingStart {
                file: "overflow.wav".into(),
                total_epochs: 1,
            }
        ));

        assert_eq!(
            std::iter::from_fn(|| rx.try_recv().ok()).count(),
            super::channel::WORKER_CRITICAL_CAPACITY
        );
    }

    #[test]
    fn terminal_message_survives_cancellation_backpressure() {
        let (tx, rx) = worker_message_channel(1);
        for index in 0..super::channel::WORKER_CRITICAL_CAPACITY {
            assert!(send_worker_message(
                &tx,
                WorkerMessage::TrainingStart {
                    file: format!("{index}.wav"),
                    total_epochs: 1,
                }
            ));
        }
        tx.cancelled.store(true, Ordering::Release);
        let sender =
            std::thread::spawn(move || send_worker_message(&tx, WorkerMessage::RunCompleted));

        assert!(matches!(
            rx.try_recv(),
            Ok(WorkerMessage::TrainingStart { .. })
        ));
        assert!(sender.join().unwrap());
        assert!(std::iter::from_fn(|| rx.try_recv().ok())
            .any(|message| message == WorkerMessage::RunCompleted));
    }

    #[test]
    fn python_worker_rejects_an_incompatible_protocol() {
        let Some(python) = python_for_tests() else {
            return;
        };
        let (_temp, root) = unique_worker_test_dir("protocol");
        std::fs::create_dir_all(&root).unwrap();
        let worker = root.join("nam_worker.py");
        std::fs::write(&worker, include_str!("../../python/nam_worker.py")).unwrap();
        let request = json!({
            "protocol_version": protocol::PROTOCOL_VERSION + 1,
            "input_path": "input.wav",
            "output_paths": ["output.wav"],
            "destination": root.join("models"),
        });

        let output = run_worker(&python, &worker, &root, &request);
        assert!(!output.status.success());
        let events = parse_worker_events(&output.stdout);
        assert!(events.iter().any(|event| {
            event["type"] == "error"
                && event["error_kind"] == "protocol"
                && event["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("cannot process request protocol"))
        }));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn python_worker_routes_default_core_train_request() {
        let Some(python) = python_for_tests() else {
            return;
        };
        let (_temp, root) = unique_worker_test_dir("core");
        write_fake_python_modules(&root);
        let worker = root.join("nam_worker.py");
        std::fs::write(&worker, include_str!("../../python/nam_worker.py")).unwrap();
        let destination = root.join("models");
        let request = json!({
            "protocol_version": protocol::PROTOCOL_VERSION,
            "input_path": "input.wav",
            "output_paths": ["output-a.wav", "output-b.wav"],
            "destination": destination,
            "output_model_basename": null,
            "batch_name_template": null,
            "architecture": "packed",
            "packed": true,
            "epochs": 2,
            "batch_size": 4,
            "lr": 0.003,
            "lr_decay": 0.9,
            "latency": 12,
            "threshold_esr": null,
            "save_plot": false,
            "fit_mrstft": true,
            "ignore_checks": true,
            "num_output_samples_per_datum": 256,
            "use_full_config_trainer": false,
            "device": "cpu",
            "metadata": {"name": "Amp", "input_level_dbu": 1.5}
        });

        let output = run_worker(&python, &worker, &root, &request);
        assert!(
            output.status.success(),
            "worker failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let events = parse_worker_events(&output.stdout);
        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "training_complete")
                .count(),
            2,
            "events: {events:#?}"
        );
        assert_eq!(
            events.last().and_then(|event| event["type"].as_str()),
            Some("all_complete")
        );

        let kwargs_path = destination.join("output-a").join("core_kwargs.json");
        let kwargs: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(kwargs_path).unwrap()).unwrap();
        assert_eq!(kwargs["architecture"], "packed");
        assert_eq!(kwargs["ny"], 256);
        assert_eq!(kwargs["ignore_checks"], true);
        assert_eq!(kwargs["user_metadata"]["name"], "Amp");
        assert_eq!(kwargs["user_metadata"]["input_level_dbu"], 1.5);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn python_worker_routes_packed_full_config_request() {
        let Some(python) = python_for_tests() else {
            return;
        };
        let (_temp, root) = unique_worker_test_dir("full");
        write_fake_python_modules(&root);
        let worker = root.join("nam_worker.py");
        std::fs::write(&worker, include_str!("../../python/nam_worker.py")).unwrap();
        let destination = root.join("models");
        let request = json!({
            "protocol_version": protocol::PROTOCOL_VERSION,
            "input_path": "input.wav",
            "output_paths": ["output.wav"],
            "destination": destination,
            "output_model_basename": "custom-full",
            "batch_name_template": null,
            "architecture": "packed",
            "packed": true,
            "epochs": 3,
            "batch_size": 5,
            "lr": 0.002,
            "lr_decay": 0.9,
            "latency": null,
            "threshold_esr": null,
            "save_plot": false,
            "fit_mrstft": true,
            "ignore_checks": true,
            "num_output_samples_per_datum": 512,
            "use_full_config_trainer": true,
            "device": "cpu",
            "metadata": {}
        });

        let output = run_worker(&python, &worker, &root, &request);
        assert!(
            output.status.success(),
            "worker failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let events = parse_worker_events(&output.stdout);
        assert!(events
            .iter()
            .any(|event| event["message"] == "Using upstream packed full-config trainer path"));
        assert!(events
            .iter()
            .any(|event| event["type"] == "training_complete"));

        let config_path = destination.join("custom-full").join("full_configs.json");
        let configs: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(config_path).unwrap()).unwrap();
        assert_eq!(configs["data"]["train"]["ny"], 512);
        assert_eq!(configs["model"]["optimizer"]["lr"], 0.002);
        assert_eq!(configs["learning"]["trainer"]["max_epochs"], 3);
        assert_eq!(configs["learning"]["train_dataloader"]["batch_size"], 5);

        std::fs::remove_dir_all(root).unwrap();
    }

    fn python_for_tests() -> Option<String> {
        for candidate in ["python3", "python"] {
            if Command::new(candidate)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
            {
                return Some(candidate.into());
            }
        }
        None
    }

    fn unique_worker_test_dir(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::Builder::new()
            .prefix(&format!("nam-trainer-worker-{name}-"))
            .tempdir()
            .unwrap();
        let path = temp.path().to_path_buf();
        (temp, path)
    }

    fn run_worker(
        python: &str,
        worker: &std::path::Path,
        pythonpath: &std::path::Path,
        request: &serde_json::Value,
    ) -> std::process::Output {
        let mut child = Command::new(python)
            .arg(worker)
            .env("PYTHONPATH", pythonpath)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        {
            let stdin = child.stdin.as_mut().unwrap();
            writeln!(stdin, "{request}").unwrap();
        }
        child.wait_with_output().unwrap()
    }

    fn parse_worker_events(stdout: &[u8]) -> Vec<serde_json::Value> {
        String::from_utf8_lossy(stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn write_fake_python_modules(root: &std::path::Path) {
        write_file(
            &root.join("pytorch_lightning.py"),
            r#"
class Callback:
    pass

class Trainer:
    def __init__(self, *args, **kwargs):
        self.callbacks = []
"#,
        );
        write_file(&root.join("nam/__init__.py"), "");
        write_file(&root.join("nam/train/__init__.py"), "");
        write_file(&root.join("nam/models/__init__.py"), "");
        write_file(
            &root.join("nam/models/metadata.py"),
            r#"
class UserMetadata(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
"#,
        );
        write_file(
            &root.join("nam/train/core.py"),
            r#"
import json
import os

class Version:
    major = 3

class DataChecks:
    passed = True

class Latency:
    pass

def train(
    input_path=None,
    output_path=None,
    train_path=None,
    epochs=100,
    latency=None,
    architecture="standard",
    batch_size=16,
    ny=8192,
    lr=0.004,
    lr_decay=0.007,
    seed=0,
    save_plot=True,
    silent=True,
    modelname=None,
    ignore_checks=False,
    fit_mrstft=True,
    threshold_esr=None,
    user_metadata=None,
):
    kwargs = {
        "input_path": input_path,
        "output_path": output_path,
        "train_path": train_path,
        "epochs": epochs,
        "latency": latency,
        "architecture": architecture,
        "batch_size": batch_size,
        "ny": ny,
        "lr": lr,
        "lr_decay": lr_decay,
        "seed": seed,
        "save_plot": save_plot,
        "silent": silent,
        "modelname": modelname,
        "ignore_checks": ignore_checks,
        "fit_mrstft": fit_mrstft,
        "threshold_esr": threshold_esr,
        "user_metadata": user_metadata,
    }
    train_path = kwargs["train_path"]
    os.makedirs(os.path.join(train_path, "nested"), exist_ok=True)
    serializable = dict(kwargs)
    serializable["user_metadata"] = dict(kwargs.get("user_metadata") or {})
    with open(os.path.join(train_path, "core_kwargs.json"), "w") as fp:
        json.dump(serializable, fp)
    with open(os.path.join(train_path, "nested", "model.nam"), "w") as fp:
        fp.write("{}")
    return object()

def _detect_input_version(input_path):
    return Version(), True

def _analyze_latency(user_latency, input_version, input_path, output_path, silent=False):
    return Latency()

def _get_final_latency(latency_analysis):
    return 7

def _check_data(input_path, output_path, input_version, final_latency, silent):
    return DataChecks()

def _get_configs(input_version, input_path, output_path, final_latency, epochs, ny, batch_size):
    return (
        {"train": {"ny": ny}, "common": {"delay": final_latency}},
        {"net": {"name": "PackedWaveNet"}, "optimizer": {"lr": 0.004}},
        {"trainer": {"max_epochs": epochs}, "train_dataloader": {"batch_size": batch_size}},
    )
"#,
        );
        write_file(
            &root.join("nam/train/full.py"),
            r#"
import json
import os

def main(data_config, model_config, learning_config, outdir, no_show=False, make_plots=True):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "full_configs.json"), "w") as fp:
        json.dump({"data": data_config, "model": model_config, "learning": learning_config}, fp)
    with open(os.path.join(outdir, "model.nam"), "w") as fp:
        fp.write("{}")
"#,
        );
    }

    fn write_file(path: &std::path::Path, contents: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }
}
