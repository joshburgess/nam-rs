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
    failed: bool,
    stderr_log_lines: usize,
    stderr_suppression_sent: bool,
}

impl ProtocolDispatcher {
    fn new(run_id: String) -> Self {
        Self {
            validator: ProtocolSequenceValidator::new(run_id),
            pending: BTreeMap::new(),
            next_sequence: 1,
            failed: false,
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
        if self.failed {
            return Vec::new();
        }
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
        if self.failed {
            return Vec::new();
        }
        let sequence = event.sequence();
        if sequence < self.next_sequence || self.pending.contains_key(&sequence) {
            let previous = self.next_sequence.saturating_sub(1);
            return self.fail(ProtocolSequenceError::NonIncreasingSequence {
                actual: sequence,
                previous,
            });
        }
        let maximum = self
            .next_sequence
            .saturating_add(MAX_PROTOCOL_REORDER_WINDOW);
        if sequence > maximum {
            return self.fail(ProtocolSequenceError::SequenceJump {
                actual: sequence,
                expected: self.next_sequence,
                maximum,
            });
        }
        self.pending.insert(sequence, event);

        let mut messages = Vec::new();
        while let Some(event) = self.pending.remove(&self.next_sequence) {
            match self.validator.validate(&event) {
                Ok(()) => {
                    messages.push(event_to_message(event));
                    self.next_sequence = self.next_sequence.saturating_add(1);
                }
                Err(error) => {
                    self.failed = true;
                    self.pending.clear();
                    messages.push(Self::sequence_error(error));
                    break;
                }
            }
        }
        messages
    }

    fn fail(&mut self, error: ProtocolSequenceError) -> Vec<WorkerMessage> {
        self.failed = true;
        self.pending.clear();
        vec![Self::sequence_error(error)]
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
        let mut worker_process = SystemWorkerProcess {
            child,
            stdin: child_stdin,
        };
        let exit_code = wait_for_child(&mut worker_process, &worker_cancelled, &SystemClock);

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

trait WorkerProcess {
    fn request_cancel(&mut self) -> std::io::Result<()>;
    fn try_wait(&mut self) -> std::io::Result<Option<Option<i32>>>;
    fn kill(&mut self) -> std::io::Result<()>;
    fn wait(&mut self) -> std::io::Result<Option<i32>>;
}

struct SystemWorkerProcess {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
}

impl WorkerProcess for SystemWorkerProcess {
    fn request_cancel(&mut self) -> std::io::Result<()> {
        let Some(mut stdin) = self.stdin.take() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "worker stdin is unavailable",
            ));
        };
        writeln!(stdin, r#"{{"command":"cancel"}}"#)?;
        stdin.flush()
    }

    fn try_wait(&mut self) -> std::io::Result<Option<Option<i32>>> {
        self.child
            .try_wait()
            .map(|status| status.map(|status| status.code()))
    }

    fn kill(&mut self) -> std::io::Result<()> {
        self.child.kill()
    }

    fn wait(&mut self) -> std::io::Result<Option<i32>> {
        self.child.wait().map(|status| status.code())
    }
}

trait Clock {
    fn now(&self) -> Instant;
    fn sleep(&self, duration: Duration);
}

struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep(&self, duration: Duration) {
        thread::sleep(duration);
    }
}

fn wait_for_child(
    child: &mut dyn WorkerProcess,
    cancelled: &AtomicBool,
    clock: &dyn Clock,
) -> Option<i32> {
    let mut force_cancel_at = None;
    loop {
        if cancelled.load(Ordering::Acquire) && force_cancel_at.is_none() {
            if child.request_cancel().is_ok() {
                force_cancel_at = Some(clock.now() + COOPERATIVE_CANCEL_GRACE);
            } else {
                force_cancel_at = Some(clock.now());
            }
        }
        match child.try_wait() {
            Ok(Some(exit_code)) => return exit_code,
            Ok(None) => {
                if force_cancel_at.is_some_and(|deadline| clock.now() >= deadline) {
                    let _ = child.kill();
                    return child.wait().ok().flatten();
                }
                clock.sleep(Duration::from_millis(20));
            }
            Err(_) => return child.wait().ok().flatten(),
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
mod tests;
