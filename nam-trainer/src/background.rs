use std::io::{self, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{
    Receiver, SendTimeoutError, Sender, TryRecvError as CrossbeamTryRecvError,
};

const CRITICAL_CAPACITY: usize = 8;
const SEND_RETRY: Duration = Duration::from_millis(10);
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);
const COMMAND_OUTPUT_LIMIT: usize = 1024 * 1024;

pub(crate) struct CommandOutput {
    pub(crate) success: bool,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) stdout_truncated_bytes: usize,
    pub(crate) stderr_truncated_bytes: usize,
}

pub(crate) trait ProcessRunner: Send + Sync {
    fn output(
        &self,
        command: &mut Command,
        cancel: &CancellationToken,
    ) -> io::Result<Option<CommandOutput>>;

    fn spawn(&self, command: &mut Command) -> io::Result<Child>;
}

pub(crate) struct SystemProcessRunner;

impl ProcessRunner for SystemProcessRunner {
    fn output(
        &self,
        command: &mut Command,
        cancel: &CancellationToken,
    ) -> io::Result<Option<CommandOutput>> {
        command_output_impl(command, cancel)
    }

    fn spawn(&self, command: &mut Command) -> io::Result<Child> {
        command.spawn()
    }
}

struct CapturedOutput {
    bytes: Vec<u8>,
    truncated_bytes: usize,
}

#[derive(Clone)]
pub(crate) struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[cfg(test)]
pub(crate) fn command_output(
    command: &mut Command,
    cancel: &CancellationToken,
) -> io::Result<Option<CommandOutput>> {
    SystemProcessRunner.output(command, cancel)
}

fn command_output_impl(
    command: &mut Command,
    cancel: &CancellationToken,
) -> io::Result<Option<CommandOutput>> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = spawn_output_reader(child.stdout.take());
    let stderr = spawn_output_reader(child.stderr.take());

    let status = loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            join_output_reader(stdout)?;
            join_output_reader(stderr)?;
            return Ok(None);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(CHILD_POLL_INTERVAL),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                join_output_reader(stdout)?;
                join_output_reader(stderr)?;
                return Err(error);
            }
        }
    };

    let stdout = join_output_reader(stdout)?;
    let stderr = join_output_reader(stderr)?;
    Ok(Some(CommandOutput {
        success: status.success(),
        stdout: stdout.bytes,
        stderr: stderr.bytes,
        stdout_truncated_bytes: stdout.truncated_bytes,
        stderr_truncated_bytes: stderr.truncated_bytes,
    }))
}

fn spawn_output_reader<R>(stream: Option<R>) -> Option<JoinHandle<io::Result<CapturedOutput>>>
where
    R: Read + Send + 'static,
{
    stream.map(|stream| thread::spawn(move || read_capped(stream, COMMAND_OUTPUT_LIMIT)))
}

fn read_capped(mut stream: impl Read, limit: usize) -> io::Result<CapturedOutput> {
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    let mut truncated_bytes = 0usize;
    let mut chunk = [0u8; 16 * 1024];
    loop {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let retained = count.min(limit.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&chunk[..retained]);
        truncated_bytes = truncated_bytes.saturating_add(count - retained);
    }
    Ok(CapturedOutput {
        bytes,
        truncated_bytes,
    })
}

fn join_output_reader(
    reader: Option<JoinHandle<io::Result<CapturedOutput>>>,
) -> io::Result<CapturedOutput> {
    let Some(reader) = reader else {
        return Ok(CapturedOutput {
            bytes: Vec::new(),
            truncated_bytes: 0,
        });
    };
    reader
        .join()
        .map_err(|_| io::Error::other("subprocess output reader panicked"))?
}

pub(crate) struct BackgroundSender<T> {
    progress: Sender<T>,
    critical: Sender<T>,
    receiver_alive: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    dropped_progress: Arc<AtomicUsize>,
}

impl<T> Clone for BackgroundSender<T> {
    fn clone(&self) -> Self {
        Self {
            progress: self.progress.clone(),
            critical: self.critical.clone(),
            receiver_alive: Arc::clone(&self.receiver_alive),
            cancelled: Arc::clone(&self.cancelled),
            dropped_progress: Arc::clone(&self.dropped_progress),
        }
    }
}

impl<T> BackgroundSender<T> {
    pub(crate) fn send_progress(&self, message: T) -> bool {
        if !self.receiver_alive.load(Ordering::Acquire) || self.cancelled() {
            return false;
        }
        match self.progress.try_send(message) {
            Ok(()) => true,
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                self.dropped_progress.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => false,
        }
    }

    pub(crate) fn send_critical(&self, mut message: T) -> bool {
        loop {
            match self.critical.send_timeout(message, SEND_RETRY) {
                Ok(()) => return true,
                Err(SendTimeoutError::Disconnected(_)) => return false,
                Err(SendTimeoutError::Timeout(returned)) => {
                    message = returned;
                    if !self.receiver_alive.load(Ordering::Acquire) || self.cancelled() {
                        return false;
                    }
                }
            }
        }
    }

    pub(crate) fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

pub(crate) struct BackgroundOperation<T> {
    progress: Receiver<T>,
    critical: Receiver<T>,
    receiver_alive: Arc<AtomicBool>,
    dropped_progress: Arc<AtomicUsize>,
    _task: ManagedTask,
}

impl<T> BackgroundOperation<T> {
    pub(crate) fn try_recv(&self) -> Result<T, mpsc::TryRecvError> {
        let critical = self.critical.try_recv();
        if let Ok(message) = critical {
            return Ok(message);
        }
        let progress = self.progress.try_recv();
        if let Ok(message) = progress {
            return Ok(message);
        }
        if matches!(progress, Err(CrossbeamTryRecvError::Disconnected))
            && matches!(critical, Err(CrossbeamTryRecvError::Disconnected))
        {
            Err(mpsc::TryRecvError::Disconnected)
        } else {
            Err(mpsc::TryRecvError::Empty)
        }
    }

    pub(crate) fn take_dropped_progress(&self) -> usize {
        self.dropped_progress.swap(0, Ordering::Relaxed)
    }
}

impl<T> Drop for BackgroundOperation<T> {
    fn drop(&mut self) {
        self.receiver_alive.store(false, Ordering::Release);
    }
}

pub(crate) struct ManagedTask {
    cancelled: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl ManagedTask {
    pub(crate) fn from_join(cancelled: Arc<AtomicBool>, join: JoinHandle<()>) -> Self {
        Self {
            cancelled,
            join: Some(join),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl Drop for ManagedTask {
    fn drop(&mut self) {
        self.cancel();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub(crate) fn spawn<T, F>(progress_capacity: usize, task: F) -> BackgroundOperation<T>
where
    T: Send + 'static,
    F: FnOnce(CancellationToken, BackgroundSender<T>) + Send + 'static,
{
    let (progress_tx, progress_rx) = crossbeam_channel::bounded(progress_capacity);
    let (critical_tx, critical_rx) = crossbeam_channel::bounded(CRITICAL_CAPACITY);
    let receiver_alive = Arc::new(AtomicBool::new(true));
    let cancelled = Arc::new(AtomicBool::new(false));
    let dropped_progress = Arc::new(AtomicUsize::new(0));
    let token = CancellationToken {
        cancelled: Arc::clone(&cancelled),
    };
    let sender = BackgroundSender {
        progress: progress_tx,
        critical: critical_tx,
        receiver_alive: Arc::clone(&receiver_alive),
        cancelled: Arc::clone(&cancelled),
        dropped_progress: Arc::clone(&dropped_progress),
    };
    let join = thread::spawn(move || task(token, sender));
    BackgroundOperation {
        progress: progress_rx,
        critical: critical_rx,
        receiver_alive,
        dropped_progress,
        _task: ManagedTask::from_join(cancelled, join),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::{spawn, BackgroundOperation};

    fn receive<T>(operation: &BackgroundOperation<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            match operation.try_recv() {
                Ok(message) => return message,
                Err(std::sync::mpsc::TryRecvError::Empty) if Instant::now() < deadline => {
                    std::thread::yield_now();
                }
                Err(error) => panic!("background message was not delivered: {error}"),
            }
        }
    }

    #[test]
    fn progress_is_bounded_and_reports_drops() {
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
        let operation = spawn(1, move |_cancel, sender| {
            assert!(sender.send_progress(1));
            assert!(sender.send_progress(2));
            assert!(sender.send_critical(3));
            ready_tx.send(()).unwrap();
        });
        ready_rx.recv().unwrap();

        assert_eq!(receive(&operation), 3);
        assert_eq!(receive(&operation), 1);
        assert_eq!(operation.take_dropped_progress(), 1);
    }

    #[test]
    fn critical_messages_are_received_before_queued_progress() {
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
        let operation = spawn(1, move |_cancel, sender| {
            assert!(sender.send_progress("progress"));
            assert!(sender.send_critical("critical"));
            ready_tx.send(()).unwrap();
        });
        ready_rx.recv().unwrap();

        assert_eq!(receive(&operation), "critical");
        assert_eq!(receive(&operation), "progress");
    }

    #[test]
    fn subprocess_capture_is_bounded_while_counting_truncated_bytes() {
        let input = vec![7u8; super::COMMAND_OUTPUT_LIMIT + 137];
        let captured = super::read_capped(Cursor::new(input), super::COMMAND_OUTPUT_LIMIT).unwrap();

        assert_eq!(captured.bytes.len(), super::COMMAND_OUTPUT_LIMIT);
        assert_eq!(captured.truncated_bytes, 137);
        assert!(captured.bytes.iter().all(|byte| *byte == 7));
    }

    #[test]
    fn dropping_operation_cancels_and_joins_task() {
        let exited = Arc::new(AtomicBool::new(false));
        let exited_writer = Arc::clone(&exited);
        let operation = spawn::<(), _>(1, move |cancel, _sender| {
            while !cancel.is_cancelled() {
                std::thread::yield_now();
            }
            exited_writer.store(true, Ordering::Release);
        });

        drop(operation);
        assert!(exited.load(Ordering::Acquire));
    }

    #[cfg(unix)]
    #[test]
    fn dropping_operation_terminates_owned_child_process() {
        let exited = Arc::new(AtomicBool::new(false));
        let exited_writer = Arc::clone(&exited);
        let operation = spawn::<(), _>(1, move |cancel, _sender| {
            let mut command = std::process::Command::new("sh");
            command.args(["-c", "sleep 30"]);
            let result = super::command_output(&mut command, &cancel);
            assert!(matches!(result, Ok(None)));
            exited_writer.store(true, Ordering::Release);
        });

        drop(operation);
        assert!(exited.load(Ordering::Acquire));
    }
}
