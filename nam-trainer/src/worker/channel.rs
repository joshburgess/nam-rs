use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use crossbeam_channel::{
    Receiver as CrossbeamReceiver, SendTimeoutError, Sender as CrossbeamSender,
    TryRecvError as CrossbeamTryRecvError,
};
use parking_lot::Mutex;

use super::WorkerMessage;

pub(super) const WORKER_CRITICAL_CAPACITY: usize = 64;
const WORKER_SEND_RETRY: Duration = Duration::from_millis(10);

#[derive(Clone)]
pub(crate) struct WorkerMessageSender {
    critical: CrossbeamSender<WorkerMessage>,
    logs: mpsc::SyncSender<WorkerMessage>,
    latest_epoch: Arc<Mutex<Option<WorkerMessage>>>,
    receiver_alive: Arc<AtomicBool>,
    pub(super) cancelled: Arc<AtomicBool>,
}

#[derive(Debug, thiserror::Error)]
#[error("worker message receiver has disconnected")]
pub(crate) struct WorkerMessageSendError;

impl WorkerMessageSender {
    pub(crate) fn send(&self, message: WorkerMessage) -> Result<(), WorkerMessageSendError> {
        send_worker_message(self, message)
            .then_some(())
            .ok_or(WorkerMessageSendError)
    }

    pub(crate) fn cancellation_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }
}

pub struct WorkerMessageReceiver {
    critical: CrossbeamReceiver<WorkerMessage>,
    logs: mpsc::Receiver<WorkerMessage>,
    latest_epoch: Arc<Mutex<Option<WorkerMessage>>>,
    receiver_alive: Arc<AtomicBool>,
}

impl WorkerMessageReceiver {
    pub fn try_recv(&self) -> Result<WorkerMessage, mpsc::TryRecvError> {
        let critical = self.critical.try_recv();
        if let Ok(message) = critical {
            return Ok(message);
        }
        if let Some(message) = self.latest_epoch.lock().take() {
            return Ok(message);
        }
        let logs = self.logs.try_recv();
        if let Ok(message) = logs {
            return Ok(message);
        }
        if matches!(critical, Err(CrossbeamTryRecvError::Disconnected))
            && matches!(logs, Err(mpsc::TryRecvError::Disconnected))
        {
            Err(mpsc::TryRecvError::Disconnected)
        } else {
            Err(mpsc::TryRecvError::Empty)
        }
    }
}

impl Drop for WorkerMessageReceiver {
    fn drop(&mut self) {
        self.receiver_alive.store(false, Ordering::Release);
    }
}

pub(crate) fn worker_message_channel(
    log_capacity: usize,
) -> (WorkerMessageSender, WorkerMessageReceiver) {
    let (critical_tx, critical_rx) = crossbeam_channel::bounded(WORKER_CRITICAL_CAPACITY);
    let (log_tx, log_rx) = mpsc::sync_channel(log_capacity);
    let latest_epoch = Arc::new(Mutex::new(None));
    let receiver_alive = Arc::new(AtomicBool::new(true));
    let cancelled = Arc::new(AtomicBool::new(false));
    (
        WorkerMessageSender {
            critical: critical_tx,
            logs: log_tx,
            latest_epoch: Arc::clone(&latest_epoch),
            receiver_alive: Arc::clone(&receiver_alive),
            cancelled,
        },
        WorkerMessageReceiver {
            critical: critical_rx,
            logs: log_rx,
            latest_epoch,
            receiver_alive,
        },
    )
}

pub(crate) fn send_worker_message(tx: &WorkerMessageSender, message: WorkerMessage) -> bool {
    if !tx.receiver_alive.load(Ordering::Acquire) {
        return false;
    }
    match message {
        message @ WorkerMessage::EpochEnd { .. } => {
            tx.latest_epoch.lock().replace(message);
            tx.receiver_alive.load(Ordering::Acquire)
        }
        message @ WorkerMessage::Log(_) => match tx.logs.try_send(message) {
            Ok(()) | Err(mpsc::TrySendError::Full(_)) => true,
            Err(mpsc::TrySendError::Disconnected(_)) => false,
        },
        message => send_critical_worker_message(tx, message),
    }
}

fn send_critical_worker_message(tx: &WorkerMessageSender, mut message: WorkerMessage) -> bool {
    let terminal = matches!(
        message,
        WorkerMessage::RunCompleted
            | WorkerMessage::Error { .. }
            | WorkerMessage::WorkerExited { .. }
    );
    loop {
        match tx.critical.send_timeout(message, WORKER_SEND_RETRY) {
            Ok(()) => return true,
            Err(SendTimeoutError::Disconnected(_)) => return false,
            Err(SendTimeoutError::Timeout(returned)) => {
                message = returned;
                if !tx.receiver_alive.load(Ordering::Acquire)
                    || (tx.cancelled.load(Ordering::Acquire) && !terminal)
                {
                    return false;
                }
            }
        }
    }
}
