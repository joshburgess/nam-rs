use super::{
    create_worker_script, event_to_message, parse_worker_line, protocol, send_worker_message,
    wait_for_child, worker_message_channel, Clock, ProtocolDispatcher, ProtocolInput,
    ProtocolSequenceError, ProtocolSequenceValidator, SystemClock, SystemWorkerProcess,
    WorkerFailureKind, WorkerHandle, WorkerMessage, WorkerProcess, WorkerStartupError,
    MAX_PROTOCOL_REORDER_WINDOW,
};
#[cfg(unix)]
use super::{encode_worker_path, WorkerRequestError};
use proptest::prelude::*;
use serde_json::json;
use std::cell::Cell;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
        prop_assert!(dispatcher.failed);
        prop_assert!(dispatcher
            .accept(ProtocolInput::StdoutEvent(log_event(1)))
            .is_empty());
        prop_assert!(dispatcher.finish().is_empty());
    }
}

#[test]
fn protocol_dispatcher_rejects_all_domain_events_after_state_failure() {
    let mut dispatcher = ProtocolDispatcher::new("run".into());
    let failed = dispatcher.accept(ProtocolInput::StdoutEvent(complete_event(0, 1)));
    assert!(matches!(
        failed.as_slice(),
        [WorkerMessage::Error {
            kind: WorkerFailureKind::ProtocolSequence,
            ..
        }]
    ));
    assert!(dispatcher.failed);

    assert!(dispatcher
        .accept(ProtocolInput::StdoutEvent(start_event(0, 2)))
        .is_empty());
    assert!(dispatcher
        .accept(ProtocolInput::StderrEvent(log_event(3)))
        .is_empty());
    assert!(dispatcher.finish().is_empty());

    assert_eq!(
        dispatcher.accept(ProtocolInput::StdoutLog("diagnostic".into())),
        vec![WorkerMessage::Log("diagnostic".into())]
    );
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
    let child = Command::new(python)
        .args(["-c", "import time; time.sleep(30)"])
        .spawn()
        .unwrap();
    let cancelled = AtomicBool::new(true);
    let started = std::time::Instant::now();

    let mut process = SystemWorkerProcess { child, stdin: None };
    let _ = wait_for_child(&mut process, &cancelled, &SystemClock);

    assert!(started.elapsed() < std::time::Duration::from_secs(2));
}

#[test]
fn system_worker_sends_cooperative_cancellation() {
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
    let mut process = SystemWorkerProcess {
        child,
        stdin: child_stdin,
    };

    process.request_cancel().unwrap();
    let exit_code = process.wait().unwrap();

    assert_eq!(exit_code, Some(0));
}

struct FakeClock {
    now: Cell<Instant>,
    sleeps: Cell<usize>,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            now: Cell::new(Instant::now()),
            sleeps: Cell::new(0),
        }
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        self.now.get()
    }

    fn sleep(&self, duration: Duration) {
        self.now.set(self.now.get() + duration);
        self.sleeps.set(self.sleeps.get() + 1);
    }
}

struct FakeWorkerProcess {
    cancel_succeeds: bool,
    polls_before_exit: Option<usize>,
    poll_error: bool,
    exit_code: Option<i32>,
    cancel_requests: usize,
    polls: usize,
    kills: usize,
    waits: usize,
}

impl FakeWorkerProcess {
    fn running() -> Self {
        Self {
            cancel_succeeds: true,
            polls_before_exit: None,
            poll_error: false,
            exit_code: Some(9),
            cancel_requests: 0,
            polls: 0,
            kills: 0,
            waits: 0,
        }
    }
}

impl WorkerProcess for FakeWorkerProcess {
    fn request_cancel(&mut self) -> io::Result<()> {
        self.cancel_requests += 1;
        if self.cancel_succeeds {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "injected cancellation failure",
            ))
        }
    }

    fn try_wait(&mut self) -> io::Result<Option<Option<i32>>> {
        self.polls += 1;
        if self.poll_error {
            return Err(io::Error::other("injected poll failure"));
        }
        Ok(self
            .polls_before_exit
            .filter(|polls| self.polls > *polls)
            .map(|_| self.exit_code))
    }

    fn kill(&mut self) -> io::Result<()> {
        self.kills += 1;
        Ok(())
    }

    fn wait(&mut self) -> io::Result<Option<i32>> {
        self.waits += 1;
        Ok(self.exit_code)
    }
}

#[test]
fn fake_worker_exits_cooperatively_before_the_kill_deadline() {
    let mut process = FakeWorkerProcess {
        polls_before_exit: Some(2),
        exit_code: Some(0),
        ..FakeWorkerProcess::running()
    };
    let clock = FakeClock::new();
    let cancelled = AtomicBool::new(true);

    assert_eq!(wait_for_child(&mut process, &cancelled, &clock), Some(0));
    assert_eq!(process.cancel_requests, 1);
    assert_eq!(process.kills, 0);
    assert_eq!(process.waits, 0);
    assert_eq!(clock.sleeps.get(), 2);
}

#[test]
fn fake_worker_is_killed_exactly_at_the_cancellation_deadline() {
    let mut process = FakeWorkerProcess::running();
    let clock = FakeClock::new();
    let cancelled = AtomicBool::new(true);

    assert_eq!(wait_for_child(&mut process, &cancelled, &clock), Some(9));
    assert_eq!(process.cancel_requests, 1);
    assert_eq!(process.kills, 1);
    assert_eq!(process.waits, 1);
    assert_eq!(
        clock.sleeps.get(),
        super::COOPERATIVE_CANCEL_GRACE.as_millis() as usize / 20
    );
}

#[test]
fn failed_cooperative_request_forces_immediate_termination() {
    let mut process = FakeWorkerProcess {
        cancel_succeeds: false,
        ..FakeWorkerProcess::running()
    };
    let clock = FakeClock::new();
    let cancelled = AtomicBool::new(true);

    assert_eq!(wait_for_child(&mut process, &cancelled, &clock), Some(9));
    assert_eq!(process.kills, 1);
    assert_eq!(process.waits, 1);
    assert_eq!(clock.sleeps.get(), 0);
}

#[test]
fn poll_failure_falls_back_to_waiting_for_exit() {
    let mut process = FakeWorkerProcess {
        poll_error: true,
        exit_code: Some(7),
        ..FakeWorkerProcess::running()
    };
    let clock = FakeClock::new();
    let cancelled = AtomicBool::new(false);

    assert_eq!(wait_for_child(&mut process, &cancelled, &clock), Some(7));
    assert_eq!(process.kills, 0);
    assert_eq!(process.waits, 1);
    assert_eq!(clock.sleeps.get(), 0);
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
    let sender = std::thread::spawn(move || send_worker_message(&tx, WorkerMessage::RunCompleted));

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
