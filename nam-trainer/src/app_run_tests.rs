use crate::worker::WorkerMessage;
use std::path::PathBuf;
use std::sync::Mutex;

struct RecordingNotifier {
    messages: Mutex<Vec<(String, String)>>,
}

impl super::Notifier for RecordingNotifier {
    fn notify(&self, title: &str, body: &str) -> Result<(), std::io::Error> {
        self.messages
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push((title.to_string(), body.to_string()));
        Ok(())
    }
}

#[test]
fn notifier_boundary_is_testable_without_platform_processes() {
    let notifier = RecordingNotifier {
        messages: Mutex::new(Vec::new()),
    };

    super::Notifier::notify(&notifier, "Training", "Complete").unwrap();

    assert_eq!(
        *notifier
            .messages
            .lock()
            .unwrap_or_else(|error| error.into_inner()),
        vec![("Training".into(), "Complete".into())]
    );
}

#[test]
fn notification_commands_are_constructed_for_each_platform() {
    let mac = super::notification_command(
        super::NotificationPlatform::MacOs,
        "NAM \"Trainer\"",
        "Done \"now\"",
    );
    assert_eq!(mac.program, "osascript");
    assert_eq!(mac.args[0], "-e");
    assert!(mac.args[1].contains("Done \\\"now\\\""));
    assert!(mac.args[1].contains("NAM \\\"Trainer\\\""));

    let windows = super::notification_command(
        super::NotificationPlatform::Windows,
        "NAM's Trainer",
        "It's done",
    );
    assert_eq!(windows.program, "powershell");
    assert_eq!(windows.args[0], "-NoProfile");
    assert!(windows.args[2].contains("NAM''s Trainer"));
    assert!(windows.args[2].contains("It''s done"));

    let linux =
        super::notification_command(super::NotificationPlatform::Linux, "Training", "Complete");
    assert_eq!(linux.program, "notify-send");
    assert_eq!(linux.args, ["Training", "Complete"]);
}

#[test]
fn diagnostics_summary_includes_environment_request_and_recent_log() {
    let mut app = test_app();
    app.input_path = Some("input.wav".into());
    app.output_paths = vec!["output.wav".into()];
    app.destination_dir = Some("models".into());
    app.training_log.push("recent failure".into());
    app.python_status = super::PythonStatus::Ok {
        version: "3.12.0".into(),
        devices: vec![super::TrainingDevice {
            id: "cpu".into(),
            name: "CPU".into(),
        }],
        warnings: Vec::new(),
        report: super::EnvironmentReport {
            nam_version: Some("0.12.0".into()),
            torch_version: Some("2.7.0".into()),
            packed_full_config_supported: true,
        },
    };

    let diagnostics = super::build_diagnostics_summary(&app);

    assert!(diagnostics.contains("nam_version: 0.12.0"));
    assert!(diagnostics.contains("\"input_path\": \"input.wav\""));
    assert!(diagnostics.contains("recent failure"));
}

#[test]
fn run_ids_are_unique_when_created_back_to_back() {
    let first = super::make_run_id();
    let second = super::make_run_id();

    assert_ne!(first, second);
}

#[test]
fn run_lifecycle_variants_preserve_valid_resource_ownership() {
    let mut run = super::TrainingRunContext::default();
    assert!(matches!(run, super::TrainingRunContext::Idle(_)));

    let (_tx, rx) = crate::worker::worker_message_channel(1);
    run.activate(None, rx);
    assert!(matches!(run, super::TrainingRunContext::Running { .. }));

    run.finish(super::TrainingRunResult::Complete);
    assert!(matches!(
        run,
        super::TrainingRunContext::Finishing {
            result: super::TrainingRunResult::Complete,
            ..
        }
    ));

    run.worker_exited();
    assert!(matches!(
        run,
        super::TrainingRunContext::Finished {
            result: super::TrainingRunResult::Complete,
            ..
        }
    ));
}

#[test]
fn unmanaged_miniforge_directory_is_never_removed() {
    let (_temp, dir) = unique_test_dir("unmanaged_miniforge");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("keep.txt"), "user data").unwrap();

    let error = super::remove_managed_miniforge(&dir).unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(dir.join("keep.txt").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn managed_miniforge_directory_can_be_removed() {
    let (_temp, dir) = unique_test_dir("managed_miniforge");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join(super::MANAGED_MINIFORGE_MARKER),
        "miniforge_version=test\n",
    )
    .unwrap();

    super::remove_managed_miniforge(&dir).unwrap();

    assert!(!dir.exists());
}

#[test]
fn pinned_installer_has_sha256_digest_and_versioned_url() {
    let installer = super::miniforge_installer_info().unwrap();

    assert_eq!(installer.sha256.len(), 64);
    assert!(installer
        .sha256
        .chars()
        .all(|character| character.is_ascii_hexdigit()));
    assert!(installer.url.contains(super::MINIFORGE_VERSION));
    assert!(!installer.url.contains("/latest/"));
}

#[test]
fn restore_interrupted_run_reports_saved_artifacts() {
    let (_temp, dir) = unique_test_dir("restore_interrupted");
    std::fs::create_dir_all(&dir).unwrap();
    let log_path = dir.join("run.training.log");
    let manifest_path = dir.join("run.training-manifest.json");
    std::fs::write(&log_path, "partial log\n").unwrap();
    std::fs::write(&manifest_path, "{}").unwrap();
    let mut app = test_app();
    app.settings.active_run_id = Some("run".into());
    app.settings.active_run_log_path = Some(log_path);
    app.settings.active_run_manifest_path = Some(manifest_path);

    app.restore_interrupted_run();

    assert!(app.run.data().artifacts.is_some());
    assert!(matches!(
        app.run.state(),
        super::TrainingState::Error(ref msg)
            if msg.contains("Previous training run did not finish")
    ));
    assert!(app
        .training_log
        .iter()
        .any(|line| line.contains("Recovered interrupted run log")));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn poll_worker_transitions_through_fake_training_flow() {
    let (_temp, dir) = unique_test_dir("fake_training_flow");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.nam");
    let mut app = test_app();
    let (tx, rx) = crate::worker::worker_message_channel(8);
    app.run.activate(None, rx);

    tx.send(WorkerMessage::TrainingStart {
        file: "output.wav".into(),
        total_epochs: 2,
    })
    .unwrap();
    tx.send(WorkerMessage::EpochEnd {
        epoch: 1,
        train_loss: 0.1,
        val_loss: 0.2,
        esr: 0.3,
    })
    .unwrap();
    tx.send(WorkerMessage::FilePublished {
        file: "output.wav".into(),
        validation_esr: 0.3,
        log_path: model_path.with_extension("training.log"),
        manifest_path: model_path.with_extension("training-manifest.json"),
        model_path: model_path.clone(),
    })
    .unwrap();

    app.poll_worker();

    assert_eq!(app.run.state(), super::TrainingState::Training);
    assert_eq!(app.run.data().completed_models.len(), 1);

    tx.send(WorkerMessage::RunCompleted).unwrap();
    app.poll_worker();

    assert_eq!(app.run.state(), super::TrainingState::Complete);
    assert_eq!(app.run.data().current_file_index, 1);
    assert_eq!(app.epoch_history.len(), 1);
    assert_eq!(app.model_path.as_deref(), Some(model_path.as_path()));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn poll_worker_limits_work_per_frame() {
    let mut app = test_app();
    let (tx, rx) = crate::worker::worker_message_channel(128);
    app.run.activate(None, rx);
    for index in 0..80 {
        tx.send(WorkerMessage::Log(format!("line {index}")))
            .unwrap();
    }

    assert!(app.poll_worker());
    assert_eq!(app.training_log.len(), 64);
    assert!(!app.poll_worker());
    assert_eq!(app.training_log.len(), 80);
}

#[test]
fn poll_worker_reports_partial_batch_failure_after_run_completion() {
    let (_temp, dir) = unique_test_dir("partial_batch_failure");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.nam");
    let mut app = test_app();
    let (tx, rx) = crate::worker::worker_message_channel(8);
    app.run.activate(None, rx);

    tx.send(WorkerMessage::FileFailed {
        file: "bad.wav".into(),
        kind: crate::worker::protocol::WorkerErrorKind::DataValidation,
        error: "sample rate mismatch".into(),
    })
    .unwrap();
    tx.send(WorkerMessage::FilePublished {
        file: "good.wav".into(),
        validation_esr: 0.2,
        log_path: model_path.with_extension("training.log"),
        manifest_path: model_path.with_extension("training-manifest.json"),
        model_path: model_path.clone(),
    })
    .unwrap();
    tx.send(WorkerMessage::RunCompleted).unwrap();

    app.poll_worker();

    assert_eq!(app.run.data().completed_models, vec![model_path]);
    assert_eq!(app.run.data().failed_files.len(), 1);
    assert!(matches!(
        app.run.state(),
        super::TrainingState::Error(ref message)
            if message.contains("1 failed file(s)") && message.contains("1 completed model(s)")
    ));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cancel_training_resets_state_and_records_log() {
    let (_temp, dir) = unique_test_dir("cancel_manifest");
    std::fs::create_dir_all(&dir).unwrap();
    let mut app = test_app();
    app.input_path = Some("original-input.wav".into());
    app.output_paths = vec!["original-output.wav".into()];
    app.python_path = "original-python".into();
    app.config.epochs = 42;
    let mut run = super::ActiveTrainingRun::from_existing(
        "run",
        dir.join("run.training.log"),
        dir.join("run.training-manifest.json"),
    );
    run.capture_test_snapshot(&app).unwrap();
    std::fs::write(&run.log_path, "").unwrap();
    app.run.data_mut().artifacts = Some(run.clone());
    app.destination_dir = Some(dir.clone());
    app.run.set_test_state(super::TrainingState::Training);
    app.settings.active_run_id = Some("run".into());
    app.settings.active_run_log_path = Some(run.log_path.clone());
    app.settings.active_run_manifest_path = Some(run.manifest_path.clone());
    app.input_path = Some("changed-input.wav".into());
    app.python_path = "changed-python".into();
    app.config.epochs = 7;

    app.cancel_training();

    assert_eq!(app.run.state(), super::TrainingState::Idle);
    assert!(!app.run.is_active());
    assert!(app
        .training_log
        .iter()
        .any(|line| line == "Training cancelled."));
    assert!(app.settings.active_run_id.is_none());
    let manifest = std::fs::read_to_string(&run.manifest_path).unwrap();
    assert!(manifest.contains("\"status\": \"cancelled\""));
    let manifest: serde_json::Value = serde_json::from_str(&manifest).unwrap();
    assert_eq!(manifest["schema_version"], 2);
    assert_eq!(manifest["input_path"], "original-input.wav");
    assert_eq!(manifest["python_path"], "original-python");
    assert_eq!(manifest["request"]["epochs"], 42);
    assert!(manifest["started_unix_seconds"].is_u64());
    assert!(manifest["completed_unix_seconds"].is_u64());

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn failed_cleanup_preserves_recovery_settings() {
    let (_temp, dir) = unique_test_dir("failed_cleanup_recovery");
    std::fs::create_dir_all(&dir).unwrap();
    let run = super::ActiveTrainingRun::from_existing(
        "run",
        dir.join("run.training.log"),
        dir.join("run.training-manifest.json"),
    );
    std::fs::write(run.staging_dir(), b"not a directory").unwrap();
    let mut app = test_app();
    app.run.data_mut().artifacts = Some(run.clone());
    app.run
        .finish(super::TrainingRunResult::Error("failed".into()));
    app.settings.active_run_id = Some("run".into());
    app.settings.active_run_log_path = Some(run.log_path.clone());
    app.settings.active_run_manifest_path = Some(run.manifest_path.clone());
    app.settings.active_run_staging_dir = Some(run.staging_dir().to_path_buf());

    app.reset_after_error();

    assert_eq!(app.settings.active_run_id.as_deref(), Some("run"));
    assert!(app.run.data().artifacts.is_some());
    assert!(app
        .user_action_error
        .as_deref()
        .is_some_and(|message| message.contains("Recovery information was retained")));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn completed_training_records_recent_run_history() {
    let (_temp, dir) = unique_test_dir("recent_run_history");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.nam");
    std::fs::write(&model_path, b"{}").unwrap();
    let mut app = test_app();
    app.input_path = Some("input.wav".into());
    app.output_paths = vec!["output.wav".into()];
    app.destination_dir = Some(dir.clone());
    let mut run = super::ActiveTrainingRun::from_existing(
        "run",
        dir.join("run.training.log"),
        dir.join("run.training-manifest.json"),
    );
    run.capture_test_snapshot(&app).unwrap();
    app.run.data_mut().artifacts = Some(run);
    let log_path = app.run.data().artifacts.as_ref().unwrap().log_path.clone();
    std::fs::write(&log_path, "line\n").unwrap();
    let (tx, rx) = crate::worker::worker_message_channel(8);
    app.run.activate(None, rx);
    tx.send(WorkerMessage::EpochEnd {
        epoch: 1,
        train_loss: 0.1,
        val_loss: 0.2,
        esr: 0.345,
    })
    .unwrap();
    let (model_path, log_path, manifest_path) = crate::run_manifest::publish_artifact_bundle(
        &model_path,
        app.run.data().artifacts.as_ref().unwrap(),
        &dir,
    )
    .unwrap();
    tx.send(WorkerMessage::FilePublished {
        file: "output.wav".into(),
        validation_esr: 0.345,
        model_path: model_path.clone(),
        log_path,
        manifest_path,
    })
    .unwrap();

    app.poll_worker();

    assert_eq!(app.settings.recent_runs.len(), 1);
    let recent = &app.settings.recent_runs[0];
    assert_eq!(recent.model_path, model_path);
    assert_eq!(recent.esr, Some(0.345));
    assert_eq!(recent.architecture, "standard");
    assert_eq!(recent.device, "cpu");

    std::fs::remove_dir_all(dir).unwrap();
}

fn test_app() -> super::TrainerApp {
    super::TrainerApp {
        input_path: None,
        output_paths: Vec::new(),
        destination_dir: None,
        config: super::TrainingConfig::default(),
        metadata: super::ModelMetadata::default(),
        allow_overwrite_outputs: false,
        show_advanced: false,
        show_metadata: false,
        training_log: Vec::new(),
        epoch_history: Vec::new(),
        model_path: None,
        run: super::TrainingRunContext::default(),
        user_action_error: None,
        settings: super::Settings::default(),
        python_path: "python3".into(),
        selected_device: "cpu".into(),
        discovered_pythons: None,
        python_discovery_rx: None,
        python_status: super::PythonStatus::Unknown,
        cuda_install: None,
        python_check_rx: None,
        install_state: super::InstallState::Idle,
        install_log: Vec::new(),
        install_rx: None,
        pending_destructive_action: None,
    }
}

fn unique_test_dir(name: &str) -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::Builder::new()
        .prefix(&format!("nam-trainer-app-{name}-"))
        .tempdir()
        .unwrap();
    let path = temp.path().to_path_buf();
    (temp, path)
}
