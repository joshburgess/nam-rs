use super::{validate_audio_files, Architecture};
use crate::training_validation::{
    reconcile_selected_device, validate_destination_dir, validate_training_artifacts,
    validate_training_metadata,
};
use crate::worker::WorkerRequestError;
use std::path::PathBuf;

#[test]
fn packed_architecture_round_trips() {
    assert_eq!(Architecture::Packed.as_str(), "packed");
    assert_eq!(Architecture::parse_lossy("packed"), Architecture::Packed);
    assert_eq!(Architecture::parse_lossy("a2"), Architecture::Packed);
    assert_eq!(Architecture::parse_lossy("packed_a2"), Architecture::Packed);
}

#[test]
fn packed_architecture_is_available_first() {
    assert_eq!(
        Architecture::all().first().copied(),
        Some(Architecture::Packed)
    );
}

#[test]
fn worker_request_reports_missing_artifacts_structurally() {
    let mut app = test_app();
    assert!(matches!(
        crate::worker::build_train_request(&app),
        Err(WorkerRequestError::MissingDestination)
    ));

    app.destination_dir = Some("/tmp/models".into());
    assert!(matches!(
        crate::worker::build_train_request(&app),
        Err(WorkerRequestError::MissingInputWav)
    ));

    app.input_path = Some("input.wav".into());
    assert!(matches!(
        crate::worker::build_train_request(&app),
        Err(WorkerRequestError::MissingOutputWavs)
    ));
}

#[test]
fn validate_audio_files_accepts_matching_wavs() {
    let (_temp, dir) = unique_test_dir("matching");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.wav");
    let output = dir.join("output.wav");
    write_test_wav(&input, 48_000, 2.0);
    write_test_wav(&output, 48_000, 2.0);

    let issues = validate_audio_files(&input, std::slice::from_ref(&output));
    assert!(
        issues.is_empty(),
        "unexpected validation issues: {issues:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_audio_files_reports_sample_rate_and_duration_issues() {
    let (_temp, dir) = unique_test_dir("mismatch");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.wav");
    let output = dir.join("output.wav");
    write_test_wav(&input, 48_000, 2.0);
    write_test_wav(&output, 44_100, 0.25);

    let issues = validate_audio_files(&input, std::slice::from_ref(&output));
    assert!(
        issues
            .iter()
            .any(|issue| issue.message.contains("sample rate") && issue.is_error()),
        "missing sample-rate issue: {issues:?}"
    );
    assert!(
        issues
            .iter()
            .any(|issue| issue.message.contains("very short") && !issue.is_error()),
        "missing short-output issue: {issues:?}"
    );
    assert!(
        issues
            .iter()
            .any(|issue| issue.message.contains("differs significantly")),
        "missing duration-ratio issue: {issues:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_training_metadata_rejects_invalid_dbu_values() {
    let config = super::TrainingConfig::default();
    let metadata = super::ModelMetadata {
        input_level_dbu: "not a number".into(),
        output_level_dbu: "12.5".into(),
        ..super::ModelMetadata::default()
    };

    let issues = validate_training_metadata(&config, &metadata);
    assert_eq!(issues, vec!["Input level dBu must be a number"]);
}

#[test]
fn validate_training_metadata_blocks_full_config_metadata_drop() {
    let config = super::TrainingConfig {
        use_full_config_trainer: true,
        ..super::TrainingConfig::default()
    };
    let metadata = super::ModelMetadata {
        name: "Amp".into(),
        ..super::ModelMetadata::default()
    };

    let issues = validate_training_metadata(&config, &metadata);
    assert!(
        issues
            .iter()
            .any(|issue| issue.contains("does not export GUI metadata")),
        "missing full-config metadata issue: {issues:?}"
    );
}

#[test]
fn validate_destination_dir_accepts_writable_directory() {
    let (_temp, dir) = unique_test_dir("destination_ok");
    std::fs::create_dir_all(&dir).unwrap();

    let issues = validate_destination_dir(&dir);
    assert!(
        issues.is_empty(),
        "unexpected destination issues: {issues:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_destination_dir_rejects_missing_and_file_paths() {
    let (_temp, dir) = unique_test_dir("destination_bad");
    let missing = dir.join("missing");
    let file = dir.join("not_a_directory");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(&file, b"not a directory").unwrap();

    let missing_issues = validate_destination_dir(&missing);
    assert_eq!(missing_issues, vec!["Output directory does not exist"]);

    let file_issues = validate_destination_dir(&file);
    assert_eq!(file_issues, vec!["Output path is not a directory"]);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parse_environment_report_reads_package_and_api_fields() {
    let val = serde_json::json!({
        "nam_version": "0.12.0",
        "torch_version": "2.7.0",
        "packed_full_config_supported": true,
    });

    let report = super::parse_environment_report(&val);
    assert_eq!(report.nam_version.as_deref(), Some("0.12.0"));
    assert_eq!(report.torch_version.as_deref(), Some("2.7.0"));
    assert!(report.packed_full_config_supported);
}

#[test]
fn save_training_log_writes_log_next_to_model() {
    let (_temp, dir) = unique_test_dir("training_log");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.nam");
    std::fs::write(&model_path, b"{}").unwrap();

    let log_path =
        super::save_training_log(&model_path, None, &["line one".into(), "line two".into()])
            .unwrap();

    assert_eq!(log_path, dir.join("model.training.log"));
    assert_eq!(
        std::fs::read_to_string(&log_path).unwrap(),
        "line one\nline two"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn push_training_log_caps_in_memory_log() {
    let settings = super::Settings::default();
    let mut app = super::TrainerApp {
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
        settings,
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
    };

    for i in 0..super::MAX_TRAINING_LOG_LINES + 5 {
        app.push_training_log(format!("line {i}"));
    }

    assert_eq!(app.training_log.len(), super::MAX_TRAINING_LOG_LINES);
    assert_eq!(app.training_log.first().map(String::as_str), Some("line 5"));
}

#[test]
fn push_install_log_caps_in_memory_log() {
    let mut app = test_app();
    for index in 0..super::MAX_INSTALL_LOG_LINES + 5 {
        app.push_install_log(format!("installer line {index}"));
    }

    assert_eq!(app.install_log.len(), super::MAX_INSTALL_LOG_LINES);
    assert_eq!(
        app.install_log.first().map(String::as_str),
        Some("installer line 5")
    );
}

#[test]
fn validate_training_artifacts_blocks_existing_outputs() {
    let (_temp, dir) = unique_test_dir("artifact_overwrite");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.wav");
    let output = dir.join("output.wav");
    let existing_model = dir.join("output.nam");
    write_test_wav(&input, 48_000, 2.0);
    write_test_wav(&output, 48_000, 2.0);
    std::fs::write(&existing_model, b"old model").unwrap();

    let issues = validate_training_artifacts(
        &dir,
        Some(&input),
        std::slice::from_ref(&output),
        10,
        false,
        None,
        None,
    );

    assert!(
        issues.iter().any(|issue| issue.contains("already exists")),
        "missing overwrite issue: {issues:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_training_artifacts_allows_existing_outputs_when_enabled() {
    let (_temp, dir) = unique_test_dir("artifact_overwrite_allowed");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.wav");
    let output = dir.join("output.wav");
    let existing_model = dir.join("output.nam");
    write_test_wav(&input, 48_000, 2.0);
    write_test_wav(&output, 48_000, 2.0);
    std::fs::write(&existing_model, b"old model").unwrap();

    let issues = validate_training_artifacts(
        &dir,
        Some(&input),
        std::slice::from_ref(&output),
        10,
        true,
        None,
        None,
    );

    assert!(
        issues.iter().all(|issue| !issue.contains("already exists")),
        "overwrite issue should be bypassed: {issues:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_training_artifacts_never_allows_internal_name_collisions() {
    let (_temp, dir) = unique_test_dir("artifact_internal_collision");
    let first_dir = dir.join("first");
    let second_dir = dir.join("second");
    std::fs::create_dir_all(&first_dir).unwrap();
    std::fs::create_dir_all(&second_dir).unwrap();
    let input = dir.join("input.wav");
    let first_output = first_dir.join("capture.wav");
    let second_output = second_dir.join("capture.wav");
    write_test_wav(&input, 48_000, 2.0);
    write_test_wav(&first_output, 48_000, 2.0);
    write_test_wav(&second_output, 48_000, 2.0);

    let issues = validate_training_artifacts(
        &dir,
        Some(&input),
        &[first_output, second_output],
        10,
        true,
        None,
        None,
    );

    assert!(
        issues.iter().any(|issue| issue.contains("same artifact")),
        "internal collisions must remain blocking: {issues:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_training_artifacts_uses_custom_output_names_for_conflicts() {
    let (_temp, dir) = unique_test_dir("artifact_custom_name");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.wav");
    let output = dir.join("output.wav");
    let existing_model = dir.join("custom_model.nam");
    write_test_wav(&input, 48_000, 2.0);
    write_test_wav(&output, 48_000, 2.0);
    std::fs::write(&existing_model, b"old model").unwrap();

    let issues = validate_training_artifacts(
        &dir,
        Some(&input),
        std::slice::from_ref(&output),
        10,
        false,
        Some("custom:model"),
        None,
    );

    assert!(
        issues
            .iter()
            .any(|issue| issue.contains("custom_model.nam")),
        "missing custom-name conflict: {issues:?}"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn training_run_artifacts_predicts_model_log_and_manifest_paths() {
    let outputs = vec!["captures/amp take.wav".to_string()];
    let artifacts = crate::artifacts::TrainingRunArtifacts::new("/tmp/models", &outputs);
    let model_paths = artifacts.predicted_model_paths();

    assert_eq!(model_paths, vec![PathBuf::from("/tmp/models/amp take.nam")]);
    assert_eq!(
        crate::artifacts::TrainingRunArtifacts::log_path_for_model(&model_paths[0]),
        PathBuf::from("/tmp/models/amp take.training.log")
    );
    assert_eq!(
        crate::artifacts::TrainingRunArtifacts::manifest_path_for_model(&model_paths[0]),
        PathBuf::from("/tmp/models/amp take.training-manifest.json")
    );
}

#[test]
fn prepare_training_run_writes_log_and_manifest() {
    let (_temp, dir) = unique_test_dir("run_artifacts");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.wav");
    let output = dir.join("output.wav");
    write_test_wav(&input, 48_000, 2.0);
    write_test_wav(&output, 48_000, 2.0);
    let mut app = test_app();
    app.input_path = Some(input);
    app.output_paths = vec![output];
    app.destination_dir = Some(dir.clone());

    let run = super::prepare_training_run(&app).unwrap();

    assert!(run.log_path.exists());
    assert!(run.manifest_path.exists());
    let manifest = std::fs::read_to_string(&run.manifest_path).unwrap();
    assert!(manifest.contains("\"status\": \"running\""));
    assert!(manifest.contains("\"python_path\": \"python3\""));
    assert!(manifest.contains("\"app_version\""));
    assert!(manifest.contains("\"os\""));
    assert!(manifest.contains("\"input_wav\""));
    assert!(manifest.contains("\"sample_rate\": 48000"));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn push_training_log_streams_to_active_run_file() {
    let (_temp, dir) = unique_test_dir("stream_log");
    std::fs::create_dir_all(&dir).unwrap();
    let mut app = test_app();
    app.input_path = Some(dir.join("input.wav"));
    app.output_paths = vec![dir.join("output.wav")];
    app.destination_dir = Some(dir.clone());
    let run = super::prepare_training_run(&app).unwrap();
    let log_path = run.log_path.clone();
    app.run.data_mut().artifacts = Some(run);

    app.push_training_log("first");
    app.push_training_log("second");

    assert_eq!(
        std::fs::read_to_string(&log_path).unwrap(),
        "first\nsecond\n"
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn reconcile_selected_device_falls_back_to_cpu() {
    let mut app = test_app();
    app.selected_device = "cuda:0".into();
    app.python_status = super::PythonStatus::Ok {
        version: "3.12.0".into(),
        devices: vec![super::TrainingDevice {
            id: "cpu".into(),
            name: "CPU".into(),
        }],
        warnings: Vec::new(),
        report: super::EnvironmentReport::default(),
    };

    let message = reconcile_selected_device(&mut app).unwrap();

    assert_eq!(app.selected_device, "cpu");
    assert!(message.contains("Falling back to CPU"));
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
        .prefix(&format!("nam-trainer-validation-{name}-"))
        .tempdir()
        .unwrap();
    let path = temp.path().to_path_buf();
    (temp, path)
}

fn write_test_wav(path: &std::path::Path, sample_rate: u32, duration_secs: f64) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    let samples = (sample_rate as f64 * duration_secs).round() as usize;
    for _ in 0..samples {
        writer.write_sample::<i16>(0).unwrap();
    }
    writer.finalize().unwrap();
}
