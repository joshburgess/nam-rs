use nam_trainer::{
    classify_training_error, sanitize_model_basename, TrainingErrorKind, TrainingRunArtifacts,
    PROTOCOL_VERSION,
};

#[test]
fn library_exports_core_trainer_gui_helpers() {
    let outputs = vec!["capture/output.wav".to_string()];
    let artifacts = TrainingRunArtifacts::new("/tmp/models", &outputs)
        .with_naming(Some("Amp:Lead".into()), None);

    assert_eq!(PROTOCOL_VERSION, 3);
    assert_eq!(sanitize_model_basename("Amp:Lead"), "Amp_Lead");
    assert_eq!(
        artifacts.predicted_model_paths()[0].to_string_lossy(),
        "/tmp/models/Amp_Lead.nam"
    );
    assert_eq!(
        classify_training_error("CUDA out of memory").kind,
        TrainingErrorKind::Device
    );
}
