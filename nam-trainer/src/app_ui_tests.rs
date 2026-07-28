use std::path::PathBuf;

#[test]
fn ui_smoke_renders_default_running_error_and_complete_states() {
    let states = [
        super::TrainingState::Idle,
        super::TrainingState::Training,
        super::TrainingState::Error("failed".into()),
        super::TrainingState::Complete,
    ];

    for state in states {
        let mut app = test_app();
        app.run.set_test_state(state);
        app.python_status = super::PythonStatus::Ok {
            version: "3.12.0".into(),
            devices: vec![super::TrainingDevice {
                id: "cpu".into(),
                name: "CPU".into(),
            }],
            warnings: Vec::new(),
            report: super::EnvironmentReport::default(),
        };
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                crate::ui::main_panel::show(&mut app, ui);
            });
        });
    }
}

#[test]
fn app_state_text_snapshot_covers_key_controls() {
    let mut app = test_app();
    app.run.set_test_state(super::TrainingState::Idle);
    let idle = app_state_text_snapshot(&app);
    assert_eq!(idle, "state=idle; controls=Train,Allow overwriting outputs");

    app.run.set_test_state(super::TrainingState::Training);
    app.run.data_mut().total_files = 2;
    app.run.data_mut().current_file_index = 1;
    let running = app_state_text_snapshot(&app);
    assert_eq!(running, "state=training; controls=Cancel; progress=file");

    app.run
        .set_test_state(super::TrainingState::Error("failed".into()));
    let error = app_state_text_snapshot(&app);
    assert_eq!(error, "state=error; controls=Reset,Copy Diagnostics");

    app.run.set_test_state(super::TrainingState::Complete);
    app.model_path = Some("/tmp/model.nam".into());
    app.destination_dir = Some("/tmp".into());
    let complete = app_state_text_snapshot(&app);
    assert_eq!(
        complete,
        "state=complete; controls=Train Again,Open Output Folder,Reveal Model,Copy Diagnostics"
    );
}

#[test]
fn control_actions_mutate_expected_state() {
    let mut app = test_app();

    app.set_allow_overwrite_outputs(true);
    assert!(app.allow_overwrite_outputs);
    assert_eq!(app.settings.allow_overwrite_outputs, Some(true));

    app.run
        .set_test_state(super::TrainingState::Error("failed".into()));
    app.training_log.push("line".into());
    app.epoch_history.push(super::EpochStats {
        epoch: 1,
        train_loss: 0.1,
        val_loss: 0.2,
        esr: 0.3,
    });
    app.reset_after_error();
    assert_eq!(app.run.state(), super::TrainingState::Idle);
    assert!(app.training_log.is_empty());
    assert!(app.epoch_history.is_empty());

    app.run.set_test_state(super::TrainingState::Complete);
    app.model_path = Some("model.nam".into());
    app.run.data_mut().artifacts = Some(super::ActiveTrainingRun::from_existing(
        "run",
        PathBuf::from("run.log"),
        PathBuf::from("run.json"),
    ));
    app.prepare_train_again();
    assert_eq!(app.run.state(), super::TrainingState::Idle);
    assert!(app.model_path.is_none());
    assert!(app.run.data().artifacts.is_none());

    app.input_path = Some("input.wav".into());
    app.output_paths = vec!["output.wav".into()];
    app.destination_dir = Some("models".into());
    let diagnostics = app.diagnostics_text();
    assert!(diagnostics.contains("request:"));
    assert!(diagnostics.contains("\"output_paths\""));
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

fn app_state_text_snapshot(app: &super::TrainerApp) -> String {
    match app.run.state() {
        super::TrainingState::Idle => "state=idle; controls=Train,Allow overwriting outputs".into(),
        super::TrainingState::Training => {
            let progress = if app.run.data().total_files > 1 {
                "; progress=file"
            } else {
                ""
            };
            format!("state=training; controls=Cancel{progress}")
        }
        super::TrainingState::Complete => {
            let mut controls = vec!["Train Again"];
            if app.destination_dir.is_some() {
                controls.push("Open Output Folder");
            }
            if app.model_path.is_some() {
                controls.push("Reveal Model");
            }
            controls.push("Copy Diagnostics");
            format!("state=complete; controls={}", controls.join(","))
        }
        super::TrainingState::Error(_) => "state=error; controls=Reset,Copy Diagnostics".into(),
    }
}
