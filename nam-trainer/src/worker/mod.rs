pub mod protocol;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;

use crate::app::TrainerApp;

#[derive(Clone, Debug, PartialEq)]
pub enum TrainingState {
    Idle,
    Training,
    Complete,
    Error(String),
}

#[derive(Debug)]
pub enum WorkerMessage {
    Log(String),
    EpochEnd {
        epoch: u32,
        train_loss: f64,
        val_loss: f64,
        esr: f64,
    },
    TrainingComplete {
        model_path: String,
        esr: f64,
    },
    Error(String),
    WorkerExited,
}

pub struct WorkerHandle {
    child: Option<Child>,
}

impl WorkerHandle {
    pub fn cancel(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Spawn the Python worker subprocess and return a handle + message receiver.
pub fn spawn(app: &TrainerApp) -> (WorkerHandle, mpsc::Receiver<WorkerMessage>) {
    let (tx, rx) = mpsc::channel();

    let request = protocol::TrainRequest {
        input_path: app.input_path.clone().unwrap_or_default(),
        output_paths: app.output_paths.clone(),
        destination: app.destination_dir.clone().unwrap_or_default(),
        architecture: app.config.architecture.as_str().to_string(),
        epochs: app.config.epochs,
        batch_size: app.config.batch_size,
        lr: app.config.lr,
        lr_decay: app.config.lr_decay,
        latency: app.config.latency,
        threshold_esr: app.config.threshold_esr,
        save_plot: app.config.save_plot,
        fit_mrstft: app.config.fit_mrstft,
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
    };

    let request_json = serde_json::to_string(&request).unwrap_or_default();
    let python_path = app.python_path.clone();

    // Find the worker script relative to the executable
    let worker_script = find_worker_script();

    thread::spawn(move || {
        let result = Command::new(&python_path)
            .arg(&worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match result {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(WorkerMessage::Error(format!(
                    "Failed to start Python ({}): {}",
                    python_path, e
                )));
                return;
            }
        };

        // Write the config to stdin
        if let Some(ref mut stdin) = child.stdin.take() {
            let _ = writeln!(stdin, "{}", request_json);
        }

        // Read stdout line by line
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };

                if line.trim().is_empty() {
                    continue;
                }

                match serde_json::from_str::<protocol::WorkerEvent>(&line) {
                    Ok(event) => {
                        let msg = event_to_message(event);
                        if tx.send(msg).is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        // Non-JSON line, treat as log output
                        if tx.send(WorkerMessage::Log(line)).is_err() {
                            break;
                        }
                    }
                }
            }
        }

        // Wait for process to finish
        let _ = child.wait();
        let _ = tx.send(WorkerMessage::WorkerExited);
    });

    // We don't have the Child in the main thread (it's in the spawned thread),
    // so we can't kill it directly. For now, we'll use a simple handle.
    // A more robust solution would use Arc<Mutex<Child>>.
    let handle = WorkerHandle { child: None };
    (handle, rx)
}

fn event_to_message(event: protocol::WorkerEvent) -> WorkerMessage {
    match event {
        protocol::WorkerEvent::EpochEnd {
            epoch,
            train_loss,
            val_loss,
            esr,
        } => WorkerMessage::EpochEnd {
            epoch,
            train_loss,
            val_loss,
            esr,
        },
        protocol::WorkerEvent::TrainingComplete {
            model_path,
            validation_esr,
            ..
        } => WorkerMessage::TrainingComplete {
            model_path,
            esr: validation_esr,
        },
        protocol::WorkerEvent::TrainingFailed { error, .. } => WorkerMessage::Error(error),
        protocol::WorkerEvent::TrainingStart { file, total_epochs } => {
            WorkerMessage::Log(format!("Training {} ({} epochs)...", file, total_epochs))
        }
        protocol::WorkerEvent::AllComplete => WorkerMessage::Log("All training complete.".into()),
        protocol::WorkerEvent::Error { message } => WorkerMessage::Error(message),
        protocol::WorkerEvent::Log { message } => WorkerMessage::Log(message),
    }
}

fn find_worker_script() -> String {
    // Look for the worker script relative to the executable, then in common locations
    let candidates = [
        // Next to the binary
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("python").join("nam_worker.py")))
            .unwrap_or_default(),
        // Development location
        std::env::current_exe()
            .ok()
            .and_then(|p| {
                p.parent()?
                    .parent()?
                    .parent()
                    .map(|d| d.join("nam-trainer").join("python").join("nam_worker.py"))
            })
            .unwrap_or_default(),
        // CWD
        std::env::current_dir()
            .ok()
            .map(|d| d.join("nam-trainer").join("python").join("nam_worker.py"))
            .unwrap_or_default(),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.display().to_string();
        }
    }

    // Fallback
    "nam_worker.py".to_string()
}

fn non_empty(s: &str) -> Option<String> {
    if s.trim().is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}
