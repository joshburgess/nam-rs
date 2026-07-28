use std::time::Instant;

use crate::app::{send_notification, EpochStats, TrainerApp, TrainingFileFailure};
use crate::errors::{TrainingErrorDetails, TrainingErrorKind};
use crate::run_controller::RunFinalization;
use crate::worker::{TrainingState, WorkerMessage};

const MAX_WORKER_MESSAGES_PER_FRAME: usize = 64;
const WORKER_POLL_TIME_BUDGET: std::time::Duration = std::time::Duration::from_millis(4);

impl TrainerApp {
    pub fn poll_worker(&mut self) -> bool {
        let started = Instant::now();
        for index in 0..MAX_WORKER_MESSAGES_PER_FRAME {
            if index > 0 && started.elapsed() >= WORKER_POLL_TIME_BUDGET {
                return true;
            }
            let message = match self
                .run
                .messages()
                .and_then(|receiver| receiver.try_recv().ok())
            {
                Some(message) => message,
                None => return false,
            };
            match message {
                WorkerMessage::Log(text) => self.push_training_log(text),
                WorkerMessage::TrainingStart {
                    ref file,
                    total_epochs,
                } => {
                    self.run.data_mut().current_file_index += 1;
                    self.push_training_log(format!("Training {file} ({total_epochs} epochs)..."));
                }
                WorkerMessage::EpochEnd {
                    epoch,
                    train_loss,
                    val_loss,
                    esr,
                } => {
                    let now = Instant::now();
                    if let Some(last) = self.run.data().last_epoch_at {
                        let elapsed = now.duration_since(last).as_secs_f64();
                        let data = self.run.data_mut();
                        data.avg_epoch_secs = Some(match data.avg_epoch_secs {
                            Some(average) => average * 0.7 + elapsed * 0.3,
                            None => elapsed,
                        });
                    }
                    self.run.data_mut().last_epoch_at = Some(now);
                    self.epoch_history.push(EpochStats {
                        epoch,
                        train_loss,
                        val_loss,
                        esr,
                    });
                    self.push_training_log(format!(
                        "Epoch {epoch}: loss={train_loss:.6} val_loss={val_loss:.6} ESR={esr:.6}"
                    ));
                }
                WorkerMessage::FileCompleted { file, .. } => {
                    let message =
                        "worker completion reached the UI before artifact publication".to_string();
                    self.push_training_log(format!("File failed: {file}: {message}"));
                    self.run.data_mut().failed_files.push(TrainingFileFailure {
                        file,
                        kind: TrainingErrorKind::SubprocessCrash,
                        message,
                    });
                }
                WorkerMessage::FilePublished {
                    file,
                    validation_esr,
                    model_path,
                    log_path,
                    manifest_path,
                } => {
                    self.record_recent_successful_run(
                        &model_path,
                        manifest_path.clone(),
                        validation_esr,
                    );
                    self.push_training_log(format!(
                        "Artifact bundle published: {}, {}, {}",
                        model_path.display(),
                        log_path.display(),
                        manifest_path.display()
                    ));
                    self.push_training_log(format!(
                        "File complete: {file} ESR={validation_esr:.6} Model: {}",
                        model_path.display()
                    ));
                    self.model_path = Some(model_path.clone());
                    self.run.data_mut().completed_models.push(model_path);
                }
                WorkerMessage::FilePublicationFailed { file, error } => {
                    let message = format!("failed to publish artifact bundle: {error}");
                    self.push_training_log(format!("File failed: {file}: {message}"));
                    self.run.data_mut().failed_files.push(TrainingFileFailure {
                        file,
                        kind: TrainingErrorKind::SubprocessCrash,
                        message,
                    });
                }
                WorkerMessage::FileFailed { file, kind, error } => {
                    let details = TrainingErrorDetails::new(kind.into(), error);
                    let classified = details.classified_message();
                    self.push_training_log(format!("File failed: {file}: {classified}"));
                    self.run.data_mut().failed_files.push(TrainingFileFailure {
                        file,
                        kind: details.kind,
                        message: details.message,
                    });
                }
                WorkerMessage::RunCompleted => {
                    let completed = self.run.data().completed_models.len();
                    let failed = self.run.data().failed_files.len();
                    if failed == 0 {
                        self.push_training_log(format!(
                            "Training run complete: {completed} model(s) created."
                        ));
                        send_notification(
                            "NAM Trainer",
                            &format!("Training complete: {completed} model(s) created"),
                        );
                        self.finalize_run(RunFinalization::Complete);
                    } else {
                        let message = format!(
                            "Training run finished with {failed} failed file(s) and {completed} completed model(s)"
                        );
                        self.push_training_log(format!("Error: {message}"));
                        self.finalize_run(RunFinalization::Failed(message));
                    }
                }
                WorkerMessage::Error { kind, message } => {
                    let details = TrainingErrorDetails::new(kind.into(), message);
                    let classified = details.classified_message();
                    self.push_training_log(format!("Error: {classified}"));
                    self.finalize_run(RunFinalization::Failed(classified));
                }
                WorkerMessage::WorkerExited { exit_code } => {
                    if self.run.state() == TrainingState::Training {
                        let message = match exit_code {
                            Some(code) => {
                                format!(
                                    "Subprocess crash: worker exited unexpectedly with code {code}"
                                )
                            }
                            None => "Subprocess crash: worker exited unexpectedly".into(),
                        };
                        self.push_training_log(format!("Error: {message}"));
                        self.finalize_run(RunFinalization::Failed(message));
                    }
                    self.run.worker_exited();
                }
            }
        }
        true
    }
}
