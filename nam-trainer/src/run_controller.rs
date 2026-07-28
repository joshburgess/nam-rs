use crate::app::{TrainerApp, TrainingRunResult};
use crate::run_manifest::{cleanup_run_resources, save_training_manifest, ManifestStatus};
use crate::worker;

pub(crate) enum RunFinalization {
    Complete,
    Failed(String),
    Cancelled,
}

impl TrainerApp {
    pub(crate) fn start_training(&mut self) {
        let run = match crate::app::prepare_training_run(self) {
            Ok(run) => run,
            Err(error) => {
                self.push_training_log(format!("Error: failed to prepare training run: {error}"));
                self.run
                    .finish_error("Failed to prepare training run artifacts.");
                return;
            }
        };

        self.training_log.clear();
        self.epoch_history.clear();
        self.model_path = None;
        let data = self.run.data_mut();
        data.completed_models.clear();
        data.failed_files.clear();
        data.artifacts = Some(run);
        self.record_active_run_settings();
        let data = self.run.data_mut();
        data.current_file_index = 0;
        data.total_files = self.output_paths.len();
        data.started_at = Some(std::time::Instant::now());
        data.last_epoch_at = None;
        data.avg_epoch_secs = None;
        self.push_training_log("Starting training...");

        let (handle, messages) = worker::spawn(self);
        self.run.activate(Some(handle), messages);
    }

    pub(crate) fn finalize_run(&mut self, finalization: RunFinalization) {
        let (status, result) = match finalization {
            RunFinalization::Complete => (ManifestStatus::Complete, TrainingRunResult::Complete),
            RunFinalization::Failed(message) => {
                (ManifestStatus::Failed, TrainingRunResult::Error(message))
            }
            RunFinalization::Cancelled => (ManifestStatus::Cancelled, TrainingRunResult::Cancelled),
        };

        let mut recovery_information_needed = false;
        if let Some(run) = self.run.data().artifacts.clone() {
            if let Err(error) = run.flush_log() {
                recovery_information_needed = true;
                if self.user_action_error.is_none() {
                    self.user_action_error = Some(format!("Training log flush failed: {error}"));
                }
            }
            let cleanup_report = cleanup_run_resources(Some(&run));
            if !cleanup_report.is_complete() {
                recovery_information_needed = true;
                self.report_cleanup_failures(&cleanup_report);
            }
            if let Err(error) =
                save_training_manifest(&run.manifest_path, &run, status, None, &cleanup_report)
            {
                recovery_information_needed = true;
                if self.user_action_error.is_none() {
                    self.user_action_error =
                        Some(format!("Training manifest save failed: {error}"));
                }
            }
        }
        let cancelled = matches!(result, TrainingRunResult::Cancelled);
        self.run.finish(result);
        if cancelled {
            self.run.worker_exited();
        }
        if !recovery_information_needed {
            self.clear_active_run_settings();
        }
    }
}
