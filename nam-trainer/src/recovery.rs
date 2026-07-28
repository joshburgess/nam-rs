use crate::app::{TrainerApp, TrainingRunResult};
use crate::run_manifest::ActiveTrainingRun;

impl TrainerApp {
    pub(crate) fn restore_interrupted_run(&mut self) {
        let Some(run_id) = self.settings.active_run_id.clone() else {
            return;
        };
        let Some(log_path) = self.settings.active_run_log_path.clone() else {
            return;
        };
        let Some(manifest_path) = self.settings.active_run_manifest_path.clone() else {
            return;
        };

        if !log_path.exists() && !manifest_path.exists() {
            self.clear_active_run_settings();
            return;
        }

        let staging_dir = self
            .settings
            .active_run_staging_dir
            .clone()
            .unwrap_or_else(|| {
                manifest_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(format!(".nam-trainer-{run_id}-staging"))
            });
        let reserved_paths = self.settings.active_run_reserved_paths.clone();
        self.run.data_mut().artifacts = Some(
            ActiveTrainingRun::from_existing(run_id, log_path.clone(), manifest_path.clone())
                .with_recovery_inventory(staging_dir.clone(), reserved_paths),
        );
        self.run.finish(TrainingRunResult::Error(
            "Previous training run did not finish. Check the recovered log and output directory."
                .into(),
        ));
        self.push_training_log(format!(
            "Recovered interrupted run log: {}",
            log_path.display()
        ));
        self.push_training_log(format!(
            "Recovered interrupted run manifest: {}",
            manifest_path.display()
        ));
        self.push_training_log(format!(
            "Recovered staging directory: {}. Reset this error to clean recovered staging and owned reservations safely.",
            staging_dir.display()
        ));
    }
}
