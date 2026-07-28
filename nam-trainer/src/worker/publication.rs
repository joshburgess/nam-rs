use std::path::{Path, PathBuf};

use super::WorkerMessage;

pub(super) struct PublicationContext {
    pub(super) run: crate::run_manifest::ActiveTrainingRun,
    pub(super) destination: PathBuf,
}

pub(super) fn prepare_worker_message(
    message: WorkerMessage,
    publication: Option<&PublicationContext>,
) -> WorkerMessage {
    let WorkerMessage::FileCompleted {
        file,
        validation_esr,
        model_path,
    } = message
    else {
        return message;
    };
    let Some(publication) = publication else {
        return WorkerMessage::FilePublicationFailed {
            file,
            error: "training artifact publication context is unavailable".to_string(),
        };
    };
    match crate::run_manifest::publish_artifact_bundle(
        Path::new(&model_path),
        &publication.run,
        &publication.destination,
    ) {
        Ok((model_path, log_path, manifest_path)) => WorkerMessage::FilePublished {
            file,
            validation_esr,
            model_path,
            log_path,
            manifest_path,
        },
        Err(error) => WorkerMessage::FilePublicationFailed {
            file,
            error: error.to_string(),
        },
    }
}
