#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainingErrorKind {
    Dependency,
    DataValidation,
    Device,
    UserCancel,
    SubprocessCrash,
    Training,
}

impl From<crate::worker::protocol::WorkerErrorKind> for TrainingErrorKind {
    fn from(kind: crate::worker::protocol::WorkerErrorKind) -> Self {
        match kind {
            crate::worker::protocol::WorkerErrorKind::Dependency => Self::Dependency,
            crate::worker::protocol::WorkerErrorKind::DataValidation => Self::DataValidation,
            crate::worker::protocol::WorkerErrorKind::Device => Self::Device,
            crate::worker::protocol::WorkerErrorKind::UserCancel => Self::UserCancel,
            crate::worker::protocol::WorkerErrorKind::Subprocess
            | crate::worker::protocol::WorkerErrorKind::Protocol => Self::SubprocessCrash,
            crate::worker::protocol::WorkerErrorKind::Training => Self::Training,
        }
    }
}

impl From<crate::worker::WorkerFailureKind> for TrainingErrorKind {
    fn from(kind: crate::worker::WorkerFailureKind) -> Self {
        match kind {
            crate::worker::WorkerFailureKind::Reported(kind) => kind.into(),
            crate::worker::WorkerFailureKind::MissingArtifact
            | crate::worker::WorkerFailureKind::UnsupportedPath => Self::DataValidation,
            crate::worker::WorkerFailureKind::Launch
            | crate::worker::WorkerFailureKind::Serialization
            | crate::worker::WorkerFailureKind::Subprocess
            | crate::worker::WorkerFailureKind::Protocol
            | crate::worker::WorkerFailureKind::ProtocolSequence => Self::SubprocessCrash,
        }
    }
}

impl TrainingErrorKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dependency => "Dependency error",
            Self::DataValidation => "Data validation failure",
            Self::Device => "Device error",
            Self::UserCancel => "User cancel",
            Self::SubprocessCrash => "Subprocess crash",
            Self::Training => "Training error",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingErrorDetails {
    pub kind: TrainingErrorKind,
    pub message: String,
}

impl TrainingErrorDetails {
    pub fn new(kind: TrainingErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn classified_message(&self) -> String {
        format!("{}: {}", self.kind.label(), self.message)
    }
}

pub fn classify_training_error(message: &str) -> TrainingErrorDetails {
    let lower = message.to_lowercase();
    let kind = if lower.contains("missing dependency")
        || lower.contains("nam not installed")
        || lower.contains("no module named")
    {
        TrainingErrorKind::Dependency
    } else if lower.contains("data checks failed")
        || lower.contains("fix metadata")
        || lower.contains("input")
        || lower.contains("output directory")
    {
        TrainingErrorKind::DataValidation
    } else if lower.contains("cuda")
        || lower.contains("cudnn")
        || lower.contains("mps")
        || lower.contains("out of memory")
    {
        TrainingErrorKind::Device
    } else if lower.contains("cancel") || lower.contains("keyboardinterrupt") {
        TrainingErrorKind::UserCancel
    } else if lower.contains("python exited")
        || lower.contains("worker process")
        || lower.contains("subprocess")
    {
        TrainingErrorKind::SubprocessCrash
    } else {
        TrainingErrorKind::Training
    };
    TrainingErrorDetails {
        kind,
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{classify_training_error, TrainingErrorKind};

    #[test]
    fn classifies_common_training_errors() {
        let dependency = classify_training_error("Missing dependency: torch");
        assert_eq!(dependency.kind, TrainingErrorKind::Dependency);
        assert!(dependency
            .classified_message()
            .starts_with("Dependency error:"));

        assert_eq!(
            classify_training_error("NAM data checks failed").kind,
            TrainingErrorKind::DataValidation
        );
        assert_eq!(
            classify_training_error("CUDA out of memory").kind,
            TrainingErrorKind::Device
        );
        assert_eq!(
            classify_training_error("KeyboardInterrupt").kind,
            TrainingErrorKind::UserCancel
        );
        assert_eq!(
            classify_training_error("Python exited with code 1").kind,
            TrainingErrorKind::SubprocessCrash
        );
    }
}
