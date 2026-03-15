use thiserror::Error;

#[derive(Error, Debug)]
pub enum NamError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Unsupported config version: {0}")]
    UnsupportedVersion(String),

    #[error("Unknown architecture: {0}")]
    UnknownArchitecture(String),

    #[error("Missing config field: {0}")]
    MissingField(String),

    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Weight count mismatch: expected {expected}, got {actual}")]
    WeightMismatch { expected: usize, actual: usize },

    #[error("Unknown activation: {0}")]
    UnknownActivation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_messages() {
        let e = NamError::UnsupportedVersion("1.0.0".into());
        assert!(format!("{}", e).contains("1.0.0"));

        let e = NamError::UnknownArchitecture("Transformer".into());
        assert!(format!("{}", e).contains("Transformer"));

        let e = NamError::MissingField("weights".into());
        assert!(format!("{}", e).contains("weights"));

        let e = NamError::WeightMismatch {
            expected: 100,
            actual: 50,
        };
        assert!(format!("{}", e).contains("100"));
        assert!(format!("{}", e).contains("50"));

        let e = NamError::UnknownActivation("Mish".into());
        assert!(format!("{}", e).contains("Mish"));
    }

    #[test]
    fn test_io_error_converts() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let nam_err: NamError = io_err.into();
        assert!(matches!(nam_err, NamError::Io(_)));
    }

    #[test]
    fn test_json_error_converts() {
        let json_result: Result<serde_json::Value, _> = serde_json::from_str("{bad");
        let json_err = json_result.unwrap_err();
        let nam_err: NamError = json_err.into();
        assert!(matches!(nam_err, NamError::Json(_)));
    }
}
