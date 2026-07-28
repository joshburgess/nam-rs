#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationSeverity {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub message: String,
}

impl ValidationIssue {
    fn error(message: impl Into<String>) -> Self {
        Self {
            severity: ValidationSeverity::Error,
            message: message.into(),
        }
    }

    fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: ValidationSeverity::Warning,
            message: message.into(),
        }
    }

    pub fn is_error(&self) -> bool {
        self.severity == ValidationSeverity::Error
    }
}

pub fn validate_audio_files(
    input_path: &std::path::Path,
    output_paths: &[std::path::PathBuf],
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    let input_reader = match hound::WavReader::open(input_path) {
        Ok(reader) => reader,
        Err(error) => {
            issues.push(ValidationIssue::error(format!(
                "Cannot read input file: {error}"
            )));
            return issues;
        }
    };
    let input_spec = input_reader.spec();
    let input_duration = duration_seconds(&input_reader);

    if input_duration < 1.0 {
        issues.push(ValidationIssue::warning(format!(
            "Input file is very short ({input_duration:.1}s). Training needs at least a few seconds of audio."
        )));
    }

    for output_path in output_paths {
        let basename = output_path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| output_path.display().to_string());

        let output_reader = match hound::WavReader::open(output_path) {
            Ok(reader) => reader,
            Err(error) => {
                issues.push(ValidationIssue::error(format!(
                    "{basename}: cannot read file: {error}"
                )));
                continue;
            }
        };
        let output_spec = output_reader.spec();
        let output_duration = duration_seconds(&output_reader);

        if output_spec.sample_rate != input_spec.sample_rate {
            issues.push(ValidationIssue::error(format!(
                "{basename}: sample rate {}Hz does not match input ({}Hz)",
                output_spec.sample_rate, input_spec.sample_rate
            )));
        }

        if output_duration < 1.0 {
            issues.push(ValidationIssue::warning(format!(
                "{basename}: very short ({output_duration:.1}s)"
            )));
        }

        let ratio = if input_duration > 0.0 {
            output_duration / input_duration
        } else {
            1.0
        };
        if !(0.5..=2.0).contains(&ratio) {
            issues.push(ValidationIssue::warning(format!(
                "{basename}: duration ({output_duration:.1}s) differs significantly from input ({input_duration:.1}s)"
            )));
        }
    }

    issues
}

fn duration_seconds(reader: &hound::WavReader<std::io::BufReader<std::fs::File>>) -> f64 {
    let spec = reader.spec();
    let channels = f64::from(spec.channels.max(1));
    let sample_rate = f64::from(spec.sample_rate.max(1));
    f64::from(reader.len()) / channels / sample_rate
}
