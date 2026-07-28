use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingRunArtifacts {
    pub destination: PathBuf,
    pub output_paths: Vec<PathBuf>,
    pub output_model_basename: Option<String>,
    pub batch_name_template: Option<String>,
}

impl TrainingRunArtifacts {
    pub fn new<P: AsRef<Path>>(destination: impl Into<PathBuf>, output_paths: &[P]) -> Self {
        Self {
            destination: destination.into(),
            output_paths: output_paths
                .iter()
                .map(|path| path.as_ref().to_path_buf())
                .collect(),
            output_model_basename: None,
            batch_name_template: None,
        }
    }

    pub fn with_naming(
        mut self,
        output_model_basename: Option<String>,
        batch_name_template: Option<String>,
    ) -> Self {
        self.output_model_basename = output_model_basename;
        self.batch_name_template = batch_name_template;
        self
    }

    pub fn predicted_model_paths(&self) -> Vec<PathBuf> {
        self.output_paths
            .iter()
            .enumerate()
            .map(|(index, output_path)| {
                self.destination
                    .join(format!("{}.nam", self.model_basename(index, output_path)))
            })
            .collect()
    }

    pub fn model_basename(&self, index: usize, output_path: &Path) -> String {
        let stem = output_stem(output_path);
        if self.output_paths.len() == 1 {
            if let Some(name) = self
                .output_model_basename
                .as_deref()
                .map(str::trim)
                .filter(|name| !name.is_empty())
            {
                return sanitize_model_basename(name);
            }
        }

        let template = self
            .batch_name_template
            .as_deref()
            .map(str::trim)
            .filter(|template| !template.is_empty())
            .unwrap_or("{stem}");
        let rendered = template
            .replace("{stem}", &stem)
            .replace("{index}", &(index + 1).to_string());
        sanitize_model_basename(&rendered)
    }

    pub fn log_path_for_model(model_path: &Path) -> PathBuf {
        model_path.with_extension("training.log")
    }

    pub fn manifest_path_for_model(model_path: &Path) -> PathBuf {
        model_path.with_extension("training-manifest.json")
    }

    pub fn conflicting_existing_artifacts(&self) -> Vec<PathBuf> {
        self.predicted_model_paths()
            .into_iter()
            .flat_map(|model_path| {
                [
                    model_path.clone(),
                    Self::log_path_for_model(&model_path),
                    Self::manifest_path_for_model(&model_path),
                ]
            })
            .filter(|artifact_path| artifact_path.exists())
            .collect()
    }

    pub fn duplicate_artifact_paths(&self) -> Vec<PathBuf> {
        let mut seen = HashSet::new();
        let mut duplicates = HashSet::new();
        for path in self
            .predicted_model_paths()
            .into_iter()
            .flat_map(|model_path| {
                [
                    model_path.clone(),
                    Self::log_path_for_model(&model_path),
                    Self::manifest_path_for_model(&model_path),
                ]
            })
        {
            let comparison_path = comparison_path(&path);
            if !seen.insert(comparison_path.clone()) {
                duplicates.insert(comparison_path);
            }
        }
        let mut duplicates: Vec<_> = duplicates.into_iter().collect();
        duplicates.sort();
        duplicates
    }
}

fn comparison_path(path: &Path) -> PathBuf {
    let resolved = path
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .and_then(|parent| path.file_name().map(|name| parent.join(name)))
        .unwrap_or_else(|| path.to_path_buf());
    if cfg!(any(target_os = "windows", target_os = "macos")) {
        PathBuf::from(resolved.to_string_lossy().to_lowercase())
    } else {
        resolved
    }
}

fn output_stem(output_path: &Path) -> String {
    output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("model")
        .to_string()
}

pub fn sanitize_model_basename(input: &str) -> String {
    let sanitized: String = input
        .trim()
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            ch if ch.is_control() => '_',
            ch => ch,
        })
        .collect();
    let sanitized = sanitized.trim_matches(['.', ' ']).trim();
    if sanitized.is_empty() {
        "model".into()
    } else {
        sanitized.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{comparison_path, TrainingRunArtifacts};
    use proptest::prelude::*;

    #[test]
    fn duplicate_paths_detect_same_stem_in_different_directories() {
        let outputs = vec![
            "captures/clean/output.wav".to_string(),
            "captures/drive/output.wav".to_string(),
        ];
        let artifacts = TrainingRunArtifacts::new("models", &outputs);

        let duplicates = artifacts.duplicate_artifact_paths();

        assert_eq!(duplicates.len(), 3);
        assert!(duplicates.iter().any(|path| path.ends_with("output.nam")));
        assert!(duplicates
            .iter()
            .any(|path| path.ends_with("output.training.log")));
        assert!(duplicates
            .iter()
            .any(|path| path.ends_with("output.training-manifest.json")));
    }

    #[test]
    fn indexed_template_avoids_duplicate_paths() {
        let outputs = vec![
            "captures/clean/output.wav".to_string(),
            "captures/drive/output.wav".to_string(),
        ];
        let artifacts = TrainingRunArtifacts::new("models", &outputs)
            .with_naming(None, Some("{index}-{stem}".into()));

        assert!(artifacts.duplicate_artifact_paths().is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn comparison_resolves_symlinked_destination_aliases() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let destination = root.join("destination");
        let alias = root.join("alias");
        std::fs::create_dir_all(&destination).unwrap();

        std::os::unix::fs::symlink(&destination, &alias).unwrap();

        assert_eq!(
            comparison_path(&destination.join("model.nam")),
            comparison_path(&alias.join("model.nam"))
        );
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    fn comparison_is_case_insensitive_on_common_case_insensitive_platforms() {
        assert_eq!(
            comparison_path(std::path::Path::new("Models/Amp.nam")),
            comparison_path(std::path::Path::new("models/amp.NAM"))
        );
    }

    proptest! {
        #[test]
        fn sanitized_model_names_are_nonempty_single_path_components(input in ".*") {
            let sanitized = super::sanitize_model_basename(&input);
            prop_assert!(!sanitized.is_empty());
            let contains_forbidden_character = sanitized.chars().any(|character| {
                character.is_control() || matches!(character, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
            });
            prop_assert!(!contains_forbidden_character);
            prop_assert!(!sanitized.starts_with(&['.', ' '][..]));
            prop_assert!(!sanitized.ends_with(&['.', ' '][..]));
        }

        #[test]
        fn indexed_batch_template_is_unique_for_arbitrary_stems(
            stems in proptest::collection::vec("[^/\\\\]{0,30}", 1..32)
        ) {
            let outputs = stems
                .iter()
                .enumerate()
                .map(|(index, stem)| format!("capture-{index}/{stem}.wav"))
                .collect::<Vec<_>>();
            let artifacts = TrainingRunArtifacts::new("models", &outputs)
                .with_naming(None, Some("{index}-{stem}".into()));
            prop_assert!(artifacts.duplicate_artifact_paths().is_empty());
        }
    }
}
