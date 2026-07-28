use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::app::{
    CudaInstall, DetectionResult, EnvironmentReport, HideConsoleExt, PythonStatus, TrainingDevice,
    NAM_MIN_PYTHON,
};
use crate::background::{
    self, BackgroundOperation, CancellationToken, ProcessRunner, SystemProcessRunner,
};

#[derive(Clone)]
pub(crate) struct PythonEntry {
    pub(crate) label: String,
    pub(crate) path: PathBuf,
}

fn discover_pythons(cancel: &CancellationToken) -> Vec<PythonEntry> {
    discover_pythons_with_runner(cancel, &SystemProcessRunner)
}

fn discover_pythons_with_runner(
    cancel: &CancellationToken,
    runner: &dyn ProcessRunner,
) -> Vec<PythonEntry> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();

    #[cfg(not(target_os = "windows"))]
    let (which_command, candidates) = ("which", ["python3", "python"]);
    #[cfg(target_os = "windows")]
    let (which_command, candidates) = ("where", ["python", "python3"]);

    for name in candidates {
        if cancel.is_cancelled() {
            break;
        }
        let mut command = Command::new(which_command);
        command.arg(name).hide_console();
        if let Ok(Some(output)) = runner.output(&mut command, cancel) {
            if !output.success
                || output.stdout_truncated_bytes > 0
                || output.stderr_truncated_bytes > 0
            {
                continue;
            }
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let path = PathBuf::from(line.trim());
                if path.as_os_str().is_empty() {
                    continue;
                }
                let resolved = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if !seen.insert(resolved) {
                    continue;
                }
                let mut command = Command::new(&path);
                command.args(["--version"]).hide_console();
                let version = runner
                    .output(&mut command, cancel)
                    .ok()
                    .flatten()
                    .map(|version| String::from_utf8_lossy(&version.stdout).trim().to_string())
                    .unwrap_or_default();
                found.push(PythonEntry {
                    label: if version.is_empty() {
                        name.to_string()
                    } else {
                        version
                    },
                    path,
                });
            }
        }
    }

    if let Some(home) = home_dir() {
        for base in ["miniconda3", "anaconda3", "miniforge3", ".conda"] {
            if cancel.is_cancelled() {
                break;
            }
            let environments = home.join(base).join("envs");
            if let Ok(entries) = std::fs::read_dir(environments) {
                for entry in entries.flatten() {
                    let python = conda_python_path(&entry.path());
                    if !python.exists() {
                        continue;
                    }
                    let resolved =
                        std::fs::canonicalize(&python).unwrap_or_else(|_| python.clone());
                    if seen.insert(resolved) {
                        found.push(PythonEntry {
                            label: format!("conda: {}", entry.file_name().to_string_lossy()),
                            path: python,
                        });
                    }
                }
            }
        }
    }
    found
}

pub(crate) fn spawn_python_discovery() -> BackgroundOperation<Vec<PythonEntry>> {
    background::spawn(1, |cancel, tx| {
        let discovered = discover_pythons(&cancel);
        if !cancel.is_cancelled() {
            let _ = tx.send_critical(discovered);
        }
    })
}

pub(crate) fn spawn_environment_detection(python: PathBuf) -> BackgroundOperation<DetectionResult> {
    background::spawn(1, move |cancel, tx| {
        let result = detect_environment(&python, &cancel, &SystemProcessRunner);
        if !cancel.is_cancelled() {
            let _ = tx.send_critical(result);
        }
    })
}

fn detect_environment(
    python: &Path,
    cancel: &CancellationToken,
    runner: &dyn ProcessRunner,
) -> DetectionResult {
    let script = include_str!("../python/detect_environment.py");
    let mut command = Command::new(python);
    command.args(["-c", script]).hide_console();
    let output = runner.output(&mut command, cancel);
    match output {
        Ok(Some(output))
            if output.stdout_truncated_bytes > 0 || output.stderr_truncated_bytes > 0 =>
        {
            DetectionResult {
                status: PythonStatus::Error(
                    "Python environment output exceeded the safety limit".into(),
                ),
                cuda_install: None,
            }
        }
        Ok(Some(output)) if output.success => {
            parse_detection_output(String::from_utf8_lossy(&output.stdout).trim())
        }
        Ok(Some(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stderr_lower = stderr.to_lowercase();
            let status = if stderr_lower.contains("was not found")
                || stderr_lower.contains("not recognized")
                || stderr_lower.contains("not found")
            {
                PythonStatus::NotFound
            } else {
                PythonStatus::Error(format!(
                    "Python error: {}",
                    stderr.lines().next().unwrap_or("unknown")
                ))
            };
            DetectionResult {
                status,
                cuda_install: None,
            }
        }
        Ok(None) | Err(_) => DetectionResult {
            status: PythonStatus::NotFound,
            cuda_install: None,
        },
    }
}

fn parse_detection_output(output: &str) -> DetectionResult {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(output) else {
        return DetectionResult {
            status: PythonStatus::Error("Unexpected Python output".into()),
            cuda_install: None,
        };
    };
    let version = value
        .get("version")
        .and_then(|version| version.as_str())
        .unwrap_or("unknown")
        .to_string();
    let cuda_install = parse_cuda_install(&value);
    let version_ok = parse_version_tuple(&version).is_some_and(|version| version >= NAM_MIN_PYTHON);
    if !version_ok {
        return DetectionResult {
            status: PythonStatus::VersionTooOld { version },
            cuda_install,
        };
    }
    if !value
        .get("nam")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return DetectionResult {
            status: PythonStatus::Error("NAM not installed".into()),
            cuda_install,
        };
    }

    let devices = value
        .get("devices")
        .and_then(serde_json::Value::as_array)
        .map(|devices| {
            devices
                .iter()
                .filter_map(|device| {
                    Some(TrainingDevice {
                        id: device.get("id")?.as_str()?.into(),
                        name: device.get("name")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .filter(|devices: &Vec<_>| !devices.is_empty())
        .unwrap_or_else(|| {
            vec![TrainingDevice {
                id: "cpu".into(),
                name: "CPU".into(),
            }]
        });
    let warnings = value
        .get("warnings")
        .and_then(serde_json::Value::as_array)
        .map(|warnings| {
            warnings
                .iter()
                .filter_map(|warning| warning.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    DetectionResult {
        status: PythonStatus::Ok {
            version,
            devices,
            warnings,
            report: parse_environment_report(&value),
        },
        cuda_install,
    }
}

fn parse_version_tuple(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.split('.');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

fn parse_cuda_install(value: &serde_json::Value) -> Option<CudaInstall> {
    let cuda = value.get("cuda_install")?;
    if cuda.is_null() {
        return None;
    }
    Some(CudaInstall {
        cuda_version: cuda.get("cuda_version")?.as_str()?.to_string(),
        wheel_index: cuda.get("wheel_index")?.as_str()?.to_string(),
        gpu_names: cuda
            .get("gpu_names")
            .and_then(serde_json::Value::as_array)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|name| name.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

pub(crate) fn parse_environment_report(value: &serde_json::Value) -> EnvironmentReport {
    EnvironmentReport {
        nam_version: value
            .get("nam_version")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        torch_version: value
            .get("torch_version")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        packed_full_config_supported: value
            .get("packed_full_config_supported")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    }
}

fn conda_python_path(environment: &Path) -> PathBuf {
    #[cfg(not(target_os = "windows"))]
    {
        environment.join("bin").join("python")
    }
    #[cfg(target_os = "windows")]
    {
        environment.join("python.exe")
    }
}

pub(crate) fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::path::Path;
    use std::process::{Child, Command};

    use super::detect_environment;
    use crate::app::PythonStatus;
    use crate::background::{CancellationToken, CommandOutput, ProcessRunner};

    struct FakeProcessRunner {
        output: io::Result<Option<CommandOutput>>,
    }

    impl ProcessRunner for FakeProcessRunner {
        fn output(
            &self,
            _command: &mut Command,
            _cancel: &CancellationToken,
        ) -> io::Result<Option<CommandOutput>> {
            match &self.output {
                Ok(Some(output)) => Ok(Some(CommandOutput {
                    success: output.success,
                    stdout: output.stdout.clone(),
                    stderr: output.stderr.clone(),
                    stdout_truncated_bytes: output.stdout_truncated_bytes,
                    stderr_truncated_bytes: output.stderr_truncated_bytes,
                })),
                Ok(None) => Ok(None),
                Err(error) => Err(io::Error::new(error.kind(), error.to_string())),
            }
        }

        fn spawn(&self, _command: &mut Command) -> io::Result<Child> {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "fake runner does not spawn",
            ))
        }
    }

    #[test]
    fn process_runner_failure_is_reported_without_spawning_python() {
        let runner = FakeProcessRunner {
            output: Err(io::Error::new(io::ErrorKind::NotFound, "missing")),
        };

        let result = detect_environment(Path::new("python"), &CancellationToken::new(), &runner);

        assert!(matches!(result.status, PythonStatus::NotFound));
    }

    #[test]
    fn truncated_environment_output_is_rejected() {
        let runner = FakeProcessRunner {
            output: Ok(Some(CommandOutput {
                success: true,
                stdout: br#"{"version":"3.11"}"#.to_vec(),
                stderr: Vec::new(),
                stdout_truncated_bytes: 1,
                stderr_truncated_bytes: 0,
            })),
        };

        let result = detect_environment(Path::new("python"), &CancellationToken::new(), &runner);

        assert!(matches!(
            result.status,
            PythonStatus::Error(ref message) if message.contains("safety limit")
        ));
    }
}
