use crate::app::HideConsoleExt;
use crate::background::{ChildProcess, ProcessRunner};

pub(crate) const MANAGED_MINIFORGE_MARKER: &str = ".nam-trainer-managed";
pub(crate) const MINIFORGE_VERSION: &str = "26.3.2-2";

pub(crate) fn managed_miniforge_dir() -> Option<std::path::PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.home_dir().join("miniforge3"))
}

pub(crate) fn remove_managed_miniforge(path: &std::path::Path) -> std::io::Result<()> {
    let marker = path.join(MANAGED_MINIFORGE_MARKER);
    if !marker.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "refusing to remove {} because it is not marked as managed by NAM Trainer",
                path.display()
            ),
        ));
    }
    std::fs::remove_dir_all(path)
}

pub(crate) fn default_python_name() -> std::path::PathBuf {
    if cfg!(target_os = "windows") {
        "python".into()
    } else {
        "python3".into()
    }
}

pub(crate) struct MiniforgeInstaller {
    pub(crate) url: &'static str,
    pub(crate) extension: &'static str,
    pub(crate) sha256: &'static str,
}

pub(crate) fn miniforge_installer_info() -> Option<MiniforgeInstaller> {
    if cfg!(target_os = "macos") {
        Some(if cfg!(target_arch = "aarch64") {
            MiniforgeInstaller {
                url: "https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-MacOSX-arm64.sh",
                extension: ".sh",
                sha256: "2657d94152343cff7c06159ac9fc09624d7879fa9575c5a0a324c571c4df0ade",
            }
        } else {
            MiniforgeInstaller {
                url: "https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-MacOSX-x86_64.sh",
                extension: ".sh",
                sha256: "a755192103de19bb2782685ac78820c2e00702e5f33e6e4f0a3bf3c214f45d69",
            }
        })
    } else if cfg!(target_os = "linux") {
        Some(if cfg!(target_arch = "aarch64") {
            MiniforgeInstaller {
                url: "https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-Linux-aarch64.sh",
                extension: ".sh",
                sha256: "f4096a92482b30f04534cddb63d8bc929118318deffac71d90fb89dc52359d22",
            }
        } else {
            MiniforgeInstaller {
                url: "https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-Linux-x86_64.sh",
                extension: ".sh",
                sha256: "42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94",
            }
        })
    } else if cfg!(target_os = "windows") {
        Some(MiniforgeInstaller {
            url: "https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-Windows-x86_64.exe",
            extension: ".exe",
            sha256: "088884aafcbf2e3355671d4e9b227b0d1cfb278e3bbe74ba2ad213c553874d70",
        })
    } else {
        None
    }
}

pub(crate) trait Downloader: Send + Sync {
    fn start(
        &self,
        url: &str,
        destination: &std::path::Path,
        runner: &dyn ProcessRunner,
    ) -> std::io::Result<Box<dyn ChildProcess>>;
}

pub(crate) struct CurlDownloader;

impl Downloader for CurlDownloader {
    fn start(
        &self,
        url: &str,
        destination: &std::path::Path,
        runner: &dyn ProcessRunner,
    ) -> std::io::Result<Box<dyn ChildProcess>> {
        let mut command = std::process::Command::new("curl");
        command
            .args(["-fSL", "-o"])
            .arg(destination)
            .arg(url)
            .stderr(std::process::Stdio::piped())
            .hide_console();
        runner.spawn(&mut command)
    }
}

pub(crate) fn download_file(
    url: &str,
    destination: &std::path::Path,
    downloader: &dyn Downloader,
    runner: &dyn ProcessRunner,
) -> std::io::Result<Box<dyn ChildProcess>> {
    downloader.start(url, destination, runner)
}

pub(crate) fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ChecksumError {
    #[error("could not calculate installer checksum: {0}")]
    Io(#[from] std::io::Error),
    #[error("installer checksum mismatch; expected {expected}, received {actual}")]
    Mismatch { expected: String, actual: String },
}

pub(crate) fn verify_sha256(path: &std::path::Path, expected: &str) -> Result<(), ChecksumError> {
    let actual = sha256_file(path)?;
    if actual == expected {
        Ok(())
    } else {
        Err(ChecksumError::Mismatch {
            expected: expected.to_string(),
            actual,
        })
    }
}

pub(crate) fn run_miniforge_installer(
    installer_path: &std::path::Path,
    install_dir: &std::path::Path,
    runner: &dyn ProcessRunner,
) -> std::io::Result<Box<dyn ChildProcess>> {
    if cfg!(target_os = "windows") {
        let destination_arg = format!("/D={}", install_dir.display());
        let mut command = std::process::Command::new(installer_path);
        command
            .args([
                "/S",
                "/InstallationType=JustMe",
                "/RegisterPython=0",
                "/AddToPath=0",
                &destination_arg,
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .hide_console();
        runner.spawn(&mut command)
    } else {
        let mut command = std::process::Command::new("bash");
        command
            .arg(installer_path)
            .args(["-b", "-u", "-p"])
            .arg(install_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .hide_console();
        runner.spawn(&mut command)
    }
}

pub(crate) fn miniforge_python_path(install_dir: &std::path::Path) -> std::path::PathBuf {
    if cfg!(target_os = "windows") {
        install_dir.join("python.exe")
    } else {
        install_dir.join("bin").join("python")
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::io;
    use std::process::Command;
    use std::sync::Mutex;

    use super::{download_file, verify_sha256, CurlDownloader, Downloader};
    use crate::background::{
        CancellationToken, ChildProcess, CommandOutput, ProcessRunner, SystemProcessRunner,
    };

    struct FailingDownloader;

    impl Downloader for FailingDownloader {
        fn start(
            &self,
            _url: &str,
            _destination: &std::path::Path,
            _runner: &dyn ProcessRunner,
        ) -> io::Result<Box<dyn ChildProcess>> {
            Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "injected download failure",
            ))
        }
    }

    struct InspectingRunner {
        command: Mutex<Option<(OsString, Vec<OsString>)>>,
    }

    impl ProcessRunner for InspectingRunner {
        fn output(
            &self,
            _command: &mut Command,
            _cancel: &CancellationToken,
        ) -> io::Result<Option<CommandOutput>> {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "output is not used",
            ))
        }

        fn spawn(&self, command: &mut Command) -> io::Result<Box<dyn ChildProcess>> {
            let captured = (
                command.get_program().to_os_string(),
                command.get_args().map(OsString::from).collect(),
            );
            *self
                .command
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(captured);
            Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "stop before launching curl",
            ))
        }
    }

    #[test]
    fn downloader_failure_is_injectable_without_starting_a_process() {
        let result = download_file(
            "https://example.invalid/installer",
            std::path::Path::new("installer"),
            &FailingDownloader,
            &SystemProcessRunner,
        );

        let error = match result {
            Ok(_) => panic!("injected downloader failure should be returned"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
    }

    #[test]
    fn curl_downloader_constructs_expected_command() {
        let runner = InspectingRunner {
            command: Mutex::new(None),
        };
        let _ = download_file(
            "https://example.invalid/installer",
            std::path::Path::new("installer.tmp"),
            &CurlDownloader,
            &runner,
        );

        let command = runner
            .command
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
            .unwrap();
        assert_eq!(command.0, OsString::from("curl"));
        assert_eq!(
            command.1,
            vec![
                OsString::from("-fSL"),
                OsString::from("-o"),
                OsString::from("installer.tmp"),
                OsString::from("https://example.invalid/installer"),
            ]
        );
    }

    #[test]
    fn installer_hash_mismatch_is_structured() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), b"not the installer").unwrap();

        let error = verify_sha256(file.path(), "0000").unwrap_err();

        assert!(error.to_string().contains("checksum mismatch"));
        assert!(error.to_string().contains("0000"));
    }
}
