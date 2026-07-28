use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::app::{CudaInstall, HideConsoleExt, InstallMessage};
use crate::background::{
    self, BackgroundOperation, BackgroundSender, CancellationToken, ProcessRunner,
    SystemProcessRunner,
};
use crate::environment::{
    download_file, miniforge_installer_info, miniforge_python_path, remove_managed_miniforge,
    run_miniforge_installer, verify_sha256, ChecksumError, CurlDownloader,
    MANAGED_MINIFORGE_MARKER, MINIFORGE_VERSION,
};

const INSTALL_LOG_CAPACITY: usize = 256;
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);

pub(crate) type InstallOperation = BackgroundOperation<InstallMessage>;

pub(crate) fn spawn_nam_install(
    python: PathBuf,
    cuda_install: Option<CudaInstall>,
) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
        let runner = SystemProcessRunner;
        if let Some(cuda) = cuda_install {
            log(
                &tx,
                format!(
                    "NVIDIA GPU detected ({}). Installing PyTorch with CUDA {} first...",
                    cuda.gpu_names.join(", "),
                    cuda.cuda_version,
                ),
            );
            let torch_args = [
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                cuda.wheel_index.as_str(),
            ];
            if !run_pip(&python, &torch_args, &cancel, &tx, &runner) {
                if !cancel.is_cancelled() {
                    log(&tx, "PyTorch CUDA install failed. Aborting.");
                    done(&tx, false);
                }
                return;
            }
        }

        log(&tx, "Installing or upgrading neural-amp-modeler...");
        let success = run_pip(
            &python,
            &["-m", "pip", "install", "--upgrade", "neural-amp-modeler"],
            &cancel,
            &tx,
            &runner,
        );
        if !cancel.is_cancelled() {
            done(&tx, success);
        }
    })
}

pub(crate) fn spawn_cuda_install(python: PathBuf, cuda: CudaInstall) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
        let runner = SystemProcessRunner;
        let args = [
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "torch",
            "--index-url",
            cuda.wheel_index.as_str(),
        ];
        let success = run_pip(&python, &args, &cancel, &tx, &runner);
        if !cancel.is_cancelled() {
            done(&tx, success);
        }
    })
}

pub(crate) fn spawn_miniforge_install(install_dir: PathBuf) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
        let runner = SystemProcessRunner;
        let downloader = CurlDownloader;
        let Some(installer) = miniforge_installer_info() else {
            log(
                &tx,
                "Automatic Python install is not supported on this platform.",
            );
            done(&tx, false);
            return;
        };

        let installer_file = match tempfile::Builder::new()
            .prefix("nam-trainer-miniforge-")
            .suffix(installer.extension)
            .tempfile()
        {
            Ok(file) => file,
            Err(error) => {
                log(
                    &tx,
                    format!("Could not create temporary installer: {error}"),
                );
                done(&tx, false);
                return;
            }
        };
        let installer_path = installer_file.into_temp_path();
        log(
            &tx,
            format!(
                "Downloading pinned Miniforge {MINIFORGE_VERSION} from {}...",
                installer.url
            ),
        );

        let download = match download_file(installer.url, &installer_path, &downloader, &runner) {
            Ok(child) => child,
            Err(error) => {
                log(&tx, format!("Failed to start download: {error}"));
                done(&tx, false);
                return;
            }
        };
        if !run_child(download, &cancel, &tx) {
            if !cancel.is_cancelled() {
                log(&tx, "Download failed.");
                done(&tx, false);
            }
            return;
        }
        log(&tx, "Download complete.");

        match verify_sha256(&installer_path, installer.sha256) {
            Ok(()) => {
                log(&tx, "Installer checksum verified.");
            }
            Err(ChecksumError::Mismatch { expected, actual }) => {
                log(
                    &tx,
                    format!(
                        "Installer checksum mismatch. Expected {}, received {actual}.",
                        expected
                    ),
                );
                done(&tx, false);
                return;
            }
            Err(ChecksumError::Io(error)) => {
                log(&tx, format!("Could not verify installer checksum: {error}"));
                done(&tx, false);
                return;
            }
        }

        log(&tx, format!("Installing to {}...", install_dir.display()));
        let installation = match run_miniforge_installer(&installer_path, &install_dir, &runner) {
            Ok(child) => child,
            Err(error) => {
                log(&tx, format!("Failed to run installer: {error}"));
                done(&tx, false);
                return;
            }
        };
        if !run_child(installation, &cancel, &tx) {
            if !cancel.is_cancelled() {
                log(&tx, "Miniforge installation failed.");
                done(&tx, false);
            }
            return;
        }

        let marker_path = install_dir.join(MANAGED_MINIFORGE_MARKER);
        if let Err(error) = crate::persistence::atomic_write(
            &marker_path,
            format!("miniforge_version={MINIFORGE_VERSION}\n").as_bytes(),
        ) {
            log(
                &tx,
                format!("Could not record managed installation marker: {error}"),
            );
            done(&tx, false);
            return;
        }

        let python_path = miniforge_python_path(&install_dir);
        log(
            &tx,
            format!("Python installed at {}", python_path.display()),
        );
        if tx.send_critical(InstallMessage::PythonInstalled { python_path }) {
            done(&tx, true);
        }
    })
}

pub(crate) fn spawn_nam_uninstall(python: PathBuf) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
        let runner = SystemProcessRunner;
        let success = run_pip(
            &python,
            &["-m", "pip", "uninstall", "-y", "neural-amp-modeler"],
            &cancel,
            &tx,
            &runner,
        );
        if cancel.is_cancelled() {
            return;
        }
        if success {
            log(&tx, "NAM uninstalled successfully.");
        }
        done(&tx, success);
    })
}

pub(crate) fn spawn_miniforge_uninstall(miniforge_dir: Option<PathBuf>) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
        let Some(miniforge_dir) = miniforge_dir else {
            log(&tx, "Could not determine the managed Miniforge path.");
            done(&tx, false);
            return;
        };
        if cancel.is_cancelled() {
            return;
        }
        log(&tx, format!("Deleting {}...", miniforge_dir.display()));
        match remove_managed_miniforge(&miniforge_dir) {
            Ok(()) if !cancel.is_cancelled() => {
                log(&tx, "Miniforge removed successfully.");
                done(&tx, true);
            }
            Ok(()) => {}
            Err(error) => {
                log(&tx, format!("Failed to remove: {error}"));
                done(&tx, false);
            }
        }
    })
}

fn run_pip(
    python: &Path,
    args: &[&str],
    cancel: &CancellationToken,
    tx: &BackgroundSender<InstallMessage>,
    runner: &dyn ProcessRunner,
) -> bool {
    let mut command = std::process::Command::new(python);
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .hide_console();
    let child = runner.spawn(&mut command);
    match child {
        Ok(child) => run_child(child, cancel, tx),
        Err(error) => {
            log(tx, format!("Failed to run pip: {error}"));
            false
        }
    }
}

fn run_child(
    mut child: Child,
    cancel: &CancellationToken,
    tx: &BackgroundSender<InstallMessage>,
) -> bool {
    let stdout_thread = spawn_log_reader(child.stdout.take(), tx.clone());
    let stderr_thread = spawn_log_reader(child.stderr.take(), tx.clone());

    let success = loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            break false;
        }
        match child.try_wait() {
            Ok(Some(status)) => break status.success(),
            Ok(None) => std::thread::sleep(CHILD_POLL_INTERVAL),
            Err(error) => {
                log(tx, format!("Could not monitor subprocess: {error}"));
                let _ = child.kill();
                let _ = child.wait();
                break false;
            }
        }
    };

    join_reader(stdout_thread);
    join_reader(stderr_thread);
    success
}

fn spawn_log_reader<R>(
    stream: Option<R>,
    tx: BackgroundSender<InstallMessage>,
) -> Option<JoinHandle<()>>
where
    R: Read + Send + 'static,
{
    stream.map(|stream| {
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                if !tx.send_progress(InstallMessage::Log(line)) {
                    break;
                }
            }
        })
    })
}

fn join_reader(thread: Option<JoinHandle<()>>) {
    if let Some(thread) = thread {
        let _ = thread.join();
    }
}

fn log(tx: &BackgroundSender<InstallMessage>, message: impl Into<String>) {
    let _ = tx.send_progress(InstallMessage::Log(message.into()));
}

fn done(tx: &BackgroundSender<InstallMessage>, success: bool) {
    let _ = tx.send_critical(InstallMessage::Done { success });
}
