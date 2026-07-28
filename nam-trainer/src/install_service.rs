use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::app::{CudaInstall, HideConsoleExt, InstallMessage};
use crate::background::{
    self, BackgroundOperation, BackgroundSender, CancellationToken, ChildProcess, ProcessRunner,
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
    spawn_nam_install_with_runner(python, cuda_install, Arc::new(SystemProcessRunner))
}

fn spawn_nam_install_with_runner(
    python: PathBuf,
    cuda_install: Option<CudaInstall>,
    runner: Arc<dyn ProcessRunner>,
) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
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
            if !run_pip(&python, &torch_args, &cancel, &tx, runner.as_ref()) {
                if cancel.is_cancelled() {
                    return;
                }
                log(
                    &tx,
                    "PyTorch CUDA install failed. Falling back to the default package source.",
                );
            }
        }

        log(&tx, "Installing or upgrading neural-amp-modeler...");
        let success = run_pip(
            &python,
            &["-m", "pip", "install", "--upgrade", "neural-amp-modeler"],
            &cancel,
            &tx,
            runner.as_ref(),
        );
        if !cancel.is_cancelled() {
            done(&tx, success);
        }
    })
}

pub(crate) fn spawn_cuda_install(python: PathBuf, cuda: CudaInstall) -> InstallOperation {
    spawn_cuda_install_with_runner(python, cuda, Arc::new(SystemProcessRunner))
}

fn spawn_cuda_install_with_runner(
    python: PathBuf,
    cuda: CudaInstall,
    runner: Arc<dyn ProcessRunner>,
) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
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
        let success = run_pip(&python, &args, &cancel, &tx, runner.as_ref());
        if !cancel.is_cancelled() {
            done(&tx, success);
        }
    })
}

pub(crate) fn spawn_miniforge_install(install_dir: PathBuf) -> InstallOperation {
    spawn_miniforge_install_with_services(
        install_dir,
        Arc::new(CurlDownloader),
        Arc::new(SystemProcessRunner),
    )
}

fn spawn_miniforge_install_with_services(
    install_dir: PathBuf,
    downloader: Arc<dyn crate::environment::Downloader>,
    runner: Arc<dyn ProcessRunner>,
) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
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

        let download = match download_file(
            installer.url,
            &installer_path,
            downloader.as_ref(),
            runner.as_ref(),
        ) {
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
        let installation =
            match run_miniforge_installer(&installer_path, &install_dir, runner.as_ref()) {
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
    spawn_nam_uninstall_with_runner(python, Arc::new(SystemProcessRunner))
}

fn spawn_nam_uninstall_with_runner(
    python: PathBuf,
    runner: Arc<dyn ProcessRunner>,
) -> InstallOperation {
    background::spawn(INSTALL_LOG_CAPACITY, move |cancel, tx| {
        let success = run_pip(
            &python,
            &["-m", "pip", "uninstall", "-y", "neural-amp-modeler"],
            &cancel,
            &tx,
            runner.as_ref(),
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
    mut child: Box<dyn ChildProcess>,
    cancel: &CancellationToken,
    tx: &BackgroundSender<InstallMessage>,
) -> bool {
    let stdout_thread = spawn_log_reader(child.take_stdout(), tx.clone());
    let stderr_thread = spawn_log_reader(child.take_stderr(), tx.clone());

    let success = loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            break false;
        }
        match child.try_wait() {
            Ok(Some(status)) => break status.success,
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

#[cfg(test)]
mod tests {
    use super::{
        spawn_miniforge_install_with_services, spawn_miniforge_uninstall,
        spawn_nam_install_with_runner, InstallOperation,
    };
    use crate::app::{CudaInstall, InstallMessage};
    use crate::background::{
        CancellationToken, ChildExit, ChildProcess, CommandOutput, ProcessRunner,
    };
    use crate::environment::{Downloader, MANAGED_MINIFORGE_MARKER};
    use std::collections::VecDeque;
    use std::ffi::OsString;
    use std::io::{self, Cursor, Read};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    struct ChildPlan {
        exit: ChildExit,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        never_exits: bool,
        kills: Arc<AtomicUsize>,
    }

    impl ChildPlan {
        fn success() -> Self {
            Self {
                exit: ChildExit {
                    success: true,
                    code: Some(0),
                },
                stdout: Vec::new(),
                stderr: Vec::new(),
                never_exits: false,
                kills: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn failure(stderr: &str) -> Self {
            Self {
                exit: ChildExit {
                    success: false,
                    code: Some(1),
                },
                stdout: Vec::new(),
                stderr: stderr.as_bytes().to_vec(),
                never_exits: false,
                kills: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    struct FakeChild {
        plan: ChildPlan,
        stdout: Option<Vec<u8>>,
        stderr: Option<Vec<u8>>,
    }

    impl ChildProcess for FakeChild {
        fn take_stdout(&mut self) -> Option<Box<dyn Read + Send>> {
            self.stdout
                .take()
                .map(|bytes| Box::new(Cursor::new(bytes)) as Box<dyn Read + Send>)
        }

        fn take_stderr(&mut self) -> Option<Box<dyn Read + Send>> {
            self.stderr
                .take()
                .map(|bytes| Box::new(Cursor::new(bytes)) as Box<dyn Read + Send>)
        }

        fn try_wait(&mut self) -> io::Result<Option<ChildExit>> {
            if self.plan.never_exits {
                Ok(None)
            } else {
                Ok(Some(self.plan.exit))
            }
        }

        fn kill(&mut self) -> io::Result<()> {
            self.plan.kills.fetch_add(1, Ordering::Relaxed);
            self.plan.never_exits = false;
            Ok(())
        }

        fn wait(&mut self) -> io::Result<ChildExit> {
            Ok(self.plan.exit)
        }
    }

    struct FakeRunner {
        plans: Mutex<VecDeque<ChildPlan>>,
        commands: Mutex<Vec<(OsString, Vec<OsString>)>>,
    }

    impl FakeRunner {
        fn new(plans: impl IntoIterator<Item = ChildPlan>) -> Self {
            Self {
                plans: Mutex::new(plans.into_iter().collect()),
                commands: Mutex::new(Vec::new()),
            }
        }

        fn commands(&self) -> Vec<(OsString, Vec<OsString>)> {
            self.commands
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }
    }

    impl ProcessRunner for FakeRunner {
        fn output(
            &self,
            _command: &mut Command,
            _cancel: &CancellationToken,
        ) -> io::Result<Option<CommandOutput>> {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "output is not used by installer workflows",
            ))
        }

        fn spawn(&self, command: &mut Command) -> io::Result<Box<dyn ChildProcess>> {
            self.commands
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push((
                    command.get_program().to_owned(),
                    command.get_args().map(OsString::from).collect(),
                ));
            let plan = self
                .plans
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .pop_front()
                .ok_or_else(|| io::Error::other("no fake child plan remains"))?;
            let stdout = Some(plan.stdout.clone());
            let stderr = Some(plan.stderr.clone());
            Ok(Box::new(FakeChild {
                plan,
                stdout,
                stderr,
            }))
        }
    }

    struct WritingDownloader {
        contents: Vec<u8>,
    }

    impl Downloader for WritingDownloader {
        fn start(
            &self,
            _url: &str,
            destination: &Path,
            runner: &dyn ProcessRunner,
        ) -> io::Result<Box<dyn ChildProcess>> {
            std::fs::write(destination, &self.contents)?;
            runner.spawn(&mut Command::new("fake-download"))
        }
    }

    fn collect_messages(operation: &InstallOperation) -> Vec<InstallMessage> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut messages = Vec::new();
        loop {
            match operation.try_recv() {
                Ok(message) => messages.push(message),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return messages,
                Err(std::sync::mpsc::TryRecvError::Empty) if Instant::now() < deadline => {
                    std::thread::yield_now();
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    panic!("installer workflow did not finish before the test deadline");
                }
            }
        }
    }

    fn done_status(messages: &[InstallMessage]) -> Option<bool> {
        messages.iter().find_map(|message| match message {
            InstallMessage::Done { success } => Some(*success),
            _ => None,
        })
    }

    #[test]
    fn pip_failure_is_reported_by_the_complete_install_workflow() {
        let runner = Arc::new(FakeRunner::new([ChildPlan::failure("pip failed")]));
        let operation =
            spawn_nam_install_with_runner(PathBuf::from("python"), None, runner.clone());

        let messages = collect_messages(&operation);

        assert_eq!(done_status(&messages), Some(false));
        assert!(messages
            .iter()
            .any(|message| matches!(message, InstallMessage::Log(line) if line == "pip failed")));
        assert_eq!(runner.commands().len(), 1);
    }

    #[test]
    fn failed_cuda_install_falls_back_to_the_default_nam_install() {
        let runner = Arc::new(FakeRunner::new([
            ChildPlan::failure("cuda failed"),
            ChildPlan::success(),
        ]));
        let operation = spawn_nam_install_with_runner(
            PathBuf::from("python"),
            Some(CudaInstall {
                cuda_version: "12.8".into(),
                wheel_index: "https://example.invalid/cu128".into(),
                gpu_names: vec!["GPU".into()],
            }),
            runner.clone(),
        );

        let messages = collect_messages(&operation);

        assert_eq!(done_status(&messages), Some(true));
        assert!(messages.iter().any(
            |message| matches!(message, InstallMessage::Log(line) if line.contains("Falling back"))
        ));
        let commands = runner.commands();
        assert_eq!(commands.len(), 2);
        assert!(commands[0].1.iter().any(|argument| argument == "torch"));
        assert!(commands[1]
            .1
            .iter()
            .any(|argument| argument == "neural-amp-modeler"));
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn checksum_failure_stops_before_running_the_installer() {
        let runner = Arc::new(FakeRunner::new([ChildPlan::success()]));
        let downloader = Arc::new(WritingDownloader {
            contents: b"tampered installer".to_vec(),
        });
        let directory = tempfile::tempdir().unwrap();
        let operation = spawn_miniforge_install_with_services(
            directory.path().join("miniforge"),
            downloader,
            runner.clone(),
        );

        let messages = collect_messages(&operation);

        assert_eq!(done_status(&messages), Some(false));
        assert!(messages.iter().any(
            |message| matches!(message, InstallMessage::Log(line) if line.contains("checksum mismatch"))
        ));
        assert_eq!(runner.commands().len(), 1);
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn cancellation_during_download_forces_child_termination() {
        let kills = Arc::new(AtomicUsize::new(0));
        let runner = Arc::new(FakeRunner::new([ChildPlan {
            exit: ChildExit {
                success: false,
                code: None,
            },
            stdout: Vec::new(),
            stderr: Vec::new(),
            never_exits: true,
            kills: Arc::clone(&kills),
        }]));
        let directory = tempfile::tempdir().unwrap();
        let operation = spawn_miniforge_install_with_services(
            directory.path().join("miniforge"),
            Arc::new(WritingDownloader {
                contents: b"partial download".to_vec(),
            }),
            runner.clone(),
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        while runner.commands().is_empty() && Instant::now() < deadline {
            std::thread::yield_now();
        }

        operation.cancel();
        let messages = collect_messages(&operation);

        assert_eq!(kills.load(Ordering::Relaxed), 1);
        assert_eq!(done_status(&messages), None);
    }

    #[test]
    fn managed_miniforge_uninstall_reports_success_and_removes_the_directory() {
        let directory = tempfile::tempdir().unwrap();
        let managed = directory.path().join("managed");
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(managed.join(MANAGED_MINIFORGE_MARKER), "managed").unwrap();
        let operation = spawn_miniforge_uninstall(Some(managed.clone()));

        let messages = collect_messages(&operation);

        assert_eq!(done_status(&messages), Some(true));
        assert!(!managed.exists());
    }
}
