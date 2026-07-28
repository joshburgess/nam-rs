use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn atomic_write(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    atomic_write_with_hook(path, contents, |_| Ok(()))
}

fn atomic_write_with_hook(
    path: &Path,
    contents: &[u8],
    mut hook: impl FnMut(AtomicWriteStage) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("path has no parent directory: {}", path.display()),
        )
    })?;
    std::fs::create_dir_all(parent)?;

    let temp_path = unique_sibling_path(path, "tmp");
    let result = write_and_replace(path, &temp_path, contents, &mut hook);
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomicWriteStage {
    Write,
    Sync,
    Replace,
    ParentSync,
}

fn write_and_replace(
    path: &Path,
    temp_path: &Path,
    contents: &[u8],
    hook: &mut impl FnMut(AtomicWriteStage) -> std::io::Result<()>,
) -> std::io::Result<()> {
    hook(AtomicWriteStage::Write)?;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(temp_path)?;
    preserve_permissions(path, temp_path)?;
    file.write_all(contents)?;
    hook(AtomicWriteStage::Sync)?;
    file.sync_all()?;
    drop(file);

    hook(AtomicWriteStage::Replace)?;
    replace_file(temp_path, path)?;
    hook(AtomicWriteStage::ParentSync)?;
    sync_parent_directory(path)
}

pub(crate) fn atomic_promote(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::OpenOptions::new()
        .write(true)
        .open(source)?
        .sync_all()?;
    replace_file(source, destination)?;
    sync_parent_directory(destination)
}

#[cfg(unix)]
fn preserve_permissions(source: &Path, destination: &Path) -> std::io::Result<()> {
    if source.exists() {
        std::fs::set_permissions(destination, std::fs::metadata(source)?.permissions())?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn preserve_permissions(_source: &Path, _destination: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    if !destination.exists() {
        return std::fs::rename(source, destination);
    }

    let backup = unique_sibling_path(destination, "backup");
    replace_existing_with(source, destination, &backup, |from, to| {
        std::fs::rename(from, to)
    })
}

#[cfg(any(windows, test))]
fn replace_existing_with(
    source: &Path,
    destination: &Path,
    backup: &Path,
    mut rename: impl FnMut(&Path, &Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    rename(destination, backup)?;
    match rename(source, destination) {
        Ok(()) => {
            let _ = std::fs::remove_file(backup);
            Ok(())
        }
        Err(error) => match rename(backup, destination) {
            Ok(()) => Err(error),
            Err(rollback_error) => Err(std::io::Error::new(
                rollback_error.kind(),
                format!(
                    "replacement failed: {error}; restoring {} also failed: {rollback_error}",
                    destination.display()
                ),
            )),
        },
    }
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

fn unique_sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{file_name}.{}.{counter}.{suffix}",
        std::process::id()
    ))
}

#[cfg(test)]
mod tests {
    use super::{atomic_write, atomic_write_with_hook, replace_existing_with, AtomicWriteStage};

    #[test]
    fn atomic_write_creates_and_replaces_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.json");

        atomic_write(&path, b"first").unwrap();
        atomic_write(&path, b"second").unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn injected_failures_leave_no_stale_temporary_files() {
        for stage in [
            AtomicWriteStage::Write,
            AtomicWriteStage::Sync,
            AtomicWriteStage::Replace,
            AtomicWriteStage::ParentSync,
        ] {
            let (_temp, directory) = unique_test_dir("fault");
            std::fs::create_dir_all(&directory).unwrap();
            let path = directory.join("settings.json");
            std::fs::write(&path, b"original").unwrap();

            let error = atomic_write_with_hook(&path, b"replacement", |current| {
                if current == stage {
                    Err(std::io::Error::other("injected failure"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();

            assert_eq!(error.kind(), std::io::ErrorKind::Other);
            let expected = if stage == AtomicWriteStage::ParentSync {
                b"replacement".as_slice()
            } else {
                b"original".as_slice()
            };
            assert_eq!(std::fs::read(&path).unwrap(), expected);
            assert!(std::fs::read_dir(&directory).unwrap().all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")));
            std::fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn failed_replacement_restores_the_original_file() {
        let (_temp, directory) = unique_test_dir("rollback");
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("new.tmp");
        let destination = directory.join("settings.json");
        let backup = directory.join("settings.backup");
        std::fs::write(&source, b"replacement").unwrap();
        std::fs::write(&destination, b"original").unwrap();
        let mut calls = 0;

        let error = replace_existing_with(&source, &destination, &backup, |from, to| {
            calls += 1;
            if calls == 2 {
                Err(std::io::Error::other("injected rename failure"))
            } else {
                std::fs::rename(from, to)
            }
        })
        .unwrap_err();

        assert!(error.to_string().contains("injected rename failure"));
        assert_eq!(std::fs::read(&destination).unwrap(), b"original");
        assert_eq!(std::fs::read(&source).unwrap(), b"replacement");
        assert!(!backup.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn atomic_write_preserves_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let (_temp, directory) = unique_test_dir("permissions");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("settings.json");
        std::fs::write(&path, b"original").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        atomic_write(&path, b"replacement").unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn unique_test_dir(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::Builder::new()
            .prefix(&format!("nam-trainer-persistence-{name}-"))
            .tempdir()
            .unwrap();
        let path = temp.path().to_path_buf();
        (temp, path)
    }
}
