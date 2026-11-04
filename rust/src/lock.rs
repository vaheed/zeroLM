//! Session tags and life-file locking, so two ZeroLM Live sessions can never
//! silently overwrite each other's life file.
//!
//! On Unix the lock is an OS advisory lock (`flock`) on `<life file>.lock`.
//! The kernel enforces it across processes *and containers* that share the
//! file, and releases it automatically when the holder exits or crashes, so
//! there is no stale-lock cleanup. (A pid file cannot do this: every container
//! has its own pid namespace.) `flock` needs a local filesystem; it is not
//! reliable on some network filesystems.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

pub const MAX_TAG_LEN: usize = 32;

/// `--tag work` -> `life.work.zlm`. Tags are limited to `[A-Za-z0-9_-]` so
/// they cannot escape the working directory or collide with other files.
pub fn tagged_life_path(tag: &str) -> Result<String, String> {
    let ok = !tag.is_empty()
        && tag.len() <= MAX_TAG_LEN
        && tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !ok {
        return Err(format!(
            "invalid tag {tag:?}: use 1-{MAX_TAG_LEN} letters, digits, '-' or '_'"
        ));
    }
    Ok(format!("life.{tag}.zlm"))
}

/// Exclusive claim on a life file; released when dropped.
#[derive(Debug)]
pub struct LifeLock {
    _file: File,
    /// Non-Unix fallback only: the lock file to delete on drop.
    #[cfg(not(unix))]
    path: PathBuf,
}

fn lock_path(life: &Path) -> PathBuf {
    let mut p = life.as_os_str().to_owned();
    p.push(".lock");
    PathBuf::from(p)
}

fn in_use(life: &Path, holder: &Path) -> String {
    let who = fs::read_to_string(holder)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .map(|pid| format!(" (pid {pid} in its own process namespace)"))
        .unwrap_or_default();
    format!(
        "life file {} is in use by another session{who}; start a separate session with --tag NAME",
        life.display()
    )
}

#[cfg(unix)]
mod sys {
    use std::os::raw::c_int;
    extern "C" {
        pub fn flock(fd: c_int, operation: c_int) -> c_int;
    }
    pub const LOCK_EX: c_int = 2;
    pub const LOCK_NB: c_int = 4;
}

impl LifeLock {
    #[cfg(unix)]
    pub fn acquire(life: &Path) -> Result<LifeLock, String> {
        use std::os::unix::io::AsRawFd;
        let path = lock_path(life);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| format!("cannot open lock {}: {e}", path.display()))?;
        // SAFETY: `flock` takes a valid open file descriptor and two ints and
        // has no memory-safety preconditions; the descriptor outlives the call.
        let rc = unsafe { sys::flock(file.as_raw_fd(), sys::LOCK_EX | sys::LOCK_NB) };
        if rc != 0 {
            let e = std::io::Error::last_os_error();
            return Err(if e.kind() == ErrorKind::WouldBlock {
                in_use(life, &path)
            } else {
                format!("cannot lock {}: {e}", path.display())
            });
        }
        // Informational only; the kernel lock is the source of truth.
        let _ = file.set_len(0);
        let _ = file.write_all(std::process::id().to_string().as_bytes());
        Ok(LifeLock { _file: file })
    }

    #[cfg(not(unix))]
    pub fn acquire(life: &Path) -> Result<LifeLock, String> {
        let path = lock_path(life);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let _ = file.write_all(std::process::id().to_string().as_bytes());
                Ok(LifeLock { _file: file, path })
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => Err(format!(
                "{} (if no session is running, delete {})",
                in_use(life, &path),
                path.display()
            )),
            Err(e) => Err(format!("cannot create lock {}: {e}", path.display())),
        }
    }
}

#[cfg(not(unix))]
impl Drop for LifeLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zerolm-lock-{}-{name}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn tags_are_validated() {
        assert_eq!(tagged_life_path("work-1_a").unwrap(), "life.work-1_a.zlm");
        for bad in ["", "a/b", "..", "a b", "x.y", &"z".repeat(33)] {
            assert!(tagged_life_path(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn second_session_is_refused_and_lock_is_released() {
        let d = tmp("excl");
        let life = d.join("life.zlm");
        let first = LifeLock::acquire(&life).unwrap();
        let err = LifeLock::acquire(&life).unwrap_err();
        assert!(err.contains("--tag"), "{err}");
        drop(first);
        assert!(LifeLock::acquire(&life).is_ok());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn different_tags_do_not_conflict() {
        let d = tmp("tags");
        let a = LifeLock::acquire(&d.join("life.a.zlm")).unwrap();
        let b = LifeLock::acquire(&d.join("life.b.zlm")).unwrap();
        drop((a, b));
        fs::remove_dir_all(&d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn leftover_lock_file_from_a_dead_process_does_not_block() {
        // Only a *held* kernel lock blocks; a file left behind by a crash does not.
        let d = tmp("stale");
        let life = d.join("life.zlm");
        fs::write(lock_path(&life), "4294967290").unwrap();
        assert!(LifeLock::acquire(&life).is_ok());
        fs::remove_dir_all(&d).unwrap();
    }
}
