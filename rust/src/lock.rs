//! Session tags and life-file locking, so two ZeroLM Live sessions can never
//! silently overwrite each other's life file.

use std::fs::{self, OpenOptions};
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

/// Exclusive claim on a life file; released (lock file removed) on drop.
#[derive(Debug)]
pub struct LifeLock {
    path: PathBuf,
}

fn lock_path(life: &Path) -> PathBuf {
    let mut p = life.as_os_str().to_owned();
    p.push(".lock");
    PathBuf::from(p)
}

/// Whether the process that wrote a lock is still running. Only Linux can be
/// checked without extra dependencies; elsewhere a lock is assumed live.
fn pid_alive(pid: u32) -> bool {
    if cfg!(target_os = "linux") {
        Path::new(&format!("/proc/{pid}")).exists()
    } else {
        true
    }
}

impl LifeLock {
    pub fn acquire(life: &Path) -> Result<LifeLock, String> {
        let path = lock_path(life);
        for _ in 0..2 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    f.write_all(std::process::id().to_string().as_bytes())
                        .map_err(|e| format!("cannot write lock {}: {e}", path.display()))?;
                    return Ok(LifeLock { path });
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    let holder = fs::read_to_string(&path)
                        .ok()
                        .and_then(|s| s.trim().parse::<u32>().ok());
                    match holder {
                        Some(pid) if pid_alive(pid) => {
                            return Err(format!(
                                "life file {} is in use by process {pid}; \
start a separate session with --tag NAME",
                                life.display()
                            ));
                        }
                        // Stale (crashed owner) or unreadable: take it over.
                        _ => {
                            let _ = fs::remove_file(&path);
                        }
                    }
                }
                Err(e) => return Err(format!("cannot create lock {}: {e}", path.display())),
            }
        }
        Err(format!("could not acquire lock {}", path.display()))
    }
}

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

    #[cfg(target_os = "linux")]
    #[test]
    fn stale_lock_from_a_dead_process_is_taken_over() {
        let d = tmp("stale");
        let life = d.join("life.zlm");
        fs::write(lock_path(&life), "4294967290").unwrap(); // no such pid
        assert!(LifeLock::acquire(&life).is_ok());
        fs::remove_dir_all(&d).unwrap();
    }
}
