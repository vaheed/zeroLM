//! Phase 1: provenance and baseline snapshot.
//!
//! Writes `reports/baseline/{environment,legacy,next}.json` and verifies
//! them. Reports from a dirty tree are rejected by `check`, so a baseline
//! can never silently include uncommitted changes.

use crate::{audit, read_json, Violation};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

pub const LEGACY_ARCHIVE: &str = "research/baseline/legacy-archive.json";
pub const OUT_DIR: &str = "reports/baseline";

#[derive(Debug, Serialize, Deserialize)]
pub struct Environment {
    pub git_commit: String,
    pub git_dirty: bool,
    pub os: String,
    pub arch: String,
    pub rustc: String,
    pub cargo: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LegacyArchive {
    pub version: u32,
    pub status: String,
    pub reason: String,
    pub git_tag: String,
    pub git_commit: String,
    pub components: Vec<String>,
    pub results_carried_over: bool,
}

fn run(cmd: &str, args: &[&str], dir: &Path) -> String {
    Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

pub fn environment(root: &Path) -> Environment {
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)).map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    Environment {
        git_commit: run("git", &["rev-parse", "HEAD"], root),
        // Untracked files under reports/ are generated output, not source changes.
        git_dirty: !run("git", &["status", "--porcelain", "--", ".", ":!reports"], root).is_empty(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        rustc: run("rustc", &["--version"], root),
        cargo: run("cargo", &["--version"], root),
        cpu_model,
        cpu_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
    }
}

pub fn validate_legacy(l: &LegacyArchive, root: &Path) -> Vec<Violation> {
    let c = "baseline-legacy";
    let mut v = Vec::new();
    if l.status != "removed" {
        v.push(Violation::new(c, "legacy status must be 'removed'"));
    }
    if l.results_carried_over {
        v.push(Violation::new(c, "results_carried_over must be false"));
    }
    if l.git_commit.len() != 40 || !l.git_commit.chars().all(|ch| ch.is_ascii_hexdigit()) {
        v.push(Violation::new(c, "git_commit must be a full 40-hex SHA"));
    } else {
        let tagged = run("git", &["rev-parse", &format!("{}^{{commit}}", l.git_tag)], root);
        // Outside a git checkout (or a shallow CI clone without tags) we cannot
        // resolve the tag; only a resolvable, mismatching tag is a violation.
        if tagged != "unknown" && tagged != l.git_commit {
            v.push(Violation::new(c, format!("tag '{}' points at {tagged}, not {}", l.git_tag, l.git_commit)));
        }
    }
    v
}

/// Generate all baseline reports. Returns the violations found (empty == ok).
pub fn generate(root: &Path) -> Result<Vec<Violation>, String> {
    let out = root.join(OUT_DIR);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let env = environment(root);
    let legacy: LegacyArchive = read_json(root, LEGACY_ARCHIVE, "baseline-legacy").map_err(|v| v.detail)?;
    let mut violations = validate_legacy(&legacy, root);
    let next = audit(root);
    violations.extend(next.violations.iter().cloned());
    let w = |name: &str, s: String| std::fs::write(out.join(name), s + "\n").map_err(|e| e.to_string());
    w("environment.json", serde_json::to_string_pretty(&env).unwrap())?;
    w("legacy.json", serde_json::to_string_pretty(&legacy).unwrap())?;
    w("next.json", serde_json::to_string_pretty(&next).unwrap())?;
    Ok(violations)
}

/// Verify previously generated reports: all present, parse, clean provenance.
pub fn check(root: &Path) -> Vec<Violation> {
    let c = "baseline-reports";
    let mut v = Vec::new();
    match read_json::<Environment>(root, &format!("{OUT_DIR}/environment.json"), c) {
        Ok(e) => {
            if e.git_dirty {
                v.push(Violation::new(c, "environment.json: git_dirty=true; baseline must come from a clean tree"));
            }
            if e.git_commit == "unknown" {
                v.push(Violation::new(c, "environment.json: git commit unknown"));
            }
        }
        Err(x) => v.push(x),
    }
    if let Err(x) = read_json::<LegacyArchive>(root, &format!("{OUT_DIR}/legacy.json"), c) {
        v.push(x);
    }
    match read_json::<serde_json::Value>(root, &format!("{OUT_DIR}/next.json"), c) {
        Ok(j) if j["passed"] == true => {}
        Ok(_) => v.push(Violation::new(c, "next.json: baseline audit did not pass")),
        Err(x) => v.push(x),
    }
    v
}
