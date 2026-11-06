use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zerolm-base-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join("reports/baseline")).unwrap();
    d
}

fn env_json(dirty: bool) -> String {
    format!(
        r#"{{"git_commit":"{}","git_dirty":{dirty},"os":"linux","arch":"x86_64","rustc":"r","cargo":"c","cpu_model":"m","cpu_cores":1}}"#,
        "a".repeat(40)
    )
}

fn write_reports(d: &Path, dirty: bool, next_passed: bool) {
    let b = d.join("reports/baseline");
    fs::write(b.join("environment.json"), env_json(dirty)).unwrap();
    fs::copy(root().join("research/baseline/legacy-archive.json"), b.join("legacy.json")).unwrap();
    fs::write(b.join("next.json"), format!(r#"{{"passed":{next_passed}}}"#)).unwrap();
}

#[test]
fn legacy_archive_is_valid_and_tag_matches() {
    let text = fs::read_to_string(root().join("research/baseline/legacy-archive.json")).unwrap();
    let l: zerolm_policy::baseline::LegacyArchive = serde_json::from_str(&text).unwrap();
    assert!(zerolm_policy::baseline::validate_legacy(&l, &root()).is_empty());
}

#[test]
fn clean_reports_pass() {
    let d = scratch("clean");
    write_reports(&d, false, true);
    assert!(zerolm_policy::baseline::check(&d).is_empty());
}

#[test]
fn dirty_provenance_is_rejected() {
    let d = scratch("dirty");
    write_reports(&d, true, true);
    assert!(zerolm_policy::baseline::check(&d).iter().any(|v| v.detail.contains("git_dirty")));
}

#[test]
fn failed_audit_is_rejected() {
    let d = scratch("fail");
    write_reports(&d, false, false);
    assert!(!zerolm_policy::baseline::check(&d).is_empty());
}

#[test]
fn missing_reports_are_rejected_not_passed() {
    let d = scratch("missing");
    assert_eq!(zerolm_policy::baseline::check(&d).len(), 3);
}

#[test]
fn generate_writes_all_reports() {
    let out = root().join("reports/baseline");
    let _ = zerolm_policy::baseline::generate(&root()).unwrap();
    for f in ["environment.json", "legacy.json", "next.json"] {
        assert!(out.join(f).is_file(), "{f}");
    }
}
