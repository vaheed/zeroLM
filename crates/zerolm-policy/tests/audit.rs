use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Copy the manifests into a scratch dir so tests can mutate them.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zerolm-policy-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for rel in [
        "research/manifest/update-policy.json",
        "research/manifest/seed-knowledge.json",
        "research/milestones.json",
        "research/schemas/audit-report.schema.json",
        "model/specs/weight-groups.json",
    ] {
        let to = dir.join(rel);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::copy(repo_root().join(rel), to).unwrap();
    }
    dir
}

fn edit(dir: &Path, rel: &str, from: &str, to: &str) {
    let p = dir.join(rel);
    let s = fs::read_to_string(&p).unwrap();
    assert!(s.contains(from), "fixture text '{from}' not found");
    fs::write(p, s.replacen(from, to, 1)).unwrap();
}

fn checks(dir: &Path) -> Vec<&'static str> {
    zerolm_policy::audit(dir).violations.iter().map(|v| v.check).collect()
}

#[test]
fn repository_passes() {
    let r = zerolm_policy::audit(&repo_root());
    assert!(r.passed, "{:?}", r.violations);
}

#[test]
fn online_gradient_needs_its_terms_and_its_decision_record() {
    // Allowed only with the single-pass, no-pretraining, fully-logged block.
    let d = scratch("backprop-offline");
    edit(&d, "research/manifest/update-policy.json", "\"offline_pretraining\": false", "\"offline_pretraining\": true");
    assert!(checks(&d).contains(&"update-policy-schema"));
    let d = scratch("backprop-unlogged");
    edit(&d, "research/manifest/update-policy.json", "\"every_update_logged\": true", "\"every_update_logged\": false");
    assert!(checks(&d).contains(&"update-policy-schema"));
    let d = scratch("backprop-norecord");
    edit(&d, "research/manifest/update-policy.json", "research/decisions/gate-a-2026-11-26.md", "research/decisions/missing.md");
    assert!(checks(&d).contains(&"update-policy-schema"));
    // The flag without its block fails.
    let d = scratch("backprop-flag");
    edit(&d, "research/manifest/update-policy.json", "\"online_backprop_in_mainline\": true", "\"online_backprop_in_mainline\": false");
    assert!(checks(&d).contains(&"update-policy-schema"));
}

#[test]
fn corpus_derived_seed_fails() {
    let d = scratch("corpus");
    edit(&d, "research/manifest/seed-knowledge.json", "\"derived_from_corpus\": false", "\"derived_from_corpus\": true");
    assert!(checks(&d).contains(&"seed-knowledge-schema"));
}

#[test]
fn undeclared_init_source_fails() {
    let d = scratch("init");
    edit(&d, "model/specs/weight-groups.json", "\"zeros\"", "\"mystery\"");
    assert!(checks(&d).contains(&"weight-groups"));
}

#[test]
fn bad_update_path_fails() {
    let d = scratch("path");
    edit(&d, "model/specs/weight-groups.json", "\"wsp_patch\"", "\"free_for_all\"");
    assert!(checks(&d).contains(&"weight-groups"));
}

#[test]
fn complete_milestone_without_command_fails() {
    let d = scratch("milestone");
    edit(&d, "research/milestones.json", "\"command\": \"cargo run -p zerolm-policy --bin zerolm-audit -- --json\"", "\"command\": null");
    assert!(checks(&d).contains(&"milestone-gates"));
}

#[test]
fn pretrained_artifact_and_download_fail() {
    let d = scratch("scan");
    fs::write(d.join("model.safetensors"), b"x").unwrap();
    fs::create_dir_all(d.join("crates")).unwrap();
    fs::write(d.join("crates/x.py"), "m = AutoModel.from_pretrained('x')").unwrap();
    let c = checks(&d);
    assert_eq!(c.iter().filter(|x| **x == "forbidden-artifacts").count(), 2);
}

#[test]
fn pretrained_tokenizer_assets_fail() {
    let d = scratch("tok");
    fs::write(d.join("tokenizer.spm"), b"x").unwrap();
    fs::create_dir_all(d.join("crates")).unwrap();
    fs::write(d.join("crates/t.rs"), "let t = tiktoken::get_encoding(\"cl100k\");").unwrap();
    let c = checks(&d);
    assert_eq!(c.iter().filter(|x| **x == "forbidden-artifacts").count(), 2);
}
