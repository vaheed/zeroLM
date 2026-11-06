//! Phase 0 auditor. Everything here is deterministic and file-based:
//! it reads the manifests under `research/` and `model/specs/` and the
//! repository tree, and returns a list of violations (empty == pass).

pub mod baseline;
pub mod manifest;
pub mod phases;
pub mod scan;
pub mod weights;

use std::path::Path;

/// One failed check. `check` is a stable id, `detail` is human-readable.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Violation {
    #[serde(deserialize_with = "leak")]
    pub check: &'static str,
    pub detail: String,
}

impl Violation {
    pub fn new(check: &'static str, detail: impl Into<String>) -> Self {
        Self { check, detail: detail.into() }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Report {
    pub phase: &'static str,
    pub passed: bool,
    pub checks_run: Vec<&'static str>,
    pub violations: Vec<Violation>,
}

/// Run every Phase 0 check against the repository rooted at `root`.
pub fn audit(root: &Path) -> Report {
    let mut violations = Vec::new();
    let mut checks_run = Vec::new();

    checks_run.push("update-policy-schema");
    let policy = match manifest::load_update_policy(root) {
        Ok(p) => {
            violations.extend(manifest::validate_update_policy(&p));
            Some(p)
        }
        Err(v) => {
            violations.push(v);
            None
        }
    };

    checks_run.push("seed-knowledge-schema");
    let mut seed_ids = std::collections::BTreeSet::new();
    match manifest::load_seed_knowledge(root) {
        Ok(s) => {
            violations.extend(manifest::validate_seed_knowledge(&s));
            seed_ids = s.entries.iter().map(|e| e.id.clone()).collect();
        }
        Err(v) => violations.push(v),
    }

    checks_run.push("weight-groups");
    match weights::load(root) {
        Ok(w) => {
            violations.extend(weights::validate(&w, policy.as_ref()));
            violations.extend(weights::check_init_sources(&w, &seed_ids));
        }
        Err(v) => violations.push(v),
    }

    checks_run.push("phase-gates");
    match phases::load(root) {
        Ok(p) => violations.extend(phases::validate(root, &p)),
        Err(v) => violations.push(v),
    }

    checks_run.push("forbidden-artifacts");
    if let Some(p) = &policy {
        violations.extend(scan::scan_tree(root, &p.forbidden));
    }

    Report { phase: "phase-0", passed: violations.is_empty(), checks_run, violations }
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(
    root: &Path,
    rel: &str,
    check: &'static str,
) -> Result<T, Violation> {
    let path = root.join(rel);
    let text = std::fs::read_to_string(path)
        .map_err(|e| Violation::new(check, format!("{rel}: cannot read: {e}")))?;
    serde_json::from_str(&text).map_err(|e| Violation::new(check, format!("{rel}: invalid: {e}")))
}

fn leak<'de, D: serde::Deserializer<'de>>(d: D) -> Result<&'static str, D::Error> {
    let s = <String as serde::Deserialize>::deserialize(d)?;
    Ok(Box::leak(s.into_boxed_str()))
}
