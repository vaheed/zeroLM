//! The corpus manifest and its validation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    /// Byte that ends a document inside a shard.
    pub document_separator: u8,
    pub splits: Splits,
    pub split_seed: u64,
    pub sources: Vec<Source>,
    /// Source ids in the order their exposure documents are streamed.
    pub exposure_order: Vec<String>,
    pub evaluation_sets: Vec<EvalSet>,
}

/// Percent of documents per role; must sum to 100.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Splits {
    pub exposure: u32,
    pub probe: u32,
    pub dev: u32,
    pub hidden: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub description: String,
    pub license: String,
    pub license_url: String,
    pub origin_url: String,
    /// Must be false: no language-model-generated text.
    pub generated_by_model: bool,
    /// Must be false: no benchmark or evaluation data in exposure sources.
    pub benchmark: bool,
    pub shards: Vec<Shard>,
    /// Optional per-source split percentages and seed. A source with few, large documents can
    /// need its own split to leave usable probe and held-out roles; the choice is made from
    /// document sizes only, before any model is run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splits: Option<Splits>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    /// Path relative to the data root.
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalKind {
    HeldoutBpb,
    Blimp,
    Lambada,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSet {
    pub id: String,
    pub kind: EvalKind,
    pub license: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

impl Manifest {
    pub fn parse(json: &str) -> Result<Self, String> {
        let m: Manifest = serde_json::from_str(json).map_err(|e| format!("manifest: {e}"))?;
        m.validate()?;
        Ok(m)
    }

    pub fn source(&self, id: &str) -> Option<&Source> {
        self.sources.iter().find(|s| s.id == id)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1 {
            return Err(format!("unsupported manifest version {}", self.version));
        }
        for s in std::iter::once(self.splits).chain(self.sources.iter().filter_map(|x| x.splits)) {
            if s.exposure + s.probe + s.dev + s.hidden != 100 || s.exposure == 0 || s.probe == 0 || s.dev == 0 || s.hidden == 0 {
                return Err("splits must be positive and sum to 100".into());
            }
        }
        let mut ids = BTreeSet::new();
        for src in &self.sources {
            if !ids.insert(src.id.as_str()) || src.id.is_empty() {
                return Err(format!("duplicate or empty source id '{}'", src.id));
            }
            if src.license.trim().is_empty() || src.license_url.trim().is_empty() || src.origin_url.trim().is_empty() {
                return Err(format!("source '{}': license, license_url, and origin_url are required", src.id));
            }
            if src.generated_by_model {
                return Err(format!("source '{}': model-generated text is excluded", src.id));
            }
            if src.benchmark {
                return Err(format!("source '{}': benchmark data is excluded from exposure", src.id));
            }
            if src.shards.is_empty() {
                return Err(format!("source '{}' has no shards", src.id));
            }
            for sh in &src.shards {
                check_file(&sh.path, &sh.sha256).map_err(|e| format!("source '{}': {e}", src.id))?;
            }
        }
        let mut seen = BTreeSet::new();
        for id in &self.exposure_order {
            if self.source(id).is_none() || !seen.insert(id.as_str()) {
                return Err(format!("exposure_order: unknown or repeated source '{id}'"));
            }
        }
        let mut eval_ids = BTreeSet::new();
        for e in &self.evaluation_sets {
            if !eval_ids.insert(e.id.as_str()) || e.license.trim().is_empty() {
                return Err(format!("evaluation set '{}': duplicate id or missing license", e.id));
            }
            check_file(&e.path, &e.sha256).map_err(|m| format!("evaluation set '{}': {m}", e.id))?;
        }
        Ok(())
    }
}

fn check_file(path: &str, sha256: &str) -> Result<(), String> {
    if path.is_empty() || path.starts_with('/') || path.split('/').any(|p| p == "..") {
        return Err(format!("path '{path}' must be relative and stay under the data root"));
    }
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(format!("'{path}': sha256 must be 64 lowercase hex digits"));
    }
    Ok(())
}
