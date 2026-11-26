use crate::{read_json, Violation};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

pub const UPDATE_POLICY: &str = "research/manifest/update-policy.json";
pub const SEED_KNOWLEDGE: &str = "research/manifest/seed-knowledge.json";

/// The update kinds defined by the constitution. Every weight group's
/// permitted update path must be one of `update_paths`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdatePolicy {
    pub version: u32,
    /// Online backpropagation in the mainline. False until the written Gate A decision; true only
    /// together with `online_gradient` (single pass, no offline pretraining, every update logged).
    pub online_backprop_in_mainline: bool,
    #[serde(default)]
    pub online_gradient: Option<OnlineGradient>,
    /// Allowed teachers for any update (observed continuation, correction, ...).
    pub allowed_signals: Vec<String>,
    /// Signals that must never drive an update (e.g. the model's own output).
    pub forbidden_signals: Vec<String>,
    /// Named update paths (none, local_plasticity, memory_write, wsp_patch).
    pub update_paths: Vec<String>,
    /// Fields every logged update event must carry.
    pub required_event_fields: Vec<String>,
    pub forbidden: Forbidden,
}

/// The terms under which online gradient updates are allowed in the mainline.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnlineGradient {
    pub single_pass: bool,
    pub offline_pretraining: bool,
    pub every_update_logged: bool,
    pub amends: String,
    /// File (relative to the repository root) holding the written decision.
    pub decision_record: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Forbidden {
    /// File extensions that indicate pretrained weight files.
    pub extensions: Vec<String>,
    /// Substrings in source files that indicate model downloads.
    pub substrings: Vec<String>,
    /// Repo-relative path prefixes that are exempt (docs, this policy).
    pub exempt_prefixes: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedKnowledge {
    pub version: u32,
    /// Languages/alphabets the bootstrap may know: byte level only at first.
    pub bootstrap_language: String,
    pub entries: Vec<SeedEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedEntry {
    pub id: String,
    /// "structural_prior" | "init_formula" | "protocol_constant"
    pub kind: String,
    pub description: String,
    /// Where the human-authored definition lives (path or spec id).
    pub source: String,
    /// Must be false: no entry may carry learned or corpus-derived content.
    pub derived_from_corpus: bool,
}

const SEED_KINDS: [&str; 3] = ["structural_prior", "init_formula", "protocol_constant"];

pub fn load_update_policy(root: &Path) -> Result<UpdatePolicy, Violation> {
    read_json(root, UPDATE_POLICY, "update-policy-schema")
}

pub fn load_seed_knowledge(root: &Path) -> Result<SeedKnowledge, Violation> {
    read_json(root, SEED_KNOWLEDGE, "seed-knowledge-schema")
}

pub fn validate_update_policy(p: &UpdatePolicy) -> Vec<Violation> {
    let c = "update-policy-schema";
    let mut v = Vec::new();
    if p.version == 0 {
        v.push(Violation::new(c, "version must be >= 1"));
    }
    match (&p.online_gradient, p.online_backprop_in_mainline) {
        (None, true) => v.push(Violation::new(c, "online_backprop_in_mainline needs an online_gradient block (single pass, logged, decision record)")),
        (Some(_), false) => v.push(Violation::new(c, "online_gradient is only meaningful with online_backprop_in_mainline true")),
        (Some(g), true) => {
            if !g.single_pass || g.offline_pretraining || !g.every_update_logged || g.decision_record.trim().is_empty() {
                v.push(Violation::new(c, "online gradient updates must be single pass, without offline pretraining, fully logged, and name a decision record (constitution invariants 2, 3, 13)"));
            }
        }
        (None, false) => {}
    }
    if !p.online_backprop_in_mainline && p.update_paths.iter().any(|x| x == "online_gradient") {
        v.push(Violation::new(c, "update path 'online_gradient' is not allowed while online_backprop_in_mainline is false"));
    }
    for need in ["none", "local_plasticity", "memory_write", "wsp_patch"] {
        if !p.update_paths.iter().any(|x| x == need) {
            v.push(Violation::new(c, format!("update_paths missing '{need}'")));
        }
    }
    if p.allowed_signals.is_empty() {
        v.push(Violation::new(c, "allowed_signals is empty"));
    }
    if !p.forbidden_signals.iter().any(|s| s == "model_generated_output") {
        v.push(Violation::new(c, "forbidden_signals must include 'model_generated_output' (rule 7)"));
    }
    let overlap: BTreeSet<_> = p.allowed_signals.iter().filter(|s| p.forbidden_signals.contains(s)).collect();
    if !overlap.is_empty() {
        v.push(Violation::new(c, format!("signals both allowed and forbidden: {overlap:?}")));
    }
    for need in ["input_stream", "signal", "rule", "changed_tensors", "checkpoint_sha256"] {
        if !p.required_event_fields.iter().any(|x| x == need) {
            v.push(Violation::new(c, format!("required_event_fields missing '{need}' (rule 3)")));
        }
    }
    if p.forbidden.extensions.is_empty() || p.forbidden.substrings.is_empty() {
        v.push(Violation::new(c, "forbidden extensions/substrings must be non-empty"));
    }
    v
}

pub fn validate_seed_knowledge(s: &SeedKnowledge) -> Vec<Violation> {
    let c = "seed-knowledge-schema";
    let mut v = Vec::new();
    if s.version == 0 {
        v.push(Violation::new(c, "version must be >= 1"));
    }
    if s.bootstrap_language != "utf8-bytes" {
        v.push(Violation::new(c, "bootstrap_language must be 'utf8-bytes' for the initial track"));
    }
    let mut ids = BTreeSet::new();
    for e in &s.entries {
        if !ids.insert(e.id.as_str()) {
            v.push(Violation::new(c, format!("duplicate seed entry id '{}'", e.id)));
        }
        if !SEED_KINDS.contains(&e.kind.as_str()) {
            v.push(Violation::new(c, format!("{}: unknown kind '{}'", e.id, e.kind)));
        }
        if e.derived_from_corpus {
            v.push(Violation::new(c, format!("{}: derived_from_corpus must be false (rule 1/2)", e.id)));
        }
        if e.description.trim().is_empty() || e.source.trim().is_empty() {
            v.push(Violation::new(c, format!("{}: description and source are required", e.id)));
        }
    }
    v
}
