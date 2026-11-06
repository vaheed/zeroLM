use crate::manifest::UpdatePolicy;
use crate::{read_json, Violation};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

pub const WEIGHT_GROUPS: &str = "model/specs/weight-groups.json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightGroups {
    pub version: u32,
    pub groups: Vec<Group>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Group {
    pub id: String,
    pub description: String,
    /// Id of a `research/manifest/seed-knowledge.json` entry, or "zeros" /
    /// "seeded_prng" (deterministic, documented in the seed manifest).
    pub init_source: String,
    /// Must be one of the policy's `update_paths`.
    pub update_path: String,
}

pub fn load(root: &Path) -> Result<WeightGroups, Violation> {
    read_json(root, WEIGHT_GROUPS, "weight-groups")
}

pub fn validate(w: &WeightGroups, policy: Option<&UpdatePolicy>) -> Vec<Violation> {
    let c = "weight-groups";
    let mut v = Vec::new();
    if w.groups.is_empty() {
        v.push(Violation::new(c, "no weight groups declared"));
    }
    let mut ids = BTreeSet::new();
    for g in &w.groups {
        if !ids.insert(g.id.as_str()) {
            v.push(Violation::new(c, format!("duplicate group '{}'", g.id)));
        }
        if g.init_source.trim().is_empty() {
            v.push(Violation::new(c, format!("{}: init_source missing", g.id)));
        }
        if g.description.trim().is_empty() {
            v.push(Violation::new(c, format!("{}: description missing", g.id)));
        }
        if let Some(p) = policy {
            if !p.update_paths.contains(&g.update_path) {
                v.push(Violation::new(
                    c,
                    format!("{}: update_path '{}' not in policy update_paths", g.id, g.update_path),
                ));
            }
        }
    }
    v
}

/// Init sources that need no seed-manifest entry.
pub const BUILTIN_INIT: [&str; 2] = ["zeros", "seeded_prng"];

/// Cross-check group init sources against seed entries.
pub fn check_init_sources(w: &WeightGroups, seed_ids: &BTreeSet<String>) -> Vec<Violation> {
    w.groups
        .iter()
        .filter(|g| !BUILTIN_INIT.contains(&g.init_source.as_str()) && !seed_ids.contains(&g.init_source))
        .map(|g| Violation::new("weight-groups", format!("{}: init_source '{}' is not a seed entry", g.id, g.init_source)))
        .collect()
}
