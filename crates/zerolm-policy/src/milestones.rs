//! The milestone tracker (`research/milestones.json`): roadmap milestones by
//! name, in roadmap order, with their status. A milestone counts as complete
//! only with an executable command and a report schema, and only after every
//! earlier milestone.

use crate::{read_json, Violation};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

pub const MILESTONES: &str = "research/milestones.json";
pub const CHECK: &str = "milestone-gates";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Milestones {
    pub version: u32,
    pub order: String,
    pub milestones: Vec<Milestone>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Milestone {
    /// Stable lowercase name, e.g. "memory".
    pub id: String,
    pub name: String,
    /// "planned" | "in_progress" | "complete"
    pub status: String,
    /// Executable command (required when complete).
    pub command: Option<String>,
    /// Repo-relative path of the report JSON schema (required when complete).
    pub report_schema: Option<String>,
}

pub fn load(root: &Path) -> Result<Milestones, Violation> {
    read_json(root, MILESTONES, CHECK)
}

pub fn validate(root: &Path, m: &Milestones) -> Vec<Violation> {
    let mut v = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, ms) in m.milestones.iter().enumerate() {
        let id = &ms.id;
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_lowercase() || c == '-') || !seen.insert(id.as_str()) {
            v.push(Violation::new(CHECK, format!("milestone ids must be unique lowercase names; got '{id}'")));
        }
        if !["planned", "in_progress", "complete"].contains(&ms.status.as_str()) {
            v.push(Violation::new(CHECK, format!("{id}: bad status '{}'", ms.status)));
        }
        if ms.status == "complete" {
            match ms.command.as_deref() {
                Some(cmd) if !cmd.trim().is_empty() => {}
                _ => v.push(Violation::new(CHECK, format!("{id} complete without a command"))),
            }
            match ms.report_schema.as_deref() {
                Some(s) if root.join(s).is_file() => {}
                Some(s) => v.push(Violation::new(CHECK, format!("{id}: report_schema '{s}' not found"))),
                None => v.push(Violation::new(CHECK, format!("{id} complete without a report_schema"))),
            }
            // A later milestone may not be complete while an earlier one is not.
            if m.milestones[..i].iter().any(|e| e.status != "complete") {
                v.push(Violation::new(CHECK, format!("{id} complete before earlier milestones")));
            }
        }
    }
    v
}
