use crate::{read_json, Violation};
use serde::Deserialize;
use std::path::Path;

pub const PHASES: &str = "research/phases.json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phases {
    pub version: u32,
    pub phases: Vec<Phase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phase {
    pub id: u32,
    pub name: String,
    /// "planned" | "in_progress" | "complete"
    pub status: String,
    /// Executable command (required when complete).
    pub command: Option<String>,
    /// Repo-relative path of the report JSON schema (required when complete).
    pub report_schema: Option<String>,
}

pub fn load(root: &Path) -> Result<Phases, Violation> {
    read_json(root, PHASES, "phase-gates")
}

pub fn validate(root: &Path, p: &Phases) -> Vec<Violation> {
    let c = "phase-gates";
    let mut v = Vec::new();
    for (i, ph) in p.phases.iter().enumerate() {
        if ph.id as usize != i {
            v.push(Violation::new(c, format!("phase ids must be 0..n in order; got {} at {i}", ph.id)));
        }
        if !["planned", "in_progress", "complete"].contains(&ph.status.as_str()) {
            v.push(Violation::new(c, format!("phase {}: bad status '{}'", ph.id, ph.status)));
        }
        if ph.status == "complete" {
            match ph.command.as_deref() {
                Some(cmd) if !cmd.trim().is_empty() => {}
                _ => v.push(Violation::new(c, format!("phase {} complete without a command", ph.id))),
            }
            match ph.report_schema.as_deref() {
                Some(s) if root.join(s).is_file() => {}
                Some(s) => v.push(Violation::new(c, format!("phase {}: report_schema '{s}' not found", ph.id))),
                None => v.push(Violation::new(c, format!("phase {} complete without a report_schema", ph.id))),
            }
        }
        // A later phase may not be complete while an earlier one is not.
        if ph.status == "complete" && p.phases[..i].iter().any(|e| e.status != "complete") {
            v.push(Violation::new(c, format!("phase {} complete before earlier phases", ph.id)));
        }
    }
    v
}
