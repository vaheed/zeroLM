//! Versioned weight patches on compiled circuits: diff, apply, revert.
//!
//! A patch edits `W_PROG` constants only (never program structure). It names
//! the circuit, the SHA-256 of the compiled circuit it applies to, and the
//! SHA-256 it produces. Each edit is a run of consecutive values in one
//! tensor with both the old and the new f32 bit patterns, so a patch is its
//! own inverse: `revert(apply(c, p), p) == c` bit for bit, and applying a
//! patch to anything but its base is refused.

use crate::compiler::Compiled;
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "zerolm-wsp-patch-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub tensor: String,
    /// Flat (row-major) index of the first value.
    pub offset: usize,
    /// f32 bit patterns as 8-digit hex.
    pub old: Vec<String>,
    pub new: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Patch {
    pub format: String,
    pub id: String,
    pub description: String,
    pub circuit: String,
    pub base_sha256: String,
    pub result_sha256: String,
    pub edits: Vec<Edit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchError {
    Format(String),
    WrongCircuit { want: String, got: String },
    BaseMismatch { want: String, got: String },
    UnknownTensor(String),
    OutOfRange(String),
    OldValueMismatch { tensor: String, index: usize },
    ResultMismatch { want: String, got: String },
}

impl std::fmt::Display for PatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

fn hex(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

fn unhex(s: &str) -> Result<f32, PatchError> {
    u32::from_str_radix(s, 16).map(f32::from_bits).map_err(|_| PatchError::Format(format!("bad value '{s}'")))
}

/// The patch that turns `base` into `target` (same circuit and shapes).
pub fn diff(base: &Compiled, target: &Compiled, id: &str, description: &str) -> Result<Patch, PatchError> {
    if base.id != target.id || base.program != target.program || base.tensors.len() != target.tensors.len() {
        return Err(PatchError::Format("diff needs the same circuit, program, and tensor list".into()));
    }
    let mut edits = Vec::new();
    for (a, b) in base.tensors.iter().zip(&target.tensors) {
        if a.name != b.name || a.rows != b.rows || a.cols != b.cols {
            return Err(PatchError::Format(format!("tensor {} changed shape or order", a.name)));
        }
        let mut i = 0;
        while i < a.data.len() {
            if a.data[i].to_bits() == b.data[i].to_bits() {
                i += 1;
                continue;
            }
            let start = i;
            while i < a.data.len() && a.data[i].to_bits() != b.data[i].to_bits() {
                i += 1;
            }
            edits.push(Edit {
                tensor: a.name.clone(),
                offset: start,
                old: a.data[start..i].iter().map(|x| hex(*x)).collect(),
                new: b.data[start..i].iter().map(|x| hex(*x)).collect(),
            });
        }
    }
    Ok(Patch {
        format: FORMAT.into(),
        id: id.into(),
        description: description.into(),
        circuit: base.id.clone(),
        base_sha256: base.sha256(),
        result_sha256: target.sha256(),
        edits,
    })
}

fn write(c: &mut Compiled, p: &Patch, forward: bool) -> Result<(), PatchError> {
    if p.format != FORMAT {
        return Err(PatchError::Format(format!("format '{}'", p.format)));
    }
    if p.circuit != c.id {
        return Err(PatchError::WrongCircuit { want: p.circuit.clone(), got: c.id.clone() });
    }
    let (from, to) = if forward { (&p.base_sha256, &p.result_sha256) } else { (&p.result_sha256, &p.base_sha256) };
    let got = c.sha256();
    if &got != from {
        return Err(PatchError::BaseMismatch { want: from.clone(), got });
    }
    let mut next = c.clone();
    for e in &p.edits {
        let t = next.tensors.iter_mut().find(|t| t.name == e.tensor).ok_or_else(|| PatchError::UnknownTensor(e.tensor.clone()))?;
        if e.old.len() != e.new.len() || e.offset + e.old.len() > t.data.len() {
            return Err(PatchError::OutOfRange(e.tensor.clone()));
        }
        let (expect, put) = if forward { (&e.old, &e.new) } else { (&e.new, &e.old) };
        for (j, (x, y)) in expect.iter().zip(put).enumerate() {
            let slot = &mut t.data[e.offset + j];
            if slot.to_bits() != unhex(x)?.to_bits() {
                return Err(PatchError::OldValueMismatch { tensor: e.tensor.clone(), index: e.offset + j });
            }
            *slot = unhex(y)?;
        }
    }
    let got = next.sha256();
    if &got != to {
        return Err(PatchError::ResultMismatch { want: to.clone(), got });
    }
    *c = next;
    Ok(())
}

/// Apply a patch atomically: on any error `c` is unchanged.
pub fn apply(c: &mut Compiled, p: &Patch) -> Result<(), PatchError> {
    write(c, p, true)
}

/// Undo an applied patch atomically.
pub fn revert(c: &mut Compiled, p: &Patch) -> Result<(), PatchError> {
    write(c, p, false)
}
