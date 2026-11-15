//! Weight-space programming (WSP) as a compiler (roadmap sections 2.8, 3.7).
//!
//! [`compiler`] turns a typed circuit specification from `model/prog/` into a
//! neural ISA program and its `W_PROG` constant tensors; [`patch`] edits those
//! constants with versioned, reversible patches. This is a development-time
//! tool whose output is pinned by hash; it is not a learning algorithm.

pub mod circuits;
pub mod compiler;
pub mod patch;

pub use compiler::{compile, parse_spec, Circuit, Compiled, ProgTensor, Spec};

/// Pinned circuit specifications shipped with the model, as (file name, JSON).
pub const SPECS: [(&str, &str); 21] = [
    ("memory-read.json", include_str!("../../../model/prog/memory-read.json")),
    ("memory-encode.json", include_str!("../../../model/prog/memory-encode.json")),
    ("relation-graph.json", include_str!("../../../model/prog/relation-graph.json")),
    ("atom-encode.json", include_str!("../../../model/prog/atom-encode.json")),
    ("relation-reach.json", include_str!("../../../model/prog/relation-reach.json")),
    ("number-encode.json", include_str!("../../../model/prog/number-encode.json")),
    ("number-decode.json", include_str!("../../../model/prog/number-decode.json")),
    ("value-encode.json", include_str!("../../../model/prog/value-encode.json")),
    ("fast-recall.json", include_str!("../../../model/prog/fast-recall.json")),
    ("fast-write.json", include_str!("../../../model/prog/fast-write.json")),
    ("relation-probe.json", include_str!("../../../model/prog/relation-probe.json")),
    ("symbol-cleanup.json", include_str!("../../../model/prog/symbol-cleanup.json")),
    ("relation-join.json", include_str!("../../../model/prog/relation-join.json")),
    ("relation-levels.json", include_str!("../../../model/prog/relation-levels.json")),
    ("relation-rules.json", include_str!("../../../model/prog/relation-rules.json")),
    ("relation-counts.json", include_str!("../../../model/prog/relation-counts.json")),
    ("relation-verdict.json", include_str!("../../../model/prog/relation-verdict.json")),
    ("memory-write.json", include_str!("../../../model/prog/memory-write.json")),
    ("memory-forget.json", include_str!("../../../model/prog/memory-forget.json")),
    ("memory-compact.json", include_str!("../../../model/prog/memory-compact.json")),
    ("memory-history.json", include_str!("../../../model/prog/memory-history.json")),
];

/// `model/prog/pinned.json`: circuit id -> SHA-256 of the compiled output.
pub const PINNED: &str = include_str!("../../../model/prog/pinned.json");

/// Compile a shipped spec by circuit id.
pub fn shipped(id: &str) -> Result<Compiled, String> {
    for (_, json) in SPECS {
        let spec = parse_spec(json)?;
        if spec.id == id {
            return compile(&spec).map_err(|e| e.to_string());
        }
    }
    Err(format!("no shipped circuit '{id}'"))
}

/// Pinned hashes as (id, sha256).
pub fn pinned() -> Result<Vec<(String, String)>, String> {
    let v: serde_json::Value = serde_json::from_str(PINNED).map_err(|e| e.to_string())?;
    let obj = v["circuits"].as_object().ok_or("pinned.json: no 'circuits' object")?;
    Ok(obj.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string())).collect())
}

/// Compile every shipped spec and compare with the pins. Returns (id, sha, pinned?) rows.
pub fn check_pins() -> Result<Vec<(String, String, bool)>, String> {
    let pins = pinned()?;
    let mut out = Vec::new();
    for (_, json) in SPECS {
        let spec = parse_spec(json)?;
        let sha = compile(&spec).map_err(|e| e.to_string())?.sha256();
        let ok = pins.iter().any(|(id, p)| *id == spec.id && *p == sha);
        out.push((spec.id, sha, ok));
    }
    if pins.len() != out.len() {
        return Err(format!("pinned.json lists {} circuits, {} are shipped", pins.len(), out.len()));
    }
    Ok(out)
}
