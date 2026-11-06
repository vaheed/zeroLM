//! Named-tensor checkpoint file and atomic save with last-valid recovery.
//!
//! File layout (all integers little-endian):
//! ```text
//! "ZLMCKPT1" | u32 header_len | header JSON | f32 tensor data... | SHA-256(all previous bytes)
//! ```
//! The header holds the configuration and the tensor directory (name, weight
//! group, shape) only. Every semantic fact lives in the f32 tensor data.
//!
//! Commit protocol for `save_atomic(dir)`:
//! 1. write `memory.zlmc.tmp` and fsync it;
//! 2. rename the current `memory.zlmc` to `memory.prev.zlmc` (if any);
//! 3. rename the temporary file to `memory.zlmc` (the commit point); fsync the directory.
//!
//! `load_latest` reads `memory.zlmc`, and if it is missing or fails its hash,
//! falls back to `memory.prev.zlmc`. The temporary file is never loaded: it is
//! an uncommitted write, not a second source of truth.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use zerolm_tensor::sha256::Sha256;

pub const MAGIC: &[u8; 8] = b"ZLMCKPT1";
pub const CURRENT: &str = "memory.zlmc";
pub const PREVIOUS: &str = "memory.prev.zlmc";
pub const TEMP: &str = "memory.zlmc.tmp";
const MAX_HEADER: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    /// Weight-group id from `model/specs/weight-groups.json`.
    pub group: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(name: &str, group: &str, rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "tensor {name}: shape mismatch");
        Tensor { name: name.into(), group: group.into(), rows, cols, data }
    }

    /// SHA-256 of the name, shape and f32 bits.
    pub fn sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.name.as_bytes());
        h.update(&(self.rows as u64).to_le_bytes());
        h.update(&(self.cols as u64).to_le_bytes());
        h.update(&f32_bytes(&self.data));
        h.finish_hex()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    group: String,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    format: String,
    meta: serde_json::Value,
    tensors: Vec<Entry>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointError {
    NotFound,
    Io(String),
    BadMagic,
    Truncated,
    HashMismatch,
    BadHeader(String),
    /// Neither the current nor the previous checkpoint is valid.
    NoValidCheckpoint { current: Box<CheckpointError>, previous: Box<CheckpointError> },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckpointError::NotFound => write!(f, "checkpoint not found"),
            CheckpointError::Io(e) => write!(f, "io: {e}"),
            CheckpointError::BadMagic => write!(f, "not a ZeroLM checkpoint"),
            CheckpointError::Truncated => write!(f, "checkpoint truncated"),
            CheckpointError::HashMismatch => write!(f, "checkpoint SHA-256 mismatch (corrupt)"),
            CheckpointError::BadHeader(e) => write!(f, "bad header: {e}"),
            CheckpointError::NoValidCheckpoint { current, previous } => {
                write!(f, "no valid checkpoint (current: {current}; previous: {previous})")
            }
        }
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// Serialize tensors and metadata into checkpoint bytes.
pub fn encode(meta: &serde_json::Value, tensors: &[Tensor]) -> Vec<u8> {
    let header = Header {
        format: "zerolm-checkpoint-v1".into(),
        meta: meta.clone(),
        tensors: tensors
            .iter()
            .map(|t| Entry { name: t.name.clone(), group: t.group.clone(), rows: t.rows, cols: t.cols })
            .collect(),
    };
    let hj = serde_json::to_vec(&header).expect("serialize header");
    let total: usize = tensors.iter().map(|t| t.data.len() * 4).sum();
    let mut out = Vec::with_capacity(8 + 4 + hj.len() + total + 32);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(hj.len() as u32).to_le_bytes());
    out.extend_from_slice(&hj);
    for t in tensors {
        out.extend_from_slice(&f32_bytes(&t.data));
    }
    let digest = sha256_raw(&out);
    out.extend_from_slice(&digest);
    out
}

fn sha256_raw(b: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b);
    let hex = h.finish_hex();
    let mut raw = [0u8; 32];
    for (i, r) in raw.iter_mut().enumerate() {
        *r = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex");
    }
    raw
}

/// Hex SHA-256 of a whole checkpoint file (what event logs record).
pub fn file_sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finish_hex()
}

/// Parse and verify checkpoint bytes. Never returns partial data.
pub fn decode(bytes: &[u8]) -> Result<(serde_json::Value, Vec<Tensor>), CheckpointError> {
    if bytes.len() < 8 + 4 + 32 {
        return Err(if bytes.starts_with(MAGIC) || bytes.is_empty() { CheckpointError::Truncated } else { CheckpointError::BadMagic });
    }
    if &bytes[..8] != MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if sha256_raw(body) != digest {
        return Err(CheckpointError::HashMismatch);
    }
    let hl = u32::from_le_bytes(body[8..12].try_into().expect("4 bytes")) as usize;
    if hl > MAX_HEADER || 12 + hl > body.len() {
        return Err(CheckpointError::BadHeader("header length out of range".into()));
    }
    let header: Header = serde_json::from_slice(&body[12..12 + hl]).map_err(|e| CheckpointError::BadHeader(e.to_string()))?;
    if header.format != "zerolm-checkpoint-v1" {
        return Err(CheckpointError::BadHeader(format!("unknown format '{}'", header.format)));
    }
    let mut off = 12 + hl;
    let mut tensors = Vec::with_capacity(header.tensors.len());
    for e in header.tensors {
        let n = e.rows.checked_mul(e.cols).and_then(|n| n.checked_mul(4)).ok_or(CheckpointError::BadHeader("shape overflow".into()))?;
        if off + n > body.len() {
            return Err(CheckpointError::BadHeader(format!("tensor {} exceeds file", e.name)));
        }
        let data = body[off..off + n].chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
        off += n;
        tensors.push(Tensor { name: e.name, group: e.group, rows: e.rows, cols: e.cols, data });
    }
    if off != body.len() {
        return Err(CheckpointError::BadHeader("trailing bytes after tensors".into()));
    }
    Ok((header.meta, tensors))
}

/// Where a loaded checkpoint came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadSource {
    Current,
    /// The current file was missing or corrupt; the previous one was used.
    Previous,
}

/// Simulated crash points for the interrupted-write tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    None,
    /// Crash while writing the temporary file (it is left truncated).
    DuringTempWrite,
    /// Crash after the temporary file is complete, before any rename.
    BeforeRename,
    /// Crash after moving current to previous, before the commit rename.
    BetweenRenames,
}

fn io(e: std::io::Error) -> CheckpointError {
    CheckpointError::Io(e.to_string())
}

/// Atomically replace the checkpoint in `dir`. Returns the file SHA-256.
pub fn save_atomic(dir: &Path, bytes: &[u8]) -> Result<String, CheckpointError> {
    save_with_fault(dir, bytes, Fault::None)
}

/// `save_atomic` with an injected crash point (tests and the benchmark only).
pub fn save_with_fault(dir: &Path, bytes: &[u8], fault: Fault) -> Result<String, CheckpointError> {
    std::fs::create_dir_all(dir).map_err(io)?;
    let (tmp, cur, prev) = paths(dir);
    {
        let mut f = std::fs::File::create(&tmp).map_err(io)?;
        if fault == Fault::DuringTempWrite {
            f.write_all(&bytes[..bytes.len() / 2]).map_err(io)?;
            return Err(CheckpointError::Io("injected crash during temp write".into()));
        }
        f.write_all(bytes).map_err(io)?;
        f.sync_all().map_err(io)?;
    }
    if fault == Fault::BeforeRename {
        return Err(CheckpointError::Io("injected crash before rename".into()));
    }
    if cur.exists() {
        std::fs::rename(&cur, prev).map_err(io)?;
    }
    if fault == Fault::BetweenRenames {
        return Err(CheckpointError::Io("injected crash between renames".into()));
    }
    std::fs::rename(tmp, cur).map_err(io)?;
    sync_dir(dir);
    Ok(file_sha256(bytes))
}

fn paths(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    (dir.join(TEMP), dir.join(CURRENT), dir.join(PREVIOUS))
}

#[cfg(unix)]
fn sync_dir(dir: &Path) {
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) {}

fn read_valid(path: &Path) -> Result<(Vec<u8>, serde_json::Value, Vec<Tensor>), CheckpointError> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(CheckpointError::NotFound),
        Err(e) => return Err(io(e)),
    };
    let (meta, tensors) = decode(&bytes)?;
    Ok((bytes, meta, tensors))
}

/// A verified checkpoint read from disk.
pub struct Loaded {
    pub source: LoadSource,
    /// Why the current file was not used (when `source` is `Previous`).
    pub current_error: Option<CheckpointError>,
    pub sha256: String,
    pub meta: serde_json::Value,
    pub tensors: Vec<Tensor>,
}

/// Load the newest valid checkpoint. `Err(NotFound)` means no checkpoint
/// exists at all (a fresh model); any other error means nothing valid could
/// be recovered and nothing was loaded.
pub fn load_latest(dir: &Path) -> Result<Loaded, CheckpointError> {
    let (_, cur, prev) = paths(dir);
    match read_valid(&cur) {
        Ok((b, meta, tensors)) => Ok(Loaded { source: LoadSource::Current, current_error: None, sha256: file_sha256(&b), meta, tensors }),
        Err(ce) => match read_valid(&prev) {
            Ok((b, meta, tensors)) => {
                Ok(Loaded { source: LoadSource::Previous, current_error: Some(ce), sha256: file_sha256(&b), meta, tensors })
            }
            Err(CheckpointError::NotFound) if ce == CheckpointError::NotFound => Err(CheckpointError::NotFound),
            Err(pe) => Err(CheckpointError::NoValidCheckpoint { current: Box::new(ce), previous: Box::new(pe) }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Tensor> {
        vec![Tensor::new("a", "g", 2, 2, vec![1.0, -2.0, 0.5, f32::MIN_POSITIVE]), Tensor::new("b", "h", 1, 3, vec![0.0; 3])]
    }

    #[test]
    fn round_trip_is_bit_exact() {
        let meta = serde_json::json!({"k": 1});
        let bytes = encode(&meta, &sample());
        let (m, t) = decode(&bytes).unwrap();
        assert_eq!(m, meta);
        assert_eq!(t, sample());
        assert_eq!(encode(&meta, &sample()), bytes, "encoding is deterministic");
    }

    #[test]
    fn every_single_byte_flip_is_detected() {
        let bytes = encode(&serde_json::json!({}), &sample());
        for i in 0..bytes.len() {
            let mut b = bytes.clone();
            b[i] ^= 0x01;
            assert!(decode(&b).is_err(), "flip at byte {i} undetected");
        }
        for n in 0..bytes.len() {
            assert!(decode(&bytes[..n]).is_err(), "truncation to {n} undetected");
        }
    }
}
