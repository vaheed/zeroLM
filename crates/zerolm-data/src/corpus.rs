//! Shard verification, whole-document splits, and the exposure stream.

use crate::manifest::{Manifest, Source};
use std::path::Path;
use zerolm_tensor::sha256::Sha256;
use zerolm_tensor::fnv1a;

/// Data role of a whole document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Exposure,
    Probe,
    Dev,
    Hidden,
}

/// Role of document `index` of `source`: a function of the manifest's split
/// fractions and seed, the source id, and the index only, so a document never
/// changes role when other documents or shards change.
pub fn role_of(m: &Manifest, source: &str, index: usize) -> Role {
    let src = m.source(source);
    let (seed, s) = (src.and_then(|x| x.split_seed).unwrap_or(m.split_seed), src.and_then(|x| x.splits).unwrap_or(m.splits));
    let mut key = Vec::new();
    key.extend_from_slice(&seed.to_le_bytes());
    key.extend_from_slice(source.as_bytes());
    key.push(0);
    key.extend_from_slice(&(index as u64).to_le_bytes());
    // FNV-1a's low bits are weak for short structured keys; mix before reducing.
    let mut h = fnv1a(&key);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    let r = (h % 100) as u32;
    if r < s.exposure {
        Role::Exposure
    } else if r < s.exposure + s.probe {
        Role::Probe
    } else if r < s.exposure + s.probe + s.dev {
        Role::Dev
    } else {
        Role::Hidden
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finish_hex()
}

/// Check every shard and evaluation file under `root` against the manifest
/// (size and SHA-256). Returns the list of problems (empty when all match).
pub fn verify(m: &Manifest, root: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    let files = m.sources.iter().flat_map(|s| s.shards.iter().map(|x| (&x.path, &x.sha256, x.bytes))).chain(m.evaluation_sets.iter().map(|e| (&e.path, &e.sha256, e.bytes)));
    for (path, sha, bytes) in files {
        match std::fs::read(root.join(path)) {
            Err(e) => problems.push(format!("{path}: {e}")),
            Ok(data) => {
                if data.len() as u64 != bytes {
                    problems.push(format!("{path}: {} bytes, manifest says {bytes}", data.len()));
                } else if sha256_hex(&data) != *sha {
                    problems.push(format!("{path}: SHA-256 differs from the manifest"));
                }
            }
        }
    }
    problems
}

/// The documents of one source, in shard order then position order.
pub fn documents(m: &Manifest, src: &Source, root: &Path) -> Result<Vec<Vec<u8>>, String> {
    let mut docs = Vec::new();
    for sh in &src.shards {
        let data = std::fs::read(root.join(&sh.path)).map_err(|e| format!("{}: {e}", sh.path))?;
        docs.extend(data.split(|b| *b == m.document_separator).filter(|d| !d.is_empty()).map(|d| d.to_vec()));
    }
    Ok(docs)
}

/// All documents of `role` in exposure order (sources in `exposure_order`
/// for `Exposure`, manifest order for every other role), each followed by
/// two newline bytes.
pub fn role_stream(m: &Manifest, root: &Path, role: Role) -> Result<Vec<u8>, String> {
    let order: Vec<&str> = if role == Role::Exposure { m.exposure_order.iter().map(String::as_str).collect() } else { m.sources.iter().map(|s| s.id.as_str()).collect() };
    let mut out = Vec::new();
    for id in order {
        let src = m.source(id).ok_or_else(|| format!("unknown source {id}"))?;
        for (i, d) in documents(m, src, root)?.iter().enumerate() {
            if role_of(m, id, i) == role {
                out.extend_from_slice(d);
                out.extend_from_slice(b"\n\n");
            }
        }
    }
    Ok(out)
}

/// Like [`role_stream`] for one source only (used by the two-source retention test).
pub fn source_role_stream(m: &Manifest, root: &Path, source: &str, role: Role) -> Result<Vec<u8>, String> {
    let src = m.source(source).ok_or_else(|| format!("unknown source {source}"))?;
    let mut out = Vec::new();
    for (i, d) in documents(m, src, root)?.iter().enumerate() {
        if role_of(m, source, i) == role {
            out.extend_from_slice(d);
            out.extend_from_slice(b"\n\n");
        }
    }
    Ok(out)
}
