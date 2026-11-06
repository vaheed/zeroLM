//! Program bundle: a program and the weight tensors it runs on, in one byte
//! string that an independent interpreter can execute.
//!
//! ```text
//! "ZLMPROG1" | u32 header_len | header JSON | f32 LE tensor data... | SHA-256(all previous bytes)
//! ```
//! The header is `{"program": <Program>, "tensors": [{"name", "rows", "cols"}, ...]}`;
//! tensor data follows in directory order.

use super::{err, Buf, IsaError, Program, View};
use serde::{Deserialize, Serialize};
use zerolm_tensor::sha256::Sha256;

pub const MAGIC: &[u8; 8] = b"ZLMPROG1";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    rows: usize,
    cols: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    program: Program,
    tensors: Vec<Entry>,
}

pub fn encode(program: &Program, tensors: &[(&str, View<'_>)]) -> Vec<u8> {
    let header = Header {
        program: program.clone(),
        tensors: tensors.iter().map(|(n, v)| Entry { name: n.to_string(), rows: v.rows, cols: v.cols }).collect(),
    };
    let h = serde_json::to_vec(&header).expect("header serializes");
    let mut out = Vec::with_capacity(16 + h.len() + tensors.iter().map(|t| t.1.data.len() * 4).sum::<usize>() + 32);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(h.len() as u32).to_le_bytes());
    out.extend_from_slice(&h);
    for (_, v) in tensors {
        for x in v.data {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut sha = Sha256::new();
    sha.update(&out);
    out.extend_from_slice(&sha.finish());
    out
}

pub fn decode(bytes: &[u8]) -> Result<(Program, Vec<(String, Buf)>), IsaError> {
    if bytes.len() < 12 + 32 || &bytes[..8] != MAGIC {
        return err("bundle: bad magic or truncated");
    }
    let (body, tail) = bytes.split_at(bytes.len() - 32);
    let mut sha = Sha256::new();
    sha.update(body);
    if sha.finish()[..] != tail[..] {
        return err("bundle: checksum mismatch");
    }
    let hl = u32::from_le_bytes(body[8..12].try_into().expect("4 bytes")) as usize;
    if 12 + hl > body.len() {
        return err("bundle: header overruns file");
    }
    let header: Header = serde_json::from_slice(&body[12..12 + hl]).map_err(|e| IsaError(format!("bundle header: {e}")))?;
    let mut at = 12 + hl;
    let mut tensors = Vec::new();
    for e in header.tensors {
        let n = e.rows * e.cols;
        if at + 4 * n > body.len() {
            return err(format!("bundle: tensor {} overruns file", e.name));
        }
        let data = body[at..at + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();
        at += 4 * n;
        tensors.push((e.name, Buf::new(e.rows, e.cols, data)));
    }
    if at != body.len() {
        return err("bundle: trailing bytes");
    }
    Ok((header.program, tensors))
}

/// SHA-256 (hex) of a bundle's bytes.
pub fn sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finish_hex()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::{Decl, Instr};

    #[test]
    fn bundles_round_trip_and_detect_corruption() {
        let p = Program {
            name: "id".into(),
            inputs: vec![Decl::new("x", None, 2)],
            weights: vec![Decl::new("w", Some(2), 2)],
            state: vec![],
            body: vec![Instr::Matvec { w: "w".into(), x: "x".into(), out: "y".into(), transpose: false }],
            outputs: vec!["y".into()],
        };
        let w = [1.0f32, -0.0, f32::MIN_POSITIVE, 3.5];
        let b = encode(&p, &[("w", View::new(2, 2, &w))]);
        let (p2, t) = decode(&b).unwrap();
        assert_eq!(p2, p);
        assert_eq!(t[0].1.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), w.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        let mut bad = b.clone();
        bad[20] ^= 1;
        assert!(decode(&bad).is_err());
    }
}
