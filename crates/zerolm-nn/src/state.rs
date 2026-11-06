//! Exact state serialization: header, little-endian f32 bits, checksum.

use zerolm_tensor::fnv1a;

const MAGIC: &[u8; 4] = b"ZLST";

#[derive(Debug, PartialEq, Eq)]
pub enum StateError {
    Truncated,
    BadMagic,
    BadChecksum,
    WrongCell { expected: String, found: String },
    WrongLength { expected: usize, found: usize },
}

pub fn encode(cell: &str, state: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + cell.len() + state.len() * 4);
    out.extend_from_slice(MAGIC);
    out.push(cell.len() as u8);
    out.extend_from_slice(cell.as_bytes());
    out.extend_from_slice(&(state.len() as u32).to_le_bytes());
    for v in state {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    let sum = fnv1a(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

pub fn decode(cell: &str, expected_len: usize, bytes: &[u8]) -> Result<Vec<f32>, StateError> {
    if bytes.len() < 5 + 4 + 8 {
        return Err(StateError::Truncated);
    }
    if &bytes[..4] != MAGIC {
        return Err(StateError::BadMagic);
    }
    let (body, tail) = bytes.split_at(bytes.len() - 8);
    if fnv1a(body).to_le_bytes() != tail {
        return Err(StateError::BadChecksum);
    }
    let n = body[4] as usize;
    let rest = body.get(5 + n..).ok_or(StateError::Truncated)?;
    let found = String::from_utf8_lossy(&body[5..5 + n]).into_owned();
    if found != cell {
        return Err(StateError::WrongCell { expected: cell.into(), found });
    }
    let len_bytes: [u8; 4] = rest.get(..4).ok_or(StateError::Truncated)?.try_into().unwrap();
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len != expected_len {
        return Err(StateError::WrongLength { expected: expected_len, found: len });
    }
    let data = &rest[4..];
    if data.len() != len * 4 {
        return Err(StateError::Truncated);
    }
    Ok(data.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap()))).collect())
}
