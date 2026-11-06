//! Value encoding into byte-code rows and nearest-code decoding.
//!
//! Semantic queries are answered by the compiled read circuit
//! (`read_circuit`), which does its own cleanup. `decode` remains a host
//! helper for write-time value comparison and `W_FAST` recall; moving those
//! onto the engine is Phase 7 work still open.

use zerolm_tensor::Matrix;

/// Read weights below this are dropped from the value read (a circuit parameter).
pub const MIN_WEIGHT: f32 = 1e-7;

/// Value bytes -> concatenated code rows, NUL-padded to `positions`.
/// The caller guarantees `bytes.len() < positions` and no NUL bytes.
pub fn encode_value(bytes: &[u8], codes: &Matrix, positions: usize) -> Vec<f32> {
    debug_assert!(bytes.len() < positions && !bytes.contains(&0));
    let mut v = Vec::with_capacity(positions * codes.cols);
    for p in 0..positions {
        v.extend_from_slice(codes.row(bytes.get(p).copied().unwrap_or(0) as usize));
    }
    v
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub bytes: Vec<u8>,
    /// Smallest (second-nearest - nearest) squared distance over decoded
    /// positions: how far the read is from flipping to another byte.
    pub margin: f32,
}

/// Nearest-code cleanup per position until the NUL code.
pub fn decode(read: &[f32], codes: &Matrix) -> Decoded {
    let e = codes.cols;
    let mut bytes = Vec::new();
    let mut margin = f32::INFINITY;
    for pos in read.chunks_exact(e) {
        let (mut b1, mut d1, mut d2) = (0usize, f32::INFINITY, f32::INFINITY);
        for b in 0..codes.rows {
            let d: f32 = codes.row(b).iter().zip(pos).map(|(c, x)| (c - x) * (c - x)).sum();
            if d < d1 {
                d2 = d1;
                d1 = d;
                b1 = b;
            } else if d < d2 {
                d2 = d;
            }
        }
        margin = margin.min(d2 - d1);
        if b1 == 0 {
            break;
        }
        bytes.push(b1 as u8);
    }
    Decoded { bytes, margin: if margin.is_finite() { margin } else { 0.0 } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_codes_decode_exactly_and_superposition_follows_the_dominant_slot() {
        let codes = Matrix::uniform(256, 16, 1, "embed", 1.0);
        let a = encode_value(b"blue", &codes, 8);
        let b = encode_value(b"green", &codes, 8);
        assert_eq!(decode(&a, &codes).bytes, b"blue");
        let r: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.9 * x + 0.1 * y).collect();
        assert_eq!(decode(&r, &codes).bytes, b"blue");
    }
}
