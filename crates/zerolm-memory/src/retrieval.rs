//! The model's retrieval computation over memory tensors.
//!
//! Read: scores s_i = q . K_i over active slots (keys are unit norm, so these
//! are cosines); weights w = softmax(beta * s); read vector r = sum_i w_i V_i.
//! Decode: each value position is cleaned up to the nearest row of the core
//! byte embedding (`core.embed`); NUL's row terminates the value.
//! Nothing here looks up a string: the answer is reconstructed from tensors.

use zerolm_tensor::{dot, Matrix};

/// Weights below this are skipped in the value read (deterministically).
pub const MIN_WEIGHT: f32 = 1e-7;

#[derive(Debug, Clone, PartialEq)]
pub struct Attention {
    /// Highest-scoring slot.
    pub best: Option<usize>,
    pub score: f32,
    /// Second-highest score (or -1 when fewer than two slots).
    pub runner_up: f32,
    /// Softmax weight of the best slot.
    pub weight: f32,
    /// (slot, weight) for every slot with weight >= MIN_WEIGHT.
    pub weights: Vec<(usize, f32)>,
}

/// Attention over the given candidate slots.
pub fn attend(keys: &Matrix, slots: &[usize], q: &[f32], beta: f32) -> Attention {
    let scores: Vec<f32> = slots.iter().map(|&i| dot(keys.row(i), q)).collect();
    let mut best = None;
    let (mut top, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (j, &s) in scores.iter().enumerate() {
        if s > top {
            second = top;
            top = s;
            best = Some(j);
        } else if s > second {
            second = s;
        }
    }
    let Some(bj) = best else {
        return Attention { best: None, score: -1.0, runner_up: -1.0, weight: 0.0, weights: Vec::new() };
    };
    let exps: Vec<f32> = scores.iter().map(|s| (beta * (s - top)).exp()).collect();
    let z: f32 = exps.iter().sum();
    let weights: Vec<(usize, f32)> =
        slots.iter().zip(&exps).map(|(&i, e)| (i, e / z)).filter(|(_, w)| *w >= MIN_WEIGHT).collect();
    Attention {
        best: Some(slots[bj]),
        score: top,
        runner_up: if second.is_finite() { second } else { -1.0 },
        weight: exps[bj] / z,
        weights,
    }
}

/// r = sum_i w_i V_i
pub fn read(values: &Matrix, weights: &[(usize, f32)]) -> Vec<f32> {
    let mut r = vec![0.0f32; values.cols];
    for &(i, w) in weights {
        r.iter_mut().zip(values.row(i)).for_each(|(a, v)| *a += w * v);
    }
    r
}

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
        let mut values = Matrix::zeros(2, a.len());
        values.data[..a.len()].copy_from_slice(&a);
        values.data[a.len()..].copy_from_slice(&b);
        let r = read(&values, &[(0, 0.9), (1, 0.1)]);
        assert_eq!(decode(&r, &codes).bytes, b"blue");
    }

    #[test]
    fn attention_picks_the_most_similar_key() {
        let mut keys = Matrix::zeros(3, 2);
        keys.data = vec![1.0, 0.0, 0.0, 1.0, 0.6, 0.8];
        let a = attend(&keys, &[0, 1, 2], &[0.0, 1.0], 32.0);
        assert_eq!(a.best, Some(1));
        assert!((a.score - 1.0).abs() < 1e-6 && (a.runner_up - 0.8).abs() < 1e-6);
        assert!(a.weight > 0.99);
        assert_eq!(attend(&keys, &[], &[0.0, 1.0], 32.0).best, None);
    }
}
