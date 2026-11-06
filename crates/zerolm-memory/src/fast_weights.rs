//! `W_FAST`: a short-lived associative matrix (tensor `memory.fast`).
//!
//! Keys first pass a frozen pattern-separation layer (tensor
//! `core.fast_separator`, seeded): s = kWTA(P k), the `active` largest
//! entries of P k set to 1/sqrt(active), the rest 0. Sparse codes of
//! correlated keys overlap far less than the keys themselves.
//!
//! Write (normalized delta rule with decay):
//!   M <- decay * M;  M <- M + lr * (v - M s) s^T / (s.s + eps)
//! Read: v_hat = M s(q). Associations are superposed in one matrix, so they
//! interfere and fade; capacity is measured, not assumed.

use zerolm_tensor::{dot, Matrix};

pub const TENSOR: &str = "memory.fast";
pub const SEPARATOR: &str = "core.fast_separator";
pub const EPS: f32 = 1e-6;

/// k-winners-take-all sparse code of `p * k` (ties broken by index).
pub fn separate(p: &Matrix, k: &[f32], active: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; p.rows];
    p.matvec(k, &mut h);
    let mut idx: Vec<usize> = (0..h.len()).collect();
    idx.sort_by(|&a, &b| h[b].partial_cmp(&h[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let mut s = vec![0.0f32; h.len()];
    let a = active.clamp(1, h.len());
    for &i in &idx[..a] {
        s[i] = 1.0 / (a as f32).sqrt();
    }
    s
}

#[derive(Debug, Clone, PartialEq)]
pub struct FastWeights {
    pub m: Matrix,
    pub decay: f32,
    pub lr: f32,
}

impl FastWeights {
    pub fn new(value_dim: usize, key_dim: usize, decay: f32, lr: f32) -> Self {
        FastWeights { m: Matrix::zeros(value_dim, key_dim), decay, lr }
    }

    pub fn read(&self, q: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.m.rows];
        self.m.matvec(q, &mut out);
        out
    }

    /// Returns the Frobenius norm of the delta-rule increment.
    pub fn write(&mut self, k: &[f32], v: &[f32]) -> f32 {
        assert_eq!(v.len(), self.m.rows);
        self.m.data.iter_mut().for_each(|w| *w *= self.decay);
        let pred = self.read(k);
        let scale = self.lr / (dot(k, k) + EPS);
        let cols = self.m.cols;
        let mut n2 = 0.0f32;
        let k2 = dot(k, k);
        for (r, (vt, pt)) in v.iter().zip(&pred).enumerate() {
            let e = scale * (vt - pt);
            n2 += e * e * k2;
            for (w, x) in self.m.data[r * cols..(r + 1) * cols].iter_mut().zip(k) {
                *w += e * x;
            }
        }
        n2.sqrt()
    }

    pub fn clear(&mut self) {
        self.m.data.iter_mut().for_each(|w| *w = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_write_is_recalled_exactly_and_fades_with_decay() {
        let mut f = FastWeights::new(3, 2, 0.5, 1.0);
        f.write(&[1.0, 0.0], &[1.0, 2.0, 3.0]);
        let r = f.read(&[1.0, 0.0]);
        assert!(r.iter().zip([1.0, 2.0, 3.0]).all(|(a, b)| (a - b).abs() < 1e-4));
        f.write(&[0.0, 1.0], &[0.0, 0.0, 1.0]);
        let r = f.read(&[1.0, 0.0]);
        assert!((r[2] - 1.5).abs() < 1e-4, "decayed by half: {r:?}");
    }
}
