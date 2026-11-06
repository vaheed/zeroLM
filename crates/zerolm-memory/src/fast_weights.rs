//! `W_FAST`: a short-lived associative matrix (tensor `memory.fast`).
//!
//! Write (normalized delta rule with decay):
//!   M <- decay * M;  M <- M + lr * (v - M k) k^T / (k.k + eps)
//! Read: v_hat = M q. Associations are superposed in one matrix, so they
//! interfere and fade; capacity is measured, not assumed.

use zerolm_tensor::{dot, Matrix};

pub const TENSOR: &str = "memory.fast";
pub const EPS: f32 = 1e-6;

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
