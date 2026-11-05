//! Tensor-product (role-filler) binding over a tape of rows.
//!
//! The tape `T` is an N x K matrix. A pointer `p` in R^N selects a row. Writing a filler vector `u`
//! to the selected row is the outer product `T += p (x) u`; reading ("unbinding") is `r = p^T T`.
//! With one-hot `p` both are exact.

pub fn unbind(tape: &[f64], p: &[f64], k: usize) -> Vec<f64> {
    let mut r = vec![0.0; k];
    for (i, pi) in p.iter().enumerate() {
        if *pi != 0.0 {
            let row = &tape[i * k..(i + 1) * k];
            for (rv, tv) in r.iter_mut().zip(row) {
                *rv += pi * tv;
            }
        }
    }
    r
}

pub fn bind(tape: &mut [f64], p: &[f64], u: &[f64], k: usize) {
    for (i, pi) in p.iter().enumerate() {
        if *pi != 0.0 {
            let row = &mut tape[i * k..(i + 1) * k];
            for (tv, uv) in row.iter_mut().zip(u) {
                *tv += pi * uv;
            }
        }
    }
}

/// Soft pointer advance: p' = p + g (S p - p), with a gate g in [0, 1].
pub fn advance(p: &[f64], shift: &super::network::Tensor, gate: f64) -> Vec<f64> {
    let sp = shift.matvec(p);
    p.iter().zip(&sp).map(|(a, b)| a + gate * (b - a)).collect()
}
