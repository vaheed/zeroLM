//! Role-filler binding in a complex phasor vector-symbolic space (FHRR).
//!
//! A vector `Hv` has `d` complex components stored as `[re_0..re_d, im_0..im_d]`.
//! Atomic symbols, roles, and relations are unit phasors computed from their
//! name by the compiled `atom-encode` circuit (`zerolm-wsp`), not here.
//! - `bind(a, b)`: elementwise complex product (associative, commutative).
//! - `unbind(a, b)`: elementwise product with the conjugate of `b`; for unit
//!   phasors `unbind(bind(a, b), b) == a` exactly up to rounding.
//! - `bundle`: elementwise sum (superposition of role-filler pairs).
//! - `sim(a, b)`: Re<a, b> / d. Unrelated unit phasors give ~0 (sd 1/sqrt(2d)),
//!   identical ones give 1, a bundle of k pairs matches each pair at ~1.
//! - Numbers use fractional-power encoding: `number(n)_k = exp(i n theta_k)`,
//!   so `bind(number(a), number(b)) == number(a + b)`: addition is binding.
//!   [`Space`] only initializes the two seeded number tables (digit phasors and
//!   the cleanup codebook); encoding and decoding run as compiled circuits
//!   (`number-encode`, `number-decode` in `zerolm-wsp`).
//!
//! Everything is programmed and seeded; nothing here is learned.

use zerolm_tensor::Rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Hv {
    pub d: usize,
    /// `[re; d]` followed by `[im; d]`.
    pub data: Vec<f32>,
}

impl Hv {
    pub fn zeros(d: usize) -> Hv {
        Hv { d, data: vec![0.0; 2 * d] }
    }

    pub fn from_angles(angles: &[f64]) -> Hv {
        let d = angles.len();
        let mut data = vec![0.0f32; 2 * d];
        for (k, a) in angles.iter().enumerate() {
            data[k] = a.cos() as f32;
            data[d + k] = a.sin() as f32;
        }
        Hv { d, data }
    }

    pub fn from_data(d: usize, data: &[f32]) -> Hv {
        assert_eq!(data.len(), 2 * d);
        Hv { d, data: data.to_vec() }
    }

    pub fn re(&self) -> &[f32] {
        &self.data[..self.d]
    }

    pub fn im(&self) -> &[f32] {
        &self.data[self.d..]
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|x| *x == 0.0)
    }

    pub fn add_assign(&mut self, other: &Hv) {
        self.data.iter_mut().zip(&other.data).for_each(|(a, b)| *a += b);
    }

    pub fn scale(&mut self, s: f32) {
        self.data.iter_mut().for_each(|x| *x *= s);
    }
}

fn mul(a: &Hv, b: &Hv, conj_b: bool) -> Hv {
    assert_eq!(a.d, b.d, "dimension mismatch");
    let d = a.d;
    let sign = if conj_b { -1.0 } else { 1.0 };
    let mut out = vec![0.0f32; 2 * d];
    for k in 0..d {
        let (ar, ai) = (a.data[k], a.data[d + k]);
        let (br, bi) = (b.data[k], sign * b.data[d + k]);
        out[k] = ar * br - ai * bi;
        out[d + k] = ar * bi + ai * br;
    }
    Hv { d, data: out }
}

/// Elementwise complex product.
pub fn bind(a: &Hv, b: &Hv) -> Hv {
    mul(a, b, false)
}

/// Elementwise product with the conjugate of `b` (inverse of `bind` for unit phasors).
pub fn unbind(a: &Hv, b: &Hv) -> Hv {
    mul(a, b, true)
}

pub fn bundle(items: &[Hv]) -> Hv {
    let mut s = Hv::zeros(items.first().map(|h| h.d).unwrap_or(0));
    for h in items {
        s.add_assign(h);
    }
    s
}

/// Re<a, b> / d.
pub fn sim(a: &Hv, b: &Hv) -> f32 {
    sim_raw(&a.data, &b.data, a.d)
}

/// `sim` on raw `[re; im]` slices (used against tensor rows).
pub fn sim_raw(a: &[f32], b: &[f32], d: usize) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for ((s, a), b) in acc.iter_mut().zip(x).zip(y) {
            *s += a * b;
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    (acc.iter().sum::<f32>() + tail) / d as f32
}

/// Cosine similarity (scale-invariant `sim`).
pub fn cosine(a: &Hv, b: &Hv) -> f32 {
    let n = (sim(a, a) * sim(b, b)).sqrt();
    if n == 0.0 {
        0.0
    } else {
        sim(a, b) / n
    }
}

/// Seeded number space (fractional-power encoding).
#[derive(Debug, Clone)]
pub struct Space {
    pub seed: u64,
    pub d: usize,
    number_base: Vec<f64>,
}

const TAU: f64 = std::f64::consts::TAU;

impl Space {
    pub fn new(seed: u64, d: usize) -> Space {
        let mut rng = Rng::from_tag(seed, "binding.number_base");
        let number_base = (0..d).map(|_| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * TAU - TAU / 2.0).collect();
        Space { seed, d, number_base }
    }

    /// Fractional-power encoding of an integer.
    pub fn number(&self, n: i64) -> Hv {
        let angles: Vec<f64> = self.number_base.iter().map(|t| (n as f64 * t) % TAU).collect();
        Hv::from_angles(&angles)
    }

    pub fn number_base(&self) -> &[f64] {
        &self.number_base
    }

    /// Digit table of the `number-encode` circuit (`10 places x 2d`): row
    /// `10 q + v` is `number(v 10^q)`.
    pub fn digit_table(&self, places: usize) -> Vec<f32> {
        (0..places).flat_map(|q| (0..10).flat_map(move |v| self.number(v * 10i64.pow(q as u32)).data)).collect()
    }

    /// Cleanup codebook of the `number-decode` circuit (`(2 range + 1) x 2d`):
    /// row `j` is `number(j - range)`.
    pub fn codebook(&self, range: i64) -> Vec<f32> {
        (-range..=range).flat_map(|n| self.number(n).data).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbinding_recovers_the_filler_and_bundles_keep_their_parts() {
        let atom = |tag: &str| {
            let mut rng = Rng::from_tag(1, tag);
            Hv::from_angles(&(0..512).map(|_| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * TAU).collect::<Vec<_>>())
        };
        let (r1, r2, a, b) = (atom("subj"), atom("obj"), atom("tom"), atom("cat"));
        let x = bundle(&[bind(&r1, &a), bind(&r2, &b)]);
        let ua = unbind(&x, &r1);
        assert!(cosine(&ua, &a) > 0.6 && cosine(&ua, &b).abs() < 0.15, "{} {}", cosine(&ua, &a), cosine(&ua, &b));
        assert!((sim(&unbind(&bind(&a, &b), &b), &a) - 1.0).abs() < 1e-4);
        assert!(sim(&a, &b).abs() < 0.15);
        assert!((sim(&x, &bind(&r1, &a)) - 1.0).abs() < 0.15);
    }

    #[test]
    fn number_binding_is_addition() {
        let s = Space::new(1, 512);
        let five = bind(&s.number(3), &s.number(2));
        assert!((sim(&five, &s.number(5)) - 1.0).abs() < 1e-3);
        assert!((sim(&unbind(&s.number(3), &s.number(7)), &s.number(-4)) - 1.0).abs() < 1e-3);
        assert!(sim(&s.number(4), &s.number(5)).abs() < 0.15);
        // Digit products give the numeral's vector (the number-encode identity).
        let digits = s.digit_table(3);
        let row = |q: usize, v: usize| Hv::from_data(512, &digits[(10 * q + v) * 1024..(10 * q + v + 1) * 1024]);
        let n907 = bind(&bind(&row(2, 9), &row(1, 0)), &row(0, 7));
        assert!((sim(&n907, &s.number(907)) - 1.0).abs() < 1e-4);
        let book = s.codebook(10);
        assert_eq!(book.len(), 21 * 1024);
        assert_eq!(&book[..1024], &s.number(-10).data[..]);
    }
}
