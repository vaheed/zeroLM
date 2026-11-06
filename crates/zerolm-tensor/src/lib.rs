//! Minimal CPU tensor kernels. Deliberately simple loops: no SIMD, no BLAS,
//! no unsafe. Add acceleration only after profiling proves it useful.

pub mod sha256;

/// FNV-1a 64-bit hash (stable, used for tags and checksums).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// SplitMix64 generator: integer-only core, so streams are identical on
/// every platform. Float conversion uses exact 24-bit arithmetic.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// Independent stream per (seed, tensor tag).
    pub fn from_tag(seed: u64, tag: &str) -> Self {
        Rng(seed ^ fnv1a(tag.as_bytes()).rotate_left(17))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// Uniform in [-a, a).
    pub fn uniform(&mut self, a: f32) -> f32 {
        (2.0 * self.next_f32() - 1.0) * a
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Deterministic in-place Fisher-Yates shuffle.
pub fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut rng = Rng::new(seed);
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
}

/// Row-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Uniform init in +-`scale`, from a stream named by `tag`.
    pub fn uniform(rows: usize, cols: usize, seed: u64, tag: &str, scale: f32) -> Self {
        let mut rng = Rng::from_tag(seed, tag);
        let data = (0..rows * cols).map(|_| rng.uniform(scale)).collect();
        Matrix { rows, cols, data }
    }

    /// Fan-in scaled init: +-1/sqrt(cols).
    pub fn fan_in(rows: usize, cols: usize, seed: u64, tag: &str) -> Self {
        Self::uniform(rows, cols, seed, tag, 1.0 / (cols.max(1) as f32).sqrt())
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// out = self * x
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot(self.row(r), x);
        }
    }

    /// out += self * x
    pub fn matvec_acc(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        for (r, o) in out.iter_mut().enumerate() {
            *o += dot(self.row(r), x);
        }
    }

    pub fn matmul(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.cols, b.rows);
        let mut c = Matrix::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                for j in 0..b.cols {
                    c.data[i * b.cols + j] += a * b.data[k * b.cols + j];
                }
            }
        }
        c
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax in place.
pub fn softmax_inplace(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
}

/// log(sum(exp(v))) computed stably.
pub fn log_sum_exp(v: &[f32]) -> f32 {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
}

/// RMS normalization in place (no learned gain).
pub fn rms_norm_inplace(v: &mut [f32], eps: f32) {
    let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

pub fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

pub fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Largest elementwise absolute difference.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}
