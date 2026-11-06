use crate::{linspace, rec, Cell, Kind, Trace};
use zerolm_tensor::{shuffle, sigmoid, Matrix};

/// Diagonal gated linear recurrence: h_t = a_t * h_{t-1} + (1 - a_t) * tanh(Wv x_t),
/// with input-dependent decay a_t = sigmoid(Wa x_t + ba). Because a_t and the
/// drive depend only on the input, the recurrence is a linear scan and can be
/// evaluated chunk-wise (see `scan_chunked`).
pub struct DiagSsm {
    e: usize,
    d: usize,
    wa: Matrix,
    wv: Matrix,
    ba: Vec<f32>,
}

impl DiagSsm {
    pub fn new(e: usize, d: usize, seed: u64) -> Self {
        DiagSsm {
            e,
            d,
            wa: Matrix::fan_in(d, e, seed, "ssm.wa"),
            wv: Matrix::fan_in(d, e, seed, "ssm.wv"),
            // sigmoid(0..4) = 0.5..0.98 : fast to slow units.
            ba: linspace(0.0, 4.0, d),
        }
    }

    /// Per-token decay `a` and drive `u` (both depend only on x).
    pub fn coeffs(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut a = self.ba.clone();
        self.wa.matvec_acc(x, &mut a);
        a.iter_mut().for_each(|v| *v = sigmoid(*v));
        let mut u = vec![0.0; self.d];
        self.wv.matvec(x, &mut u);
        for (ui, ai) in u.iter_mut().zip(&a) {
            *ui = (1.0 - ai) * ui.tanh();
        }
        (a, u)
    }

    /// Evaluate the recurrence over `xs` by composing affine maps per chunk:
    /// within a chunk, (A, U) with h_out = A*h_in + U. Mathematically equal to
    /// stepping token by token; differs only by float re-association.
    pub fn scan_chunked(&self, h0: &[f32], xs: &[Vec<f32>], chunk: usize) -> Vec<f32> {
        let mut h = h0.to_vec();
        for block in xs.chunks(chunk.max(1)) {
            let mut big_a = vec![1.0f32; self.d];
            let mut big_u = vec![0.0f32; self.d];
            for x in block {
                let (a, u) = self.coeffs(x);
                for i in 0..self.d {
                    big_a[i] *= a[i];
                    big_u[i] = a[i] * big_u[i] + u[i];
                }
            }
            for i in 0..self.d {
                h[i] = big_a[i] * h[i] + big_u[i];
            }
        }
        h
    }
}

impl Cell for DiagSsm {
    fn kind(&self) -> Kind {
        Kind::Ssm
    }
    fn input_dim(&self) -> usize {
        self.e
    }
    fn hidden_dim(&self) -> usize {
        self.d
    }
    fn param_count(&self) -> usize {
        2 * self.d * self.e + self.d
    }
    fn state_len(&self) -> usize {
        self.d
    }
    fn init_state(&self) -> Vec<f32> {
        vec![0.0; self.d]
    }
    fn step(&self, state: &mut [f32], x: &[f32], mut trace: Option<&mut dyn Trace>) {
        let (a, u) = self.coeffs(x);
        for i in 0..self.d {
            state[i] = a[i] * state[i] + u[i];
        }
        rec(&mut trace, "ssm.decay", &a);
        rec(&mut trace, "ssm.drive", &u);
    }
    fn groups(&self) -> Vec<(&'static str, &[f32])> {
        vec![("ssm.wa", &self.wa.data[..]), ("ssm.wv", &self.wv.data), ("ssm.ba", &self.ba)]
    }
    fn groups_mut(&mut self) -> Vec<(&'static str, &mut [f32])> {
        vec![("ssm.wa", &mut self.wa.data[..]), ("ssm.wv", &mut self.wv.data), ("ssm.ba", &mut self.ba)]
    }
    fn sever_recurrence(&mut self) {
        self.ba.iter_mut().for_each(|v| *v = -60.0);
        self.wa.data.fill(0.0);
    }
    fn shuffle_recurrence(&mut self, seed: u64) {
        shuffle(&mut self.wa.data, seed);
        shuffle(&mut self.ba, seed.wrapping_add(1));
    }
    fn trace_points(&self) -> usize {
        2
    }
}
