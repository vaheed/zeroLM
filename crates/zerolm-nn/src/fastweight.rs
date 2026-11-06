use crate::{linspace, rec, Cell, Kind, Trace};
use zerolm_tensor::{shuffle, sigmoid, Matrix};

/// Delta-rule constants (seed entry `delta-rule-constants`).
pub const DECAY: f32 = 0.995;
pub const LEARNING_RATE: f32 = 0.5;
pub const EPS: f32 = 1e-6;
/// Safety clamp on every associative entry; keeps the state finite.
pub const M_CLAMP: f32 = 8.0;

/// Leaky recurrent state plus an online associative matrix M (d x dk) updated
/// by the normalized delta rule:
///   M <- decay*M + lr * (v - M k) k^T / (eps + k.k)
/// M is part of the per-session state, not of the slow weights.
pub struct FastWeight {
    e: usize,
    d: usize,
    dk: usize,
    wx: Matrix,
    u: Matrix,
    wk: Matrix,
    wq: Matrix,
    wv: Matrix,
    r: Matrix,
    b: Vec<f32>,
    bz: Vec<f32>,
}

impl FastWeight {
    pub fn new(e: usize, d: usize, seed: u64) -> Self {
        let dk = (d / 2).max(2);
        FastWeight {
            e,
            d,
            dk,
            wx: Matrix::fan_in(d, e, seed, "fw.wx"),
            u: Matrix::fan_in(d, d, seed, "fw.u"),
            wk: Matrix::fan_in(dk, e, seed, "fw.wk"),
            wq: Matrix::fan_in(dk, e, seed, "fw.wq"),
            wv: Matrix::fan_in(d, e, seed, "fw.wv"),
            r: Matrix::fan_in(d, d, seed, "fw.r"),
            b: vec![0.0; d],
            bz: linspace(-4.0, 0.0, d),
        }
    }

    pub fn key_dim(&self) -> usize {
        self.dk
    }

    /// Key for input x (what would be used to write/read the memory).
    pub fn key(&self, x: &[f32]) -> Vec<f32> {
        let mut k = vec![0.0; self.dk];
        self.wk.matvec(x, &mut k);
        k.iter_mut().for_each(|v| *v = v.tanh());
        k
    }
}

impl Cell for FastWeight {
    fn kind(&self) -> Kind {
        Kind::FastWeight
    }
    fn input_dim(&self) -> usize {
        self.e
    }
    fn hidden_dim(&self) -> usize {
        self.d
    }
    fn param_count(&self) -> usize {
        // wx + wv + u + r  (d*e, d*e, d*d, d*d) + wk + wq (2*dk*e) + b + bz
        2 * self.d * self.e + 2 * self.d * self.d + 2 * self.dk * self.e + 2 * self.d
    }
    fn state_len(&self) -> usize {
        self.d + self.d * self.dk
    }
    fn init_state(&self) -> Vec<f32> {
        vec![0.0; self.state_len()]
    }
    fn step(&self, state: &mut [f32], x: &[f32], mut trace: Option<&mut dyn Trace>) {
        let (d, dk) = (self.d, self.dk);
        let (h, m) = state.split_at_mut(d);
        // read: r_vec = M q  (uses M before this token's write)
        let mut q = vec![0.0; dk];
        self.wq.matvec(x, &mut q);
        q.iter_mut().for_each(|v| *v = v.tanh());
        let mut readout = vec![0.0; d];
        for (i, o) in readout.iter_mut().enumerate() {
            *o = zerolm_tensor::dot(&m[i * dk..(i + 1) * dk], &q);
        }
        // candidate and leaky update
        let mut c = self.b.clone();
        self.wx.matvec_acc(x, &mut c);
        self.u.matvec_acc(h, &mut c);
        self.r.matvec_acc(&readout, &mut c);
        for i in 0..d {
            let z = sigmoid(self.bz[i]);
            h[i] = (1.0 - z) * h[i] + z * c[i].tanh();
        }
        // write: normalized delta rule
        let k = self.key(x);
        let mut v = vec![0.0; d];
        self.wv.matvec(x, &mut v);
        v.iter_mut().for_each(|t| *t = t.tanh());
        let kk = zerolm_tensor::dot(&k, &k) + EPS;
        let mut err = vec![0.0; d];
        for i in 0..d {
            err[i] = v[i] - zerolm_tensor::dot(&m[i * dk..(i + 1) * dk], &k);
        }
        for i in 0..d {
            for j in 0..dk {
                let t = DECAY * m[i * dk + j] + LEARNING_RATE * err[i] * k[j] / kk;
                m[i * dk + j] = t.clamp(-M_CLAMP, M_CLAMP);
            }
        }
        rec(&mut trace, "fw.read", &readout);
        rec(&mut trace, "fw.error", &err);
    }
    fn groups(&self) -> Vec<(&'static str, &[f32])> {
        vec![
            ("fw.wx", &self.wx.data[..]),
            ("fw.u", &self.u.data),
            ("fw.wk", &self.wk.data),
            ("fw.wq", &self.wq.data),
            ("fw.wv", &self.wv.data),
            ("fw.r", &self.r.data),
            ("fw.b", &self.b),
            ("fw.bz", &self.bz),
        ]
    }
    fn groups_mut(&mut self) -> Vec<(&'static str, &mut [f32])> {
        vec![
            ("fw.wx", &mut self.wx.data[..]),
            ("fw.u", &mut self.u.data),
            ("fw.wk", &mut self.wk.data),
            ("fw.wq", &mut self.wq.data),
            ("fw.wv", &mut self.wv.data),
            ("fw.r", &mut self.r.data),
            ("fw.b", &mut self.b),
            ("fw.bz", &mut self.bz),
        ]
    }
    fn sever_recurrence(&mut self) {
        self.bz.iter_mut().for_each(|v| *v = 60.0);
        self.u.data.fill(0.0);
        self.r.data.fill(0.0);
    }
    fn shuffle_recurrence(&mut self, seed: u64) {
        shuffle(&mut self.u.data, seed);
        shuffle(&mut self.r.data, seed.wrapping_add(1));
    }
    fn trace_points(&self) -> usize {
        2
    }
}
