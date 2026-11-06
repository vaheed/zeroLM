use crate::{linspace, rec, Cell, Kind, Trace};
use zerolm_tensor::{shuffle, sigmoid, Matrix};

/// Compact GRU with multi-timescale update-gate biases.
pub struct Gru {
    e: usize,
    d: usize,
    wz: Matrix,
    wr: Matrix,
    wh: Matrix,
    uz: Matrix,
    ur: Matrix,
    uh: Matrix,
    bz: Vec<f32>,
    br: Vec<f32>,
    bh: Vec<f32>,
}

impl Gru {
    pub fn new(e: usize, d: usize, seed: u64) -> Self {
        Gru {
            e,
            d,
            wz: Matrix::fan_in(d, e, seed, "gru.wz"),
            wr: Matrix::fan_in(d, e, seed, "gru.wr"),
            wh: Matrix::fan_in(d, e, seed, "gru.wh"),
            uz: Matrix::fan_in(d, d, seed, "gru.uz"),
            ur: Matrix::fan_in(d, d, seed, "gru.ur"),
            uh: Matrix::fan_in(d, d, seed, "gru.uh"),
            // z low => unit keeps its state (slow); spread gives many timescales.
            bz: linspace(-4.0, 0.0, d),
            br: vec![0.0; d],
            bh: vec![0.0; d],
        }
    }

    /// One update of hidden vector `h` (length d) in place.
    pub(crate) fn update(&self, h: &mut [f32], x: &[f32], trace: &mut Option<&mut dyn Trace>) {
        let d = self.d;
        let hp = h.to_vec();
        let mut z = self.bz.clone();
        self.wz.matvec_acc(x, &mut z);
        self.uz.matvec_acc(&hp, &mut z);
        let mut r = self.br.clone();
        self.wr.matvec_acc(x, &mut r);
        self.ur.matvec_acc(&hp, &mut r);
        z.iter_mut().for_each(|v| *v = sigmoid(*v));
        r.iter_mut().for_each(|v| *v = sigmoid(*v));
        let rh: Vec<f32> = r.iter().zip(&hp).map(|(a, b)| a * b).collect();
        let mut c = self.bh.clone();
        self.wh.matvec_acc(x, &mut c);
        self.uh.matvec_acc(&rh, &mut c);
        for i in 0..d {
            c[i] = c[i].tanh();
            h[i] = (1.0 - z[i]) * hp[i] + z[i] * c[i];
        }
        rec(trace, "gru.update_gate", &z);
        rec(trace, "gru.reset_gate", &r);
        rec(trace, "gru.candidate", &c);
    }

    pub(crate) fn sever(&mut self) {
        self.bz.iter_mut().for_each(|v| *v = 60.0);
        self.wz.data.fill(0.0);
        self.uz.data.fill(0.0);
        self.ur.data.fill(0.0);
        self.uh.data.fill(0.0);
    }

    pub(crate) fn shuffle_rec(&mut self, seed: u64) {
        shuffle(&mut self.uz.data, seed);
        shuffle(&mut self.uh.data, seed.wrapping_add(1));
    }

    pub(crate) fn count(&self) -> usize {
        3 * self.d * self.e + 3 * self.d * self.d + 3 * self.d
    }

    pub(crate) fn groups_vec(&self) -> Vec<(&'static str, &[f32])> {
        vec![
            ("gru.wz", &self.wz.data[..]),
            ("gru.wr", &self.wr.data),
            ("gru.wh", &self.wh.data),
            ("gru.uz", &self.uz.data),
            ("gru.ur", &self.ur.data),
            ("gru.uh", &self.uh.data),
            ("gru.bz", &self.bz),
            ("gru.br", &self.br),
            ("gru.bh", &self.bh),
        ]
    }

    pub(crate) fn groups_mut_vec(&mut self) -> Vec<(&'static str, &mut [f32])> {
        vec![
            ("gru.wz", &mut self.wz.data[..]),
            ("gru.wr", &mut self.wr.data),
            ("gru.wh", &mut self.wh.data),
            ("gru.uz", &mut self.uz.data),
            ("gru.ur", &mut self.ur.data),
            ("gru.uh", &mut self.uh.data),
            ("gru.bz", &mut self.bz),
            ("gru.br", &mut self.br),
            ("gru.bh", &mut self.bh),
        ]
    }
}

impl Cell for Gru {
    fn kind(&self) -> Kind {
        Kind::Gru
    }
    fn input_dim(&self) -> usize {
        self.e
    }
    fn hidden_dim(&self) -> usize {
        self.d
    }
    fn param_count(&self) -> usize {
        self.count()
    }
    fn state_len(&self) -> usize {
        self.d
    }
    fn init_state(&self) -> Vec<f32> {
        vec![0.0; self.d]
    }
    fn step(&self, state: &mut [f32], x: &[f32], mut trace: Option<&mut dyn Trace>) {
        self.update(state, x, &mut trace);
    }
    fn groups(&self) -> Vec<(&'static str, &[f32])> {
        self.groups_vec()
    }
    fn groups_mut(&mut self) -> Vec<(&'static str, &mut [f32])> {
        self.groups_mut_vec()
    }
    fn sever_recurrence(&mut self) {
        self.sever();
    }
    fn shuffle_recurrence(&mut self, seed: u64) {
        self.shuffle_rec(seed);
    }
    fn trace_points(&self) -> usize {
        3
    }
}
