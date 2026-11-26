//! Plastic copy of the embedding and the selected GRU core (roadmap Phase 9,
//! weight group `plastic_core`), with the readout and the candidate local rule:
//! e-prop (eligibility propagation) with fixed random feedback alignment.
//!
//! Research code for the representation-feasibility gate. The frozen `core.*`
//! tensors of the memory milestone are never touched: this module owns a copy.
//!
//! Per observed byte `y` (strictly chronological, the only learning signal):
//! 1. consume the previous byte: `h_t = GRU(h_{t-1}, embed(prev))`;
//! 2. predict `p_t = softmax(R [h_t ; embed(prev)] + b)` and score `y`;
//! 3. broadcast `l_t = B (1_y - p_t)` with the fixed seeded matrix `B` (d x 256);
//! 4. update each eligibility trace with its recursion
//!    `eps_t = (1 - z_t) eps_{t-1} + d h_t / d W |_{h_{t-1} fixed}`
//!    and change the weight by `eta * l_t[i] * eps_t`;
//! 5. update the readout by the normalized delta rule of the online-learning
//!    milestone.
//!
//! No error flows backward through time or through the transposed readout.
//! The recursion keeps only the diagonal (own-unit) recurrent dependence, and
//! for the reset gate only the own-unit path into the candidate: an
//! approximation by design, not exact RTRL.

#![allow(clippy::needless_range_loop)] // strided index arithmetic over flat tensors reads clearer with explicit indices

use crate::online_predictor::Constants;
use zerolm_engine::bootstrap::{CELL_BUDGET, DEFAULT_KIND, EMBED_DIM};
use zerolm_engine::Model;
use crate::online_predictor::fdot;
use zerolm_tensor::{all_finite, sigmoid, softmax_inplace, Rng};

pub const VOCAB: usize = 256;
pub const START_BYTE: u8 = b'\n';

/// Names of the core tensors, in flat-layout order.
pub const TENSORS: [&str; 10] = ["embed", "gru.wz", "gru.wr", "gru.wh", "gru.uz", "gru.ur", "gru.uh", "gru.bz", "gru.br", "gru.bh"];

/// Embedding and GRU parameters in one flat vector (the layout of the BPTT
/// control, so the two can exchange cores).
#[derive(Debug, Clone, PartialEq)]
pub struct Core {
    pub e: usize,
    pub d: usize,
    pub p: Vec<f32>,
}

impl Core {
    /// Offsets of the ten tensors and the total length.
    pub fn offsets(e: usize, d: usize) -> ([usize; 10], usize) {
        let sizes = [VOCAB * e, d * e, d * e, d * e, d * d, d * d, d * d, d, d, d];
        let mut at = 0;
        let mut o = [0; 10];
        for (i, s) in sizes.iter().enumerate() {
            o[i] = at;
            at += s;
        }
        (o, at)
    }

    /// The seeded initialization shared by every arm.
    pub fn from_seed(seed: u64) -> Self {
        Self::from_seed_budget(seed, CELL_BUDGET)
    }

    /// Same initialization at another cell-parameter budget (the larger core of Gate A).
    pub fn from_seed_budget(seed: u64, budget: usize) -> Self {
        let model = Model::new(DEFAULT_KIND, seed, EMBED_DIM, budget);
        let (e, d) = (EMBED_DIM, model.cell.hidden_dim());
        let (o, total) = Core::offsets(e, d);
        let mut p = vec![0.0f32; total];
        p[o[0]..o[0] + VOCAB * e].copy_from_slice(&model.embed.data);
        for (name, data) in model.cell.groups() {
            let i = TENSORS.iter().position(|t| *t == name).unwrap_or_else(|| panic!("unexpected group {name}"));
            p[o[i]..o[i] + data.len()].copy_from_slice(data);
        }
        Core { e, d, p }
    }

    pub fn range(&self, tensor: usize) -> std::ops::Range<usize> {
        let (o, total) = Core::offsets(self.e, self.d);
        o[tensor]..o.get(tensor + 1).copied().unwrap_or(total)
    }

    pub fn tensor(&self, tensor: usize) -> &[f32] {
        &self.p[self.range(tensor)]
    }

    /// Frobenius norm of `self - other` per tensor.
    pub fn change_norms(&self, other: &Core) -> Vec<f64> {
        (0..10).map(|t| self.tensor(t).iter().zip(other.tensor(t)).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt()).collect()
    }

    /// Initial core plus a seeded random-direction change whose Frobenius norm
    /// per tensor equals `norms` (the validity control of Gate A).
    pub fn perturbed(&self, norms: &[f64], seed: u64) -> Core {
        let mut out = self.clone();
        for (t, &target) in norms.iter().enumerate() {
            let r = self.range(t);
            let mut rng = Rng::new(seed ^ (0x5151 + t as u64));
            let dir: Vec<f32> = r.clone().map(|_| rng.uniform(1.0)).collect();
            let n = dir.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt().max(1e-30);
            for (w, x) in out.p[r].iter_mut().zip(&dir) {
                *w += (*x as f64 / n * target) as f32;
            }
        }
        out
    }

    /// Values of one forward step from `hp` with input byte `prev`.
    pub fn step(&self, hp: &[f32], prev: u8) -> Step {
        let (e, d) = (self.e, self.d);
        let (o, _) = Core::offsets(e, d);
        let x = self.p[o[0] + prev as usize * e..o[0] + (prev as usize + 1) * e].to_vec();
        let mat = |t: usize, cols: usize, v: &[f32], out: &mut [f32]| {
            for (i, acc) in out.iter_mut().enumerate() {
                *acc += fdot(&self.p[o[t] + i * cols..o[t] + (i + 1) * cols], v);
            }
        };
        let mut z = self.p[o[7]..o[7] + d].to_vec();
        let mut r = self.p[o[8]..o[8] + d].to_vec();
        mat(1, e, &x, &mut z);
        mat(4, d, hp, &mut z);
        mat(2, e, &x, &mut r);
        mat(5, d, hp, &mut r);
        z.iter_mut().for_each(|v| *v = sigmoid(*v));
        r.iter_mut().for_each(|v| *v = sigmoid(*v));
        let rh: Vec<f32> = r.iter().zip(hp).map(|(a, b)| a * b).collect();
        let mut c = self.p[o[9]..o[9] + d].to_vec();
        mat(3, e, &x, &mut c);
        mat(6, d, &rh, &mut c);
        c.iter_mut().for_each(|v| *v = v.tanh());
        let h: Vec<f32> = (0..d).map(|i| (1.0 - z[i]) * hp[i] + z[i] * c[i]).collect();
        Step { x, hp: hp.to_vec(), z, r, c, h }
    }
}

/// Cached values of one GRU step.
#[derive(Debug, Clone)]
pub struct Step {
    pub x: Vec<f32>,
    pub hp: Vec<f32>,
    pub z: Vec<f32>,
    pub r: Vec<f32>,
    pub c: Vec<f32>,
    pub h: Vec<f32>,
}

/// Softmax readout over `[h ; embed(prev)]`, updated by the normalized delta rule.
#[derive(Debug, Clone)]
pub struct Readout {
    pub f: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl Readout {
    pub fn zeros(f: usize) -> Self {
        Readout { f, w: vec![0.0; VOCAB * f], b: vec![0.0; VOCAB] }
    }

    pub fn probs(&self, phi: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = (0..VOCAB).map(|k| fdot(&self.w[k * self.f..(k + 1) * self.f], phi) + self.b[k]).collect();
        softmax_inplace(&mut out);
        out
    }

    /// Normalized delta rule (the online-learning milestone's `Rule::Delta`).
    pub fn delta_update(&mut self, phi: &[f32], p: &[f32], y: u8, c: &Constants) {
        let phi2 = fdot(phi, phi);
        let mut err: Vec<f32> = p.iter().map(|v| -v).collect();
        err[y as usize] += 1.0;
        let err2 = fdot(&err, &err);
        let mut scale = c.delta_lr / (phi2 + c.delta_eps);
        let n = scale * (err2 * phi2).sqrt();
        if n > c.max_update_norm {
            scale *= c.max_update_norm / n;
        }
        for (k, e) in err.iter().enumerate() {
            let s = scale * e;
            for (w, x) in self.w[k * self.f..(k + 1) * self.f].iter_mut().zip(phi) {
                *w += s * x;
            }
            self.b[k] += c.delta_bias_lr * e;
        }
    }
}

/// What changes during exposure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plasticity {
    /// Nothing (A0).
    Frozen,
    /// Readout only (A1).
    ReadoutOnly,
    /// Embedding, GRU, and readout by e-prop (A2).
    Eprop,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpropConfig {
    /// Core learning rate `eta`.
    pub eta: f32,
    /// Per-step bound on the Frobenius norm of the core update.
    pub max_core_update: f32,
    /// Seed of the fixed feedback matrix B.
    pub feedback_seed: u64,
}

impl Default for EpropConfig {
    fn default() -> Self {
        EpropConfig { eta: 0.01, max_core_update: 0.05, feedback_seed: 777 }
    }
}

/// One logged update (rule, signal, changed tensors, checkpoint hash are in `Model` level logs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Update {
    pub offset: usize,
    pub target: u8,
    pub p_target: f32,
    pub core_update_norm: f32,
}

pub struct PlasticModel {
    pub core: Core,
    pub readout: Readout,
    pub plasticity: Plasticity,
    pub delta: Constants,
    pub cfg: EpropConfig,
    feedback: Vec<f32>,
    /// Eligibility traces, same layout as the core's GRU tensors (embedding excluded).
    trace: Vec<f32>,
    state: Vec<f32>,
    prev: u8,
    learning: bool,
    offset: usize,
    pub log: Vec<Update>,
    pub log_updates: bool,
}

impl PlasticModel {
    pub fn new(core: Core, plasticity: Plasticity, delta: Constants, cfg: EpropConfig) -> Self {
        let (e, d) = (core.e, core.d);
        let mut rng = Rng::new(cfg.feedback_seed);
        let feedback = (0..d * VOCAB).map(|_| rng.uniform(1.0) / (VOCAB as f32).sqrt()).collect();
        let n_trace = core.p.len() - VOCAB * e;
        PlasticModel {
            readout: Readout::zeros(d + e),
            trace: vec![0.0; n_trace],
            state: vec![0.0; d],
            prev: START_BYTE,
            learning: plasticity != Plasticity::Frozen,
            plasticity,
            delta,
            cfg,
            feedback,
            core,
            offset: 0,
            log: Vec::new(),
            log_updates: false,
        }
    }

    pub fn set_learning(&mut self, on: bool) {
        self.learning = on && self.plasticity != Plasticity::Frozen;
    }

    pub fn reset_state(&mut self) {
        self.state.iter_mut().for_each(|v| *v = 0.0);
        self.prev = START_BYTE;
    }

    /// Replace the readout with a fresh zero readout (the representation probe).
    pub fn fresh_readout(&mut self) {
        self.readout = Readout::zeros(self.core.d + self.core.e);
    }

    pub fn feedback_matrix(&self) -> &[f32] {
        &self.feedback
    }

    /// Run the step for the previous byte and return the readout input and the predicted distribution.
    fn forward(&self) -> (Step, Vec<f32>, Vec<f32>) {
        let s = self.core.step(&self.state, self.prev);
        let mut phi = s.h.clone();
        phi.extend_from_slice(&s.x);
        let p = self.readout.probs(&phi);
        (s, phi, p)
    }

    /// Predict `y`, learn from it if enabled, advance. Returns the bits assigned to `y`.
    pub fn observe(&mut self, y: u8) -> Result<f64, &'static str> {
        let (s, phi, p) = self.forward();
        if !all_finite(&p) {
            return Err("non-finite prediction");
        }
        let bits = -(p[y as usize].max(1e-30) as f64).log2();
        let mut core_norm = 0.0;
        if self.learning {
            if self.plasticity == Plasticity::Eprop {
                core_norm = self.eprop_update(&s, &p, y);
            }
            self.readout.delta_update(&phi, &p, y, &self.delta);
            if !all_finite(&self.readout.w) || !all_finite(&self.core.p) {
                return Err("non-finite weights");
            }
            if self.log_updates {
                self.log.push(Update { offset: self.offset, target: y, p_target: p[y as usize], core_update_norm: core_norm });
            }
        }
        self.state = s.h;
        self.prev = y;
        self.offset += 1;
        Ok(bits)
    }

    /// Mean bits per byte over `bytes`, learning as it goes (prequential).
    pub fn observe_all(&mut self, bytes: &[u8]) -> Result<f64, &'static str> {
        let mut bits = 0.0;
        for &b in bytes {
            bits += self.observe(b)?;
        }
        Ok(bits / bytes.len().max(1) as f64)
    }

    /// Per-position bits of `bytes` from a fresh state; no weight changes.
    pub fn score(&self, bytes: &[u8]) -> Vec<f32> {
        self.score_detail(bytes).into_iter().map(|d| d[0]).collect()
    }

    /// Per position `[bits, confidence (max probability), 1.0 if the argmax is the byte else 0.0]`.
    pub fn score_detail(&self, bytes: &[u8]) -> Vec<[f32; 3]> {
        let mut h = vec![0.0f32; self.core.d];
        let mut prev = START_BYTE;
        let mut out = Vec::with_capacity(bytes.len());
        for &y in bytes {
            let s = self.core.step(&h, prev);
            let mut phi = s.h.clone();
            phi.extend_from_slice(&s.x);
            let p = self.readout.probs(&phi);
            let (am, conf) = p.iter().enumerate().fold((0usize, 0.0f32), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
            out.push([-(p[y as usize].max(1e-30)).log2(), conf, (am == y as usize) as u8 as f32]);
            h = s.h;
            prev = y;
        }
        out
    }

    pub fn evaluate(&self, bytes: &[u8]) -> f64 {
        let b = self.score(bytes);
        b.iter().map(|x| *x as f64).sum::<f64>() / b.len().max(1) as f64
    }

    /// Direct (one-step, `h_{t-1}` fixed) derivatives `d h_i / d W_ij` for every GRU weight,
    /// in the flat order of the core without the embedding.
    pub fn direct_terms(core: &Core, s: &Step) -> Vec<f32> {
        let mut out = vec![0.0f32; core.p.len() - VOCAB * core.e];
        Self::advance_traces(&mut out, core, s);
        out
    }

    /// Advance every trace by its recursion `eps <- (1 - z_i) eps + d h_i / d W_ij`.
    pub fn advance_traces(trace: &mut [f32], core: &Core, s: &Step) {
        let (e, d) = (core.e, core.d);
        let (o, _) = Core::offsets(e, d);
        let base = VOCAB * e;
        let uh_diag: Vec<f32> = (0..d).map(|i| core.p[o[6] + i * d + i]).collect();
        let rh: Vec<f32> = s.r.iter().zip(&s.hp).map(|(a, b)| a * b).collect();
        let mut axpy = |t: usize, i: usize, cols: usize, keep: f32, g: f32, v: &[f32]| {
            let at = o[t] - base + i * cols;
            for (w, x) in trace[at..at + cols].iter_mut().zip(v) {
                *w = keep * *w + g * x;
            }
        };
        let one = [1.0f32];
        for i in 0..d {
            let (z, r, c, hp) = (s.z[i], s.r[i], s.c[i], s.hp[i]);
            let keep = 1.0 - z;
            let gz = (c - hp) * z * (1.0 - z);
            let gc = z * (1.0 - c * c);
            let gr = gc * uh_diag[i] * hp * r * (1.0 - r);
            axpy(1, i, e, keep, gz, &s.x);
            axpy(2, i, e, keep, gr, &s.x);
            axpy(3, i, e, keep, gc, &s.x);
            axpy(4, i, d, keep, gz, &s.hp);
            axpy(5, i, d, keep, gr, &s.hp);
            axpy(6, i, d, keep, gc, &rh);
            axpy(7, i, 1, keep, gz, &one);
            axpy(8, i, 1, keep, gr, &one);
            axpy(9, i, 1, keep, gc, &one);
        }
    }

    fn eprop_update(&mut self, s: &Step, p: &[f32], y: u8) -> f32 {
        let (e, d) = (self.core.e, self.core.d);
        let (o, _) = Core::offsets(e, d);
        let base = VOCAB * e;
        Self::advance_traces(&mut self.trace, &self.core, s);
        // Learning signal l = B (1_y - p).
        let mut l = vec![0.0f32; d];
        for (i, li) in l.iter_mut().enumerate() {
            let row = &self.feedback[i * VOCAB..(i + 1) * VOCAB];
            *li = -fdot(row, p) + row[y as usize];
        }
        // GRU weights: delta W_ij = eta * l_i * eps_ij (applied after the norm bound below).
        let mut sq = 0.0f64;
        for t in 1..10 {
            let cols = match t {
                1..=3 => e,
                4..=6 => d,
                _ => 1,
            };
            for i in 0..d {
                let at = o[t] - base + i * cols;
                let k = self.cfg.eta * l[i];
                sq += (k as f64).powi(2) * self.trace[at..at + cols].iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
            }
        }
        // Embedding row of the input byte: one-step signal l . d h / d x (no trace is carried for a shared input).
        let mut dx = vec![0.0f32; e];
        let (z, r, c, hp) = (&s.z, &s.r, &s.c, &s.hp);
        let (wz, wr, wh) = (&self.core.p[o[1]..o[1] + d * e], &self.core.p[o[2]..o[2] + d * e], &self.core.p[o[3]..o[3] + d * e]);
        let uh = &self.core.p[o[6]..o[6] + d * d];
        for i in 0..d {
            let gz = (c[i] - hp[i]) * z[i] * (1.0 - z[i]);
            let gc = z[i] * (1.0 - c[i] * c[i]);
            let gr = gc * uh[i * d + i] * hp[i] * r[i] * (1.0 - r[i]);
            for j in 0..e {
                dx[j] += self.cfg.eta * l[i] * (gz * wz[i * e + j] + gr * wr[i * e + j] + gc * wh[i * e + j]);
            }
        }
        sq += dx.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        let norm = sq.sqrt() as f32;
        let scale = if norm > self.cfg.max_core_update { self.cfg.max_core_update / norm } else { 1.0 };
        for t in 1..10 {
            let cols = match t {
                1..=3 => e,
                4..=6 => d,
                _ => 1,
            };
            for i in 0..d {
                let at = o[t] - base + i * cols;
                let k = scale * self.cfg.eta * l[i];
                for (w, x) in self.core.p[o[t] + i * cols..o[t] + (i + 1) * cols].iter_mut().zip(&self.trace[at..at + cols]) {
                    *w += k * x;
                }
            }
        }
        let row = self.prev as usize * e;
        for j in 0..e {
            self.core.p[row + j] += scale * dx[j];
        }
        norm.min(self.cfg.max_core_update)
    }

    /// Diagnostic: the summed e-prop direction `sum_t l_t eps_t` over `bytes` (fresh state, current
    /// weights, no change to any weight) for the GRU tensors, with the given feedback matrix
    /// (d x 256). Used to compare against exact gradients on a tiny network.
    pub fn eprop_direction(&self, bytes: &[u8], feedback: &[f32]) -> Vec<f32> {
        let (e, d) = (self.core.e, self.core.d);
        let (o, _) = Core::offsets(e, d);
        let base = VOCAB * e;
        let mut trace = vec![0.0f32; self.trace.len()];
        let mut acc = vec![0.0f32; self.trace.len()];
        let (mut h, mut prev) = (vec![0.0f32; d], START_BYTE);
        for &y in bytes {
            let s = self.core.step(&h, prev);
            let mut phi = s.h.clone();
            phi.extend_from_slice(&s.x);
            let p = self.readout.probs(&phi);
            Self::advance_traces(&mut trace, &self.core, &s);
            for i in 0..d {
                let row = &feedback[i * VOCAB..(i + 1) * VOCAB];
                let li = -fdot(row, &p) + row[y as usize];
                for t in 1..10 {
                    let cols = match t {
                        1..=3 => e,
                        4..=6 => d,
                        _ => 1,
                    };
                    let at = o[t] - base + i * cols;
                    for k in 0..cols {
                        acc[at + k] += li * trace[at + k];
                    }
                }
            }
            h = s.h;
            prev = y;
        }
        acc
    }

    /// Current eligibility traces (tests).
    pub fn traces(&self) -> &[f32] {
        &self.trace
    }

    pub fn state(&self) -> &[f32] {
        &self.state
    }
}
