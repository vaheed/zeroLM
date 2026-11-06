//! Online next-byte predictor with explicit local update rules.
//!
//! Protocol per observed byte (strictly chronological):
//! 1. predict a distribution from features of the bytes consumed so far;
//! 2. reveal the actual byte (the only learning signal: `observed_next_byte`);
//! 3. if learning is enabled, apply the configured local rule to the readout;
//! 4. log the update event; then consume the byte into the recurrent state.
//!
//! The embedding and GRU core are frozen (seeded, never updated). Only `readout.weight` and `readout.bias` change. No gradient is
//! propagated through the recurrence or across time: each rule uses only the
//! presynaptic features (or their eligibility trace) and the postsynaptic
//! target/error at the current step.
//!
//! Limitation stated plainly: the normalized delta rule on a softmax readout
//! equals one step of the cross-entropy gradient *for that single layer*.
//! It is local (outer product of error and input), not backpropagation
//! through layers or time, but it is not gradient-free in the strict sense.
//! The Hebbian rule uses no prediction error at all.

use serde::{Deserialize, Serialize};
use zerolm_engine::bootstrap::{CELL_BUDGET, DEFAULT_KIND, EMBED_DIM};
use zerolm_engine::{Model, VOCAB};
use zerolm_tensor::sha256::Sha256;
use zerolm_tensor::{all_finite, softmax_inplace, Matrix};

pub const SIGNAL: &str = "observed_next_byte";
pub const CHANGED_TENSORS: [&str; 2] = ["readout.weight", "readout.bias"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    /// No updates ever: the frozen initialized baseline.
    Frozen,
    /// Normalized delta rule: dW = lr (onehot(y) - p) phi^T / (|phi|^2 + eps).
    Delta,
    /// Target-clamped Hebbian rule on an eligibility trace of the features,
    /// with multiplicative decay; prediction by cosine similarity to the
    /// accumulated prototypes plus a log decayed-count prior. Uses no
    /// prediction error.
    Hebbian,
}

impl Rule {
    pub const ALL: [Rule; 3] = [Rule::Frozen, Rule::Delta, Rule::Hebbian];

    pub fn name(self) -> &'static str {
        match self {
            Rule::Frozen => "frozen",
            Rule::Delta => "delta",
            Rule::Hebbian => "hebbian",
        }
    }

    pub fn parse(s: &str) -> Option<Rule> {
        Rule::ALL.into_iter().find(|r| r.name() == s)
    }
}

/// Rule constants (frozen before the main experiment in
/// `model/specs/online-learning.json`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constants {
    pub delta_lr: f32,
    pub delta_bias_lr: f32,
    pub delta_eps: f32,
    /// Hard bound on the Frobenius norm of one delta-rule update.
    pub max_update_norm: f32,
    pub hebb_lr: f32,
    /// Fraction of every Hebbian weight lost per event.
    pub hebb_decay: f32,
    /// Eligibility-trace persistence (0 = current features only).
    pub hebb_trace: f32,
    /// Logit scale applied to cosine similarity.
    pub hebb_gain: f32,
}

impl Default for Constants {
    fn default() -> Self {
        Constants {
            delta_lr: 1.0,
            delta_bias_lr: 0.05,
            delta_eps: 1e-3,
            max_update_norm: 4.0,
            hebb_lr: 0.01,
            hebb_decay: 0.0005,
            hebb_trace: 0.0,
            hebb_gain: 4.0,
        }
    }
}

/// One logged update (with learning disabled, nothing is logged).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct UpdateEvent {
    pub seq: usize,
    /// Index of the target byte in the input stream; the prediction was made
    /// from exactly the bytes `[0, stream_offset)`.
    pub stream_offset: usize,
    pub target: u8,
    pub predicted: u8,
    pub p_target: f32,
    pub update_norm: f32,
}

/// Block record with the fields required by `update-policy.json`. The full
/// weight hash is taken at block boundaries (hashing every byte would cost
/// more than the update itself); `targets_sha256` pins every target byte.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CheckpointRecord {
    pub input_stream: String,
    pub signal: &'static str,
    pub rule: &'static str,
    pub changed_tensors: Vec<&'static str>,
    pub first_seq: usize,
    pub last_seq: usize,
    pub targets_sha256: String,
    pub checkpoint_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Observation {
    pub p_target: f32,
    pub bits: f64,
    pub predicted: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EvalMetrics {
    pub bytes: usize,
    pub bpb: f64,
    pub accuracy: f64,
    /// Expected calibration error of the top-1 prediction (10 equal bins).
    pub ece: f64,
}

#[derive(Debug, PartialEq)]
pub enum LearnError {
    NonFinite(&'static str),
}

pub struct OnlinePredictor {
    pub rule: Rule,
    pub consts: Constants,
    model: Model,
    readout: Matrix,
    bias: Vec<f32>,
    /// Cached Euclidean norm of each readout row (derived from `readout`).
    row_norm: Vec<f32>,
    learning: bool,
    input_stream: String,
    // Session state (not weights).
    state: Vec<f32>,
    phi: Vec<f32>,
    trace: Vec<f32>,
    offset: usize,
    // Logs.
    events: Vec<UpdateEvent>,
    checkpoints: Vec<CheckpointRecord>,
    block_start: usize,
    block_targets: Sha256,
    checkpoint_every: usize,
    scratch: Vec<f32>,
}

/// The byte assumed before the first observed byte (a line boundary).
pub const START_BYTE: u8 = b'\n';

impl OnlinePredictor {
    pub fn new(seed: u64, rule: Rule, consts: Constants, input_stream: &str) -> Self {
        let model = Model::new(DEFAULT_KIND, seed, EMBED_DIM, CELL_BUDGET);
        let d = model.cell.hidden_dim();
        let f = d + EMBED_DIM;
        let mut p = OnlinePredictor {
            rule,
            consts,
            state: model.init_state(),
            model,
            readout: Matrix::zeros(VOCAB, f),
            bias: vec![0.0; VOCAB],
            row_norm: vec![0.0; VOCAB],
            learning: rule != Rule::Frozen,
            input_stream: input_stream.to_string(),
            phi: Vec::new(),
            trace: vec![0.0; f],
            offset: 0,
            events: Vec::new(),
            checkpoints: Vec::new(),
            block_start: 0,
            block_targets: Sha256::new(),
            checkpoint_every: 1000,
            scratch: vec![0.0; VOCAB],
        };
        p.phi = p.features_of(&p.state, START_BYTE);
        p
    }

    pub fn feature_dim(&self) -> usize {
        self.phi.len()
    }

    pub fn param_count(&self) -> usize {
        self.model.param_count() + self.readout.data.len() + self.bias.len()
    }

    pub fn plastic_param_count(&self) -> usize {
        self.readout.data.len() + self.bias.len()
    }

    /// Enable/disable learning. `Frozen` never learns.
    pub fn set_learning(&mut self, on: bool) {
        self.learning = on && self.rule != Rule::Frozen;
    }

    pub fn learning(&self) -> bool {
        self.learning
    }

    pub fn set_checkpoint_every(&mut self, n: usize) {
        self.checkpoint_every = n.max(1);
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn events(&self) -> &[UpdateEvent] {
        &self.events
    }

    pub fn checkpoints(&self) -> &[CheckpointRecord] {
        &self.checkpoints
    }

    /// Current features (a function of consumed bytes only).
    pub fn features(&self) -> &[f32] {
        &self.phi
    }

    /// SHA-256 over every weight (frozen and plastic). Session state excluded.
    pub fn weights_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"zerolm-online-predictor-v1\0");
        h.update(self.model.weights_sha256().as_bytes());
        for (name, data) in [("readout.weight", &self.readout.data[..]), ("readout.bias", &self.bias)] {
            h.update(name.as_bytes());
            h.update(&(data.len() as u64).to_le_bytes());
            h.update_f32(data);
        }
        h.finish_hex()
    }

    /// Features after consuming bytes up to and including `last`:
    /// `[h ; embed(last)]` (GRU hidden state and the last byte's embedding).
    fn features_of(&self, st: &[f32], last: u8) -> Vec<f32> {
        let d = self.model.cell.hidden_dim();
        let mut phi = Vec::with_capacity(self.readout.cols);
        phi.extend_from_slice(&st[..d]);
        phi.extend_from_slice(self.model.embed.row(last as usize));
        phi
    }

    /// Ablation: make the GRU state independent of the past, so features
    /// carry only the last byte. Call before observing anything.
    pub fn sever_recurrence(&mut self) {
        self.model.cell.sever_recurrence();
        self.state = self.model.init_state();
        self.phi = self.features_of(&self.state, START_BYTE);
    }

    fn probs(&self, phi: &[f32], out: &mut [f32]) {
        match self.rule {
            Rule::Hebbian => {
                let pn = fdot(phi, phi).sqrt();
                let prior = self.consts.hebb_lr * 1e-2;
                for (i, o) in out.iter_mut().enumerate() {
                    let cos = fdot(self.readout.row(i), phi) / (self.row_norm[i] * pn + 1e-6);
                    *o = self.consts.hebb_gain * cos + (self.bias[i] + prior).ln();
                }
            }
            _ => {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = fdot(self.readout.row(i), phi) + self.bias[i];
                }
            }
        }
        softmax_inplace(out);
    }

    /// Distribution over the next byte, given only the bytes consumed so far.
    pub fn predict(&self, out: &mut [f32]) {
        self.probs(&self.phi, out);
    }

    /// Observe the next byte: predict, reveal, (maybe) update, then consume.
    pub fn observe(&mut self, byte: u8) -> Result<Observation, LearnError> {
        let mut p = std::mem::take(&mut self.scratch);
        self.predict(&mut p);
        if !all_finite(&p) {
            return Err(LearnError::NonFinite("prediction"));
        }
        let p_target = p[byte as usize];
        let predicted = argmax(&p) as u8;
        let obs = Observation { p_target, bits: -(p_target.max(1e-30) as f64).log2(), predicted };
        if self.learning {
            let norm = self.update(&p, byte);
            if !all_finite(&self.readout.data) || !all_finite(&self.bias) {
                return Err(LearnError::NonFinite("readout"));
            }
            self.events.push(UpdateEvent {
                seq: self.events.len(),
                stream_offset: self.offset,
                target: byte,
                predicted,
                p_target,
                update_norm: norm,
            });
            self.block_targets.update(&[byte]);
            if self.events.len() - self.block_start >= self.checkpoint_every {
                self.close_block();
            }
        }
        self.scratch = p;
        self.model.step(&mut self.state, byte, None).map_err(|_| LearnError::NonFinite("state"))?;
        self.phi = self.features_of(&self.state, byte);
        self.offset += 1;
        Ok(obs)
    }

    /// Observe a whole stream chronologically, one event at a time. Returns
    /// the online (prequential) bits per byte.
    pub fn observe_all(&mut self, bytes: &[u8]) -> Result<f64, LearnError> {
        let mut bits = 0.0;
        for &b in bytes {
            bits += self.observe(b)?.bits;
        }
        Ok(bits / bytes.len().max(1) as f64)
    }

    /// Write the pending checkpoint record (call at the end of a stream).
    pub fn close_block(&mut self) {
        if self.events.len() == self.block_start {
            return;
        }
        let targets = std::mem::replace(&mut self.block_targets, Sha256::new()).finish_hex();
        self.checkpoints.push(CheckpointRecord {
            input_stream: self.input_stream.clone(),
            signal: SIGNAL,
            rule: self.rule.name(),
            changed_tensors: CHANGED_TENSORS.to_vec(),
            first_seq: self.block_start,
            last_seq: self.events.len() - 1,
            targets_sha256: targets,
            checkpoint_sha256: self.weights_sha256(),
        });
        self.block_start = self.events.len();
    }

    /// Apply the rule for target `y` given prediction `p`; returns the
    /// Frobenius norm of the change to `readout.weight`.
    fn update(&mut self, p: &[f32], y: u8) -> f32 {
        let c = self.consts;
        let f = self.phi.len();
        let y = y as usize;
        match self.rule {
            Rule::Frozen => 0.0,
            Rule::Delta => {
                let phi2 = fdot(&self.phi, &self.phi);
                let mut err: Vec<f32> = p.iter().map(|v| -v).collect();
                err[y] += 1.0;
                let err2 = fdot(&err, &err);
                let mut scale = c.delta_lr / (phi2 + c.delta_eps);
                let n = scale * (err2 * phi2).sqrt();
                if n > c.max_update_norm {
                    scale *= c.max_update_norm / n;
                }
                for (i, e) in err.iter().enumerate() {
                    let k = scale * e;
                    for (w, x) in self.readout.data[i * f..(i + 1) * f].iter_mut().zip(&self.phi) {
                        *w += k * x;
                    }
                    self.bias[i] += c.delta_bias_lr * e;
                }
                scale * (err2 * phi2).sqrt()
            }
            Rule::Hebbian => {
                for (t, x) in self.trace.iter_mut().zip(&self.phi) {
                    *t = c.hebb_trace * *t + (1.0 - c.hebb_trace) * x;
                }
                let tn = fdot(&self.trace, &self.trace).sqrt().max(1e-6);
                let keep = 1.0 - c.hebb_decay;
                let total2: f32 = self.row_norm.iter().map(|n| n * n).sum();
                self.readout.data.iter_mut().for_each(|w| *w *= keep);
                self.bias.iter_mut().for_each(|b| *b *= keep);
                self.row_norm.iter_mut().for_each(|n| *n *= keep);
                let mut hebb2 = 0.0;
                for (w, t) in self.readout.data[y * f..(y + 1) * f].iter_mut().zip(&self.trace) {
                    let dw = c.hebb_lr * t / tn;
                    *w += dw;
                    hebb2 += dw * dw;
                }
                self.bias[y] += c.hebb_lr;
                let row = &self.readout.data[y * f..(y + 1) * f];
                self.row_norm[y] = fdot(row, row).sqrt();
                // Decay part and Hebbian part (upper bound by the triangle inequality).
                c.hebb_decay * total2.sqrt() + hebb2.sqrt()
            }
        }
    }

    /// Held-out evaluation from a fresh session state. Takes `&self`, so it
    /// cannot change any weight: evaluation data never reaches an update.
    pub fn evaluate(&self, bytes: &[u8]) -> Result<EvalMetrics, LearnError> {
        let mut st = self.model.init_state();
        let mut phi = self.features_of(&st, START_BYTE);
        let mut p = vec![0.0; VOCAB];
        let (mut bits, mut correct) = (0.0f64, 0usize);
        let mut bins = [(0.0f64, 0usize); 10];
        for &b in bytes {
            self.probs(&phi, &mut p);
            if !all_finite(&p) {
                return Err(LearnError::NonFinite("prediction"));
            }
            bits -= (p[b as usize].max(1e-30) as f64).log2();
            let top = argmax(&p);
            let hit = top == b as usize;
            correct += hit as usize;
            let conf = p[top] as f64;
            let bin = ((conf * 10.0) as usize).min(9);
            bins[bin].0 += conf;
            bins[bin].1 += hit as usize;
            self.model.step(&mut st, b, None).map_err(|_| LearnError::NonFinite("state"))?;
            phi = self.features_of(&st, b);
        }
        let n = bytes.len().max(1) as f64;
        let ece = bins.iter().map(|&(conf, hits)| (conf - hits as f64).abs()).sum::<f64>() / n;
        Ok(EvalMetrics { bytes: bytes.len(), bpb: bits / n, accuracy: correct as f64 / n, ece })
    }
}

/// Dot product with 8 independent accumulators (deterministic order).
pub fn fdot(a: &[f32], b: &[f32]) -> f32 {
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
    acc.iter().sum::<f32>() + tail
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
