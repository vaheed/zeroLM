//! CONTROL, never part of the mainline: the selected GRU core with the same
//! seeded initialization as the online-learning predictor, trained by
//! truncated backpropagation through time (BPTT) in a single chronological
//! pass. Embedding, all six GRU matrices and three biases, and the readout
//! (zero-initialized, as in the predictor) are trained by Adam on the
//! next-byte cross entropy. Constitution rule 2 forbids online
//! backpropagation in the mainline; this exists to measure how much a
//! stronger learner extracts from the same architecture and bytes, and is
//! labelled CONTROL wherever it appears.
//!
//! Protocol: bytes arrive in chunks of `window`. Each byte of a chunk is
//! scored (prequential bits) with the weights from before the chunk's
//! update, then one Adam step is taken on the chunk's mean loss. The hidden
//! state is carried across chunks but detached (truncation). Held-out
//! evaluation starts from a fresh state and never updates.

use zerolm_engine::bootstrap::{CELL_BUDGET, DEFAULT_KIND, EMBED_DIM};
use zerolm_engine::Model;
use zerolm_tensor::sigmoid;

pub const VOCAB: usize = 256;
/// The byte assumed before the first observed byte (as in the predictor).
pub const START_BYTE: u8 = b'\n';

/// How the circuit's two tables start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitInit {
    /// `U = V = C`, one seeded unit-row code table `C` (a closed-form induction construction: content-independent).
    Compiled,
    /// `U` and `V` independent seeded unit-row tables (same shapes and norms, no alignment).
    Random,
}

/// The soft induction circuit (novelty experiment N2, `docs/experimental-design.md`): attention over the last
/// `window` positions, scoring candidate `j` for the byte at position `n` by
/// `s_j = beta * sum_{i<lags} <U[x_{n-1-i}], V[x_{j-1-i}]>`, a null slot with score `s0`, and a copy
/// distribution `o[k] = sum_j a_j [x_j = k]` added to the logits as `gamma * o[k]`, `gamma = gamma0 + w.h`.
/// Research arm; the host loops here are the reference of what a compiled circuit would have to match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circuit {
    pub rank: usize,
    pub lags: usize,
    pub window: usize,
    pub beta: f32,
    pub s0: f32,
    pub gamma0: f32,
    pub init: CircuitInit,
    /// Learning-rate multiplier of the circuit tensors (`U`, `V`, `beta`, `s0`, `gamma0`).
    pub lr_mult: f32,
    /// Freeze `U`, `V`, `beta`, `s0`, `gamma0`; `w` (the hidden-state gate) still learns.
    pub frozen: bool,
}

impl Circuit {
    pub fn compiled() -> Self {
        Circuit { rank: 16, lags: 3, window: 256, beta: 8.0, s0: 20.0, gamma0: 4.0, init: CircuitInit::Compiled, lr_mult: 1.0, frozen: false }
    }
}

/// How the gated-delta memory tables start (experiment GDM, `docs/gdm-experiment.md`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GdmInit {
    /// Key/query table `U`, value table `V` and decoder `D` are one seeded unit-row code table (a
    /// content-independent n-gram association construction), so a stored value decodes to its own byte.
    Compiled,
    /// `U`, `V` and `D` independent seeded unit-row tables: same shapes and norms, no alignment.
    Random,
}

/// Gated delta-rule fast-weight memory (Gated DeltaNet, Yang et al. 2024), one head, bounded state
/// `S in R^{rank x rank}`. Per input byte `c_t` with history `c_{t-1}, ...`:
/// `k_t = norm(sum_{i<lags} s_i * U[c_{t-1-i}])`, `q_t = norm(sum_{i<lags} s_i * U[c_{t-i}])`, `v_t = V[c_t]`,
/// `alpha_t = sigmoid(a0 + wa.x_t)`, `beta_t = sigmoid(b0 + wb.x_t)`,
/// `S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`, `o_t = S_t q_t`,
/// logits += `gamma * D o_t`. `s_i` are fixed position-tag sign vectors (not parameters).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gdm {
    pub rank: usize,
    pub lags: usize,
    pub init: GdmInit,
    pub a0: f32,
    pub b0: f32,
    pub gamma0: f32,
    /// Learning-rate multiplier of the memory tensors.
    pub lr_mult: f32,
    /// Memory mechanism disabled: its output is not added to the logits and its tensors never change.
    pub off: bool,
}

impl Gdm {
    pub fn compiled() -> Self {
        Gdm { rank: 32, lags: 3, init: GdmInit::Compiled, a0: 5.0, b0: 0.0, gamma0: 4.0, lr_mult: 1.0, off: false }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub window: usize,
    pub lr: f32,
    pub clip: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub circuit: Option<Circuit>,
    pub gdm: Option<Gdm>,
}

impl Default for Config {
    fn default() -> Self {
        Config { window: 32, lr: 3e-3, clip: 1.0, beta1: 0.9, beta2: 0.999, circuit: None, gdm: None }
    }
}

/// Parameter layout inside one flat vector.
struct Layout {
    e: usize,
    d: usize,
    embed: usize,
    w: [usize; 3],
    u: [usize; 3],
    b: [usize; 3],
    ro: usize,
    rb: usize,
    /// Circuit tensors (equal to `total` when there is no circuit): U, V, beta, s0, gamma0, then w.
    cu: usize,
    cv: usize,
    cbeta: usize,
    cs0: usize,
    cg0: usize,
    cw: usize,
    /// Gated-delta memory tensors (equal to `total` when absent): U, V, D, a0, b0, gamma0, wa, wb.
    gu: usize,
    gv: usize,
    gd: usize,
    ga0: usize,
    gb0: usize,
    gg0: usize,
    gwa: usize,
    gwb: usize,
    total: usize,
}

impl Layout {
    fn new(e: usize, d: usize, rank: Option<usize>, gdm_rank: Option<usize>) -> Self {
        let mut at = 0;
        let mut take = |n: usize| {
            let o = at;
            at += n;
            o
        };
        let embed = take(VOCAB * e);
        let w = [take(d * e), take(d * e), take(d * e)];
        let u = [take(d * d), take(d * d), take(d * d)];
        let b = [take(d), take(d), take(d)];
        let ro = take(VOCAB * (d + e));
        let rb = take(VOCAB);
        let (cu, cv, cbeta, cs0, cg0, cw) = match rank {
            Some(r) => (take(VOCAB * r), take(VOCAB * r), take(1), take(1), take(1), take(d)),
            None => {
                let end = take(0);
                (end, end, end, end, end, end)
            }
        };
        let (gu, gv, gd, ga0, gb0, gg0, gwa, gwb) = match gdm_rank {
            Some(r) => (take(VOCAB * r), take(VOCAB * r), take(VOCAB * r), take(1), take(1), take(1), take(e), take(e)),
            None => {
                let end = take(0);
                (end, end, end, end, end, end, end, end)
            }
        };
        let at = take(0);
        Layout { e, d, embed, w, u, b, ro, rb, cu, cv, cbeta, cs0, cg0, cw, gu, gv, gd, ga0, gb0, gg0, gwa, gwb, total: at }
    }
}

/// One logged update (constitution invariant 3): which bytes drove it, the signal, the rule, and the size of the step.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateEvent {
    pub seq: usize,
    /// Offset of the chunk's first byte in the exposure stream.
    pub stream_offset: usize,
    pub bytes: usize,
    /// Bits the chunk cost under the weights from before this update.
    pub bits_before: f64,
    pub grad_norm: f32,
    pub update_norm: f32,
}

/// A block of consecutive updates and the weights' hash after the last one.
#[derive(Debug, Clone, PartialEq)]
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

pub const SIGNAL: &str = "observed_next_byte";
pub const RULE: &str = "truncated_bptt_adam";
pub const CHANGED: [&str; 12] = ["embed", "gru.wz", "gru.wr", "gru.wh", "gru.uz", "gru.ur", "gru.uh", "gru.bz", "gru.br", "gru.bh", "readout.weight", "readout.bias"];

struct EventLog {
    stream: String,
    every: usize,
    events: Vec<UpdateEvent>,
    checkpoints: Vec<CheckpointRecord>,
    block_start: usize,
    block_targets: zerolm_tensor::sha256::Sha256,
}

pub struct Bptt {
    learning: bool,
    log: Option<EventLog>,
    cfg: Config,
    l: Layout,
    p: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    steps: u64,
    state: Vec<f32>,
    prev: u8,
    /// Recent bytes (starting with the start byte), kept for the circuit's attention window.
    tail: Vec<u8>,
    /// Total bits and bytes of the prequential pass so far.
    bits: f64,
    seen: usize,
    /// Gated-delta fast-weight state `S` (row-major rank x rank; empty without a memory). State, not a parameter.
    fast: Vec<f32>,
    /// Fixed position-tag sign vectors, `lags x rank` (constants of the construction, not parameters).
    tags: Vec<f32>,
    /// SHA-256 of the initial parameters (initialization provenance, `bptt_snapshot`).
    init_sha256: String,
    /// Seed the model was built with.
    seed: u64,
}

#[path = "bptt_snapshot.rs"]
pub mod snapshot;

/// Cached forward values of one gated-delta memory step.
struct GdStep {
    khat_n: f32,
    qhat_n: f32,
    k: Vec<f32>,
    q: Vec<f32>,
    v: Vec<f32>,
    alpha: f32,
    beta: f32,
    sk: Vec<f32>,
    s_prev: Vec<f32>,
    s_new: Vec<f32>,
    o: Vec<f32>,
    /// False when the key context does not exist yet (first byte): no write happened.
    ok_key: bool,
}

/// Cached forward values of one step.
struct CircStep {
    js: Vec<usize>,
    /// Raw similarity sum per candidate (before `beta`).
    c: Vec<f32>,
    a: Vec<f32>,
    o: Vec<f32>,
    gamma: f32,
}

struct Step {
    circ: Option<CircStep>,
    gd: Option<GdStep>,
    prev: u8,
    hp: Vec<f32>,
    z: Vec<f32>,
    r: Vec<f32>,
    c: Vec<f32>,
    h: Vec<f32>,
    p: Vec<f32>,
}

fn unit_rows(rng: &mut zerolm_tensor::Rng, rank: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(VOCAB * rank);
    for _ in 0..VOCAB {
        let mut row: Vec<f32> = (0..rank).map(|_| rng.uniform(1.0)).collect();
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        row.iter_mut().for_each(|x| *x /= n);
        v.extend(row);
    }
    v
}

fn init_circuit(p: &mut [f32], l: &Layout, c: &Circuit, seed: u64) {
    let n = VOCAB * c.rank;
    let (u, v) = match c.init {
        CircuitInit::Compiled => {
            let code = unit_rows(&mut zerolm_tensor::Rng::from_tag(0, "circuit-code-table-v1"), c.rank);
            (code.clone(), code)
        }
        CircuitInit::Random => (unit_rows(&mut zerolm_tensor::Rng::from_tag(seed, "circuit-random-u"), c.rank), unit_rows(&mut zerolm_tensor::Rng::from_tag(seed, "circuit-random-v"), c.rank)),
    };
    p[l.cu..l.cu + n].copy_from_slice(&u);
    p[l.cv..l.cv + n].copy_from_slice(&v);
    p[l.cbeta] = c.beta;
    p[l.cs0] = c.s0;
    p[l.cg0] = c.gamma0;
    // w starts at zero: the hidden state does not modulate the copy gain until it learns to.
}

fn init_gdm(p: &mut [f32], l: &Layout, g: &Gdm, seed: u64) {
    let n = VOCAB * g.rank;
    let (u, v, d) = match g.init {
        GdmInit::Compiled => {
            let code = unit_rows(&mut zerolm_tensor::Rng::from_tag(0, "gdm-code-table-v1"), g.rank);
            (code.clone(), code.clone(), code)
        }
        GdmInit::Random => (
            unit_rows(&mut zerolm_tensor::Rng::from_tag(seed, "gdm-random-u"), g.rank),
            unit_rows(&mut zerolm_tensor::Rng::from_tag(seed, "gdm-random-v"), g.rank),
            unit_rows(&mut zerolm_tensor::Rng::from_tag(seed, "gdm-random-d"), g.rank),
        ),
    };
    p[l.gu..l.gu + n].copy_from_slice(&u);
    p[l.gv..l.gv + n].copy_from_slice(&v);
    p[l.gd..l.gd + n].copy_from_slice(&d);
    p[l.ga0] = g.a0;
    p[l.gb0] = g.b0;
    p[l.gg0] = g.gamma0;
    // wa, wb start at zero: the gates do not depend on the byte until learning makes them.
}

fn matvec_acc(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        out[r] += row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
    }
}

/// Outer-product accumulate: g[r][c] += a[r] * b[c].
fn outer_acc(g: &mut [f32], cols: usize, a: &[f32], b: &[f32]) {
    for (r, &ar) in a.iter().enumerate() {
        if ar != 0.0 {
            for (gc, &bc) in g[r * cols..(r + 1) * cols].iter_mut().zip(b) {
                *gc += ar * bc;
            }
        }
    }
}

/// out[c] += sum_r w[r][c] * a[r].
fn transpose_acc(w: &[f32], cols: usize, a: &[f32], out: &mut [f32]) {
    for (r, &ar) in a.iter().enumerate() {
        if ar != 0.0 {
            for (o, &wc) in out.iter_mut().zip(&w[r * cols..(r + 1) * cols]) {
                *o += ar * wc;
            }
        }
    }
}

impl Bptt {
    /// The same seeded initialization as `OnlinePredictor::new(seed, ..)`:
    /// the model's embedding and GRU weights, a zero readout and bias.
    pub fn new(seed: u64, cfg: Config) -> Self {
        Self::with_budget(seed, cfg, CELL_BUDGET)
    }

    pub fn with_budget(seed: u64, cfg: Config, budget: usize) -> Self {
        let model = Model::new(DEFAULT_KIND, seed, EMBED_DIM, budget);
        let d = model.cell.hidden_dim();
        let l = Layout::new(EMBED_DIM, d, cfg.circuit.map(|c| c.rank), cfg.gdm.map(|g| g.rank));
        let mut p = vec![0.0f32; l.total];
        p[l.embed..l.embed + VOCAB * EMBED_DIM].copy_from_slice(&model.embed.data);
        for (name, data) in model.cell.groups() {
            let at = match name {
                "gru.wz" => l.w[0],
                "gru.wr" => l.w[1],
                "gru.wh" => l.w[2],
                "gru.uz" => l.u[0],
                "gru.ur" => l.u[1],
                "gru.uh" => l.u[2],
                "gru.bz" => l.b[0],
                "gru.br" => l.b[1],
                "gru.bh" => l.b[2],
                other => panic!("unexpected GRU group {other}"),
            };
            p[at..at + data.len()].copy_from_slice(data);
        }
        if let Some(c) = cfg.circuit {
            init_circuit(&mut p, &l, &c, seed);
        }
        let (fast, tags) = match cfg.gdm {
            Some(g) => {
                init_gdm(&mut p, &l, &g, seed);
                let mut rng = zerolm_tensor::Rng::from_tag(0, "gdm-position-tags-v1");
                (vec![0.0f32; g.rank * g.rank], (0..g.lags * g.rank).map(|_| if rng.uniform(1.0) >= 0.0 { 1.0 } else { -1.0 }).collect())
            }
            None => (Vec::new(), Vec::new()),
        };
        let init_sha256 = {
            let mut h = zerolm_tensor::sha256::Sha256::new();
            h.update(b"zerolm-gradient-core-init-v1\0");
            h.update_f32(&p);
            h.finish_hex()
        };
        Bptt { init_sha256, seed, learning: true, log: None, tail: vec![START_BYTE], cfg, m: vec![0.0; l.total], v: vec![0.0; l.total], steps: 0, state: vec![0.0; d], prev: START_BYTE, bits: 0.0, seen: 0, fast, tags, l, p }
    }

    /// The embedding and GRU tensors as a plastic core (same flat layout).
    pub fn core(&self) -> zerolm_learning::plastic::Core {
        zerolm_learning::plastic::Core { e: self.l.e, d: self.l.d, p: self.p[..self.l.ro].to_vec() }
    }

    /// Readout weights and bias (zero-initialized, trained here).
    pub fn readout(&self) -> (&[f32], &[f32]) {
        (&self.p[self.l.ro..self.l.rb], &self.p[self.l.rb..self.l.total])
    }

    /// Indices of the circuit tensors (U, V, beta, s0, gamma0, w); empty without a circuit.
    pub fn circuit_param_range(&self) -> std::ops::Range<usize> {
        self.l.cu..self.l.total
    }

    /// Indices of the gated-delta memory tensors (U, V, D, a0, b0, gamma0, wa, wb); empty without a memory.
    pub fn gdm_param_range(&self) -> std::ops::Range<usize> {
        self.l.gu..self.l.total
    }

    /// Offsets inside the memory range: (V, D, scalars a0/b0/gamma0, wa, wb).
    pub fn gdm_offsets(&self) -> [usize; 5] {
        [self.l.gv, self.l.gd, self.l.ga0, self.l.gwa, self.l.gwb]
    }

    /// Parameters that can change: all, minus the memory tensors when the memory is off.
    pub fn effective_param_count(&self) -> usize {
        if self.cfg.gdm.is_some_and(|g| g.off) { self.l.gu } else { self.l.total }
    }

    /// Length of the persistent fast-weight state in floats (0 without a memory).
    pub fn fast_state_len(&self) -> usize {
        self.fast.len()
    }

    pub fn param_count(&self) -> usize {
        self.l.total
    }

    /// Circuit attention for the byte after `hist` (all earlier bytes, last = the input byte).
    fn circuit_forward(&self, params: &[f32], c: &Circuit, hist: &[u8]) -> CircStep {
        let (r, m, n) = (c.rank, c.lags, hist.len());
        let mut o = vec![0.0f32; VOCAB];
        if n < m + 1 {
            return CircStep { js: vec![], c: vec![], a: vec![], o, gamma: 0.0 };
        }
        let (beta, s0) = (params[self.l.cbeta], params[self.l.cs0]);
        let lo = m.max(n.saturating_sub(c.window));
        let js: Vec<usize> = (lo..n).collect();
        let mut sims = Vec::with_capacity(js.len());
        for &j in &js {
            let mut acc = 0.0f32;
            for i in 0..m {
                let (qb, kb) = (hist[n - 1 - i] as usize, hist[j - 1 - i] as usize);
                let (u, v) = (&params[self.l.cu + qb * r..self.l.cu + (qb + 1) * r], &params[self.l.cv + kb * r..self.l.cv + (kb + 1) * r]);
                acc += u.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
            }
            sims.push(acc);
        }
        let mx = sims.iter().map(|x| beta * x).fold(s0, f32::max);
        let mut a: Vec<f32> = sims.iter().map(|x| (beta * x - mx).exp()).collect();
        let z: f32 = a.iter().sum::<f32>() + (s0 - mx).exp();
        a.iter_mut().for_each(|x| *x /= z);
        for (&j, &aj) in js.iter().zip(&a) {
            o[hist[j] as usize] += aj;
        }
        CircStep { js, c: sims, a, o, gamma: 0.0 }
    }

    /// Position-tagged bundle of the code rows of `hist[end], hist[end-1], ...` (`lags` terms), unit-normalised.
    fn bundle(&self, params: &[f32], g: &Gdm, hist: &[u8], end: usize) -> (Vec<f32>, f32) {
        let r = g.rank;
        let mut hat = vec![0.0f32; r];
        for i in 0..g.lags.min(end + 1) {
            let code = &params[self.l.gu + hist[end - i] as usize * r..][..r];
            for j in 0..r {
                hat[j] += self.tags[i * r + j] * code[j];
            }
        }
        let nrm = (hat.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        hat.iter_mut().for_each(|x| *x /= nrm);
        (hat, nrm)
    }

    fn gdm_forward(&self, params: &[f32], g: &Gdm, s: &[f32], hist: &[u8], x: &[f32]) -> GdStep {
        let (r, n) = (g.rank, hist.len());
        let (q, qhat_n) = self.bundle(params, g, hist, n - 1);
        let ok_key = n >= 2;
        let (k, khat_n) = if ok_key { self.bundle(params, g, hist, n - 2) } else { (vec![0.0; r], 1.0) };
        let cur = hist[n - 1] as usize;
        let v = params[self.l.gv + cur * r..][..r].to_vec();
        let dot = |w: &[f32]| w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
        let alpha = sigmoid(params[self.l.ga0] + dot(&params[self.l.gwa..self.l.gwa + self.l.e]));
        let beta = sigmoid(params[self.l.gb0] + dot(&params[self.l.gwb..self.l.gwb + self.l.e]));
        let mut sk = vec![0.0f32; r];
        for i in 0..r {
            sk[i] = s[i * r..(i + 1) * r].iter().zip(&k).map(|(a, b)| a * b).sum();
        }
        let mut s_new = vec![0.0f32; r * r];
        for i in 0..r {
            for j in 0..r {
                s_new[i * r + j] = alpha * (s[i * r + j] - beta * sk[i] * k[j]) + beta * v[i] * k[j];
            }
        }
        let o: Vec<f32> = (0..r).map(|i| s_new[i * r..(i + 1) * r].iter().zip(&q).map(|(a, b)| a * b).sum()).collect();
        GdStep { khat_n, qhat_n, k, q, v, alpha, beta, sk, s_prev: s.to_vec(), s_new, o, ok_key }
    }

    fn forward_step(&self, params: &[f32], h: &[f32], s: &[f32], hist: &[u8]) -> Step {
        let prev = *hist.last().expect("history starts with the start byte");
        let (e, d) = (self.l.e, self.l.d);
        let x = &params[self.l.embed + prev as usize * e..self.l.embed + (prev as usize + 1) * e];
        let mut z = params[self.l.b[0]..self.l.b[0] + d].to_vec();
        let mut r = params[self.l.b[1]..self.l.b[1] + d].to_vec();
        matvec_acc(&params[self.l.w[0]..], d, e, x, &mut z);
        matvec_acc(&params[self.l.u[0]..], d, d, h, &mut z);
        matvec_acc(&params[self.l.w[1]..], d, e, x, &mut r);
        matvec_acc(&params[self.l.u[1]..], d, d, h, &mut r);
        z.iter_mut().for_each(|v| *v = sigmoid(*v));
        r.iter_mut().for_each(|v| *v = sigmoid(*v));
        let rh: Vec<f32> = r.iter().zip(h).map(|(a, b)| a * b).collect();
        let mut c = params[self.l.b[2]..self.l.b[2] + d].to_vec();
        matvec_acc(&params[self.l.w[2]..], d, e, x, &mut c);
        matvec_acc(&params[self.l.u[2]..], d, d, &rh, &mut c);
        c.iter_mut().for_each(|v| *v = v.tanh());
        let hn: Vec<f32> = (0..d).map(|i| (1.0 - z[i]) * h[i] + z[i] * c[i]).collect();
        // Readout over [h, embed(prev)].
        let f = d + e;
        let mut logits = params[self.l.rb..self.l.rb + VOCAB].to_vec();
        for (k, lg) in logits.iter_mut().enumerate() {
            let row = &params[self.l.ro + k * f..self.l.ro + (k + 1) * f];
            *lg += row[..d].iter().zip(&hn).map(|(a, b)| a * b).sum::<f32>() + row[d..].iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
        }
        let circ = self.cfg.circuit.map(|c| {
            let mut cs = self.circuit_forward(params, &c, hist);
            cs.gamma = params[self.l.cg0] + params[self.l.cw..self.l.cw + d].iter().zip(&hn).map(|(a, b)| a * b).sum::<f32>();
            for (lg, ok) in logits.iter_mut().zip(&cs.o) {
                *lg += cs.gamma * ok;
            }
            cs
        });
        let gd = match self.cfg.gdm {
            Some(g) if !g.off => {
                let st = self.gdm_forward(params, &g, s, hist, x);
                let gamma = params[self.l.gg0];
                for (kb, lg) in logits.iter_mut().enumerate() {
                    *lg += gamma * params[self.l.gd + kb * g.rank..][..g.rank].iter().zip(&st.o).map(|(a, b)| a * b).sum::<f32>();
                }
                Some(st)
            }
            _ => None,
        };
        zerolm_tensor::softmax_inplace(&mut logits);
        Step { circ, gd, prev, hp: h.to_vec(), z, r, c, h: hn, p: logits }
    }

    /// Mean cross entropy (nats) of `bytes` from state `h0`, and the gradient
    /// of that mean with respect to every parameter.
    fn loss_and_grad(&self, params: &[f32], h0: &[f32], s0: &[f32], tail: &[u8], bytes: &[u8]) -> (f64, Vec<f32>, Vec<Step>) {
        let (e, d) = (self.l.e, self.l.d);
        let f = d + e;
        let mut steps: Vec<Step> = Vec::with_capacity(bytes.len());
        let mut h = h0.to_vec();
        let mut fs = s0.to_vec();
        let mut buf = tail.to_vec();
        buf.extend_from_slice(bytes);
        let mut loss = 0.0f64;
        for (t, &y) in bytes.iter().enumerate() {
            let s = self.forward_step(params, &h, &fs, &buf[..tail.len() + t]);
            loss -= (s.p[y as usize].max(1e-30) as f64).ln();
            h = s.h.clone();
            if let Some(gd) = &s.gd {
                fs = gd.s_new.clone();
            }
            steps.push(s);
        }
        let n = bytes.len() as f32;
        let mut g = vec![0.0f32; self.l.total];
        let mut dh = vec![0.0f32; d];
        let gr = self.cfg.gdm.map_or(0, |g| g.rank);
        let mut ds = vec![0.0f32; gr * gr];
        for t in (0..bytes.len()).rev() {
            let s = &steps[t];
            let x = &params[self.l.embed + s.prev as usize * e..self.l.embed + (s.prev as usize + 1) * e];
            let mut dlogit = s.p.clone();
            dlogit[bytes[t] as usize] -= 1.0;
            dlogit.iter_mut().for_each(|v| *v /= n);
            // Readout.
            let mut phi = s.h.clone();
            phi.extend_from_slice(x);
            outer_acc(&mut g[self.l.ro..self.l.ro + VOCAB * f], f, &dlogit, &phi);
            for (gb, dl) in g[self.l.rb..self.l.rb + VOCAB].iter_mut().zip(&dlogit) {
                *gb += dl;
            }
            let mut dphi = vec![0.0f32; f];
            transpose_acc(&params[self.l.ro..self.l.ro + VOCAB * f], f, &dlogit, &mut dphi);
            for i in 0..d {
                dh[i] += dphi[i];
            }
            if let (Some(cs), Some(c)) = (&s.circ, &self.cfg.circuit) {
                let hist = &buf[..tail.len() + t];
                let (r, m, n) = (c.rank, c.lags, hist.len());
                let dgamma: f32 = dlogit.iter().zip(&cs.o).map(|(a, b)| a * b).sum();
                g[self.l.cg0] += dgamma;
                for i in 0..d {
                    g[self.l.cw + i] += dgamma * s.h[i];
                    dh[i] += dgamma * params[self.l.cw + i];
                }
                if !cs.js.is_empty() {
                    let da: Vec<f32> = cs.js.iter().map(|&j| cs.gamma * dlogit[hist[j] as usize]).collect();
                    let dot: f32 = cs.a.iter().zip(&da).map(|(a, b)| a * b).sum();
                    let a_null = 1.0 - cs.a.iter().sum::<f32>();
                    g[self.l.cs0] += -a_null * dot;
                    let beta = params[self.l.cbeta];
                    for (idx, &j) in cs.js.iter().enumerate() {
                        let ds = cs.a[idx] * (da[idx] - dot);
                        g[self.l.cbeta] += ds * cs.c[idx];
                        let w = beta * ds;
                        if w == 0.0 {
                            continue;
                        }
                        for i in 0..m {
                            let (qb, kb) = (hist[n - 1 - i] as usize, hist[j - 1 - i] as usize);
                            for k in 0..r {
                                g[self.l.cu + qb * r + k] += w * params[self.l.cv + kb * r + k];
                                g[self.l.cv + kb * r + k] += w * params[self.l.cu + qb * r + k];
                            }
                        }
                    }
                }
            }
            let mut dx: Vec<f32> = dphi[d..].to_vec();
            if let (Some(gs), Some(gcfg)) = (&s.gd, &self.cfg.gdm) {
                let r = gcfg.rank;
                let hist = &buf[..tail.len() + t];
                let n = hist.len();
                let gamma = params[self.l.gg0];
                let mut dgamma = 0.0f32;
                let mut dout = vec![0.0f32; r];
                for kb in 0..VOCAB {
                    let drow = &params[self.l.gd + kb * r..][..r];
                    let dl = dlogit[kb];
                    if dl == 0.0 {
                        continue;
                    }
                    dgamma += dl * drow.iter().zip(&gs.o).map(|(a, b)| a * b).sum::<f32>();
                    for j in 0..r {
                        g[self.l.gd + kb * r + j] += gamma * dl * gs.o[j];
                        dout[j] += gamma * dl * drow[j];
                    }
                }
                g[self.l.gg0] += dgamma;
                // o = S_new q
                let mut dq = vec![0.0f32; r];
                for i in 0..r {
                    for j in 0..r {
                        ds[i * r + j] += dout[i] * gs.q[j];
                        dq[j] += gs.s_new[i * r + j] * dout[i];
                    }
                }
                // S_new = alpha (S - beta (S k) k^T) + beta v k^T
                let (al, be) = (gs.alpha, gs.beta);
                let mut dk = vec![0.0f32; r];
                let mut dv = vec![0.0f32; r];
                let (mut dal, mut dbe) = (0.0f32, 0.0f32);
                let mut dtk = vec![0.0f32; r]; // (dT k)_i with dT = -al*be*dS
                let mut dts_sk = vec![0.0f32; r]; // (dT^T sk)_j
                for i in 0..r {
                    for j in 0..r {
                        let dsij = ds[i * r + j];
                        let tij = gs.sk[i] * gs.k[j];
                        dal += dsij * (gs.s_prev[i * r + j] - be * tij);
                        dbe += dsij * (-al * tij + gs.v[i] * gs.k[j]);
                        let dt = -al * be * dsij;
                        dtk[i] += dt * gs.k[j];
                        dts_sk[j] += dt * gs.sk[i];
                        dk[j] += be * dsij * gs.v[i];
                        dv[i] += be * dsij * gs.k[j];
                    }
                }
                let mut ds_prev = vec![0.0f32; r * r];
                for i in 0..r {
                    for j in 0..r {
                        ds_prev[i * r + j] = al * ds[i * r + j] + dtk[i] * gs.k[j];
                        dk[j] += gs.s_prev[i * r + j] * dtk[i];
                    }
                }
                for j in 0..r {
                    dk[j] += dts_sk[j];
                }
                ds = ds_prev;
                // gates
                let da = dal * al * (1.0 - al);
                let db = dbe * be * (1.0 - be);
                g[self.l.ga0] += da;
                g[self.l.gb0] += db;
                for i in 0..e {
                    g[self.l.gwa + i] += da * x[i];
                    g[self.l.gwb + i] += db * x[i];
                    dx[i] += da * params[self.l.gwa + i] + db * params[self.l.gwb + i];
                }
                // value table
                let cur = hist[n - 1] as usize;
                for j in 0..r {
                    g[self.l.gv + cur * r + j] += dv[j];
                }
                // key / query bundles back to the code table: dhat = (dz - z (z.dz)) / norm
                let mut back = |z: &[f32], dz: &[f32], nrm: f32, end: usize| {
                    let zd: f32 = z.iter().zip(dz).map(|(a, b)| a * b).sum();
                    for i in 0..gcfg.lags.min(end + 1) {
                        let row = self.l.gu + hist[end - i] as usize * r;
                        for j in 0..r {
                            g[row + j] += self.tags[i * r + j] * (dz[j] - z[j] * zd) / nrm;
                        }
                    }
                };
                back(&gs.q, &dq, gs.qhat_n, n - 1);
                if gs.ok_key {
                    back(&gs.k, &dk, gs.khat_n, n - 2);
                }
            }
            // GRU step backward.
            let dz: Vec<f32> = (0..d).map(|i| dh[i] * (s.c[i] - s.hp[i])).collect();
            let dc: Vec<f32> = (0..d).map(|i| dh[i] * s.z[i]).collect();
            let mut dhp: Vec<f32> = (0..d).map(|i| dh[i] * (1.0 - s.z[i])).collect();
            let da_c: Vec<f32> = (0..d).map(|i| dc[i] * (1.0 - s.c[i] * s.c[i])).collect();
            let rh: Vec<f32> = (0..d).map(|i| s.r[i] * s.hp[i]).collect();
            outer_acc(&mut g[self.l.w[2]..self.l.w[2] + d * e], e, &da_c, x);
            outer_acc(&mut g[self.l.u[2]..self.l.u[2] + d * d], d, &da_c, &rh);
            for i in 0..d {
                g[self.l.b[2] + i] += da_c[i];
            }
            transpose_acc(&params[self.l.w[2]..self.l.w[2] + d * e], e, &da_c, &mut dx);
            let mut drh = vec![0.0f32; d];
            transpose_acc(&params[self.l.u[2]..self.l.u[2] + d * d], d, &da_c, &mut drh);
            let da_r: Vec<f32> = (0..d).map(|i| drh[i] * s.hp[i] * s.r[i] * (1.0 - s.r[i])).collect();
            for i in 0..d {
                dhp[i] += drh[i] * s.r[i];
            }
            let da_z: Vec<f32> = (0..d).map(|i| dz[i] * s.z[i] * (1.0 - s.z[i])).collect();
            for (k, da) in [(1usize, &da_r), (0usize, &da_z)] {
                outer_acc(&mut g[self.l.w[k]..self.l.w[k] + d * e], e, da, x);
                outer_acc(&mut g[self.l.u[k]..self.l.u[k] + d * d], d, da, &s.hp);
                for i in 0..d {
                    g[self.l.b[k] + i] += da[i];
                }
                transpose_acc(&params[self.l.w[k]..self.l.w[k] + d * e], e, da, &mut dx);
                transpose_acc(&params[self.l.u[k]..self.l.u[k] + d * d], d, da, &mut dhp);
            }
            for i in 0..e {
                g[self.l.embed + s.prev as usize * e + i] += dx[i];
            }
            dh = dhp;
        }
        (loss / bytes.len() as f64, g, steps)
    }

    /// Score the chunk with the current weights (prequential), then take one
    /// Adam step on it. Returns the chunk's bits.
    pub fn train_chunk(&mut self, bytes: &[u8]) -> f64 {
        let (loss, mut g, steps) = self.loss_and_grad(&self.p, &self.state, &self.fast, &self.tail, bytes);
        let bits = loss * bytes.len() as f64 / std::f64::consts::LN_2;
        self.bits += bits;
        let offset = self.seen;
        self.seen += bytes.len();
        self.tail.extend_from_slice(bytes);
        let keep = self.cfg.circuit.map_or(0, |c| c.window + c.lags + 2).max(self.cfg.gdm.map_or(0, |g| g.lags + 2)).max(8);
        if self.tail.len() > keep {
            self.tail.drain(..self.tail.len() - keep);
        }
        if !self.learning {
            // Scored only: carry the state, change nothing.
            if let Some(gd) = steps.last().and_then(|st| st.gd.as_ref()) {
            self.fast = gd.s_new.clone();
        }
            self.state = steps.last().expect("non-empty chunk").h.clone();
            self.prev = *bytes.last().expect("non-empty chunk");
            return bits;
        }
        if self.cfg.circuit.is_some_and(|c| c.frozen) {
            g[self.l.cu..self.l.cw].iter_mut().for_each(|x| *x = 0.0);
        }
        let norm = g.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32;
        let before = if self.log.is_some() { Some(self.p.clone()) } else { None };
        if norm > self.cfg.clip {
            let s = self.cfg.clip / norm;
            g.iter_mut().for_each(|x| *x *= s);
        }
        self.steps += 1;
        let (b1, b2) = (self.cfg.beta1, self.cfg.beta2);
        let (c1, c2) = (1.0 - b1.powi(self.steps as i32), 1.0 - b2.powi(self.steps as i32));
        let mult = self.cfg.circuit.map_or(1.0, |c| c.lr_mult);
        for (i, &gi) in g.iter().enumerate() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            let lr = if i >= self.l.cu && i < self.l.cw {
                self.cfg.lr * mult
            } else if i >= self.l.gu {
                self.cfg.lr * self.cfg.gdm.map_or(1.0, |g| g.lr_mult)
            } else {
                self.cfg.lr
            };
            self.p[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
        // Carry the state forward, detached, as computed before the update.
        if let Some(gd) = steps.last().and_then(|st| st.gd.as_ref()) {
            self.fast = gd.s_new.clone();
        }
        self.state = steps.last().expect("non-empty chunk").h.clone();
        self.prev = *bytes.last().expect("non-empty chunk");
        if let Some(before) = before {
            let update_norm = self.p.iter().zip(&before).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt() as f32;
            let hash = self.weights_sha256_if_due();
            let changed = self.changed_tensors();
            let log = self.log.as_mut().expect("log");
            let seq = log.events.len();
            log.events.push(UpdateEvent { seq, stream_offset: offset, bytes: bytes.len(), bits_before: bits, grad_norm: norm, update_norm });
            log.block_targets.update(bytes);
            if let Some(h) = hash {
                let targets = std::mem::replace(&mut log.block_targets, zerolm_tensor::sha256::Sha256::new()).finish_hex();
                log.checkpoints.push(CheckpointRecord { input_stream: log.stream.clone(), signal: SIGNAL, rule: RULE, changed_tensors: changed.clone(), first_seq: log.block_start, last_seq: seq, targets_sha256: targets, checkpoint_sha256: h });
                log.block_start = seq + 1;
            }
        }
        bits
    }

    /// Log every update; close a checkpoint block (with the weights' hash) every `every` updates.
    pub fn enable_log(&mut self, input_stream: &str, every: usize) {
        self.log = Some(EventLog { stream: input_stream.into(), every: every.max(1), events: Vec::new(), checkpoints: Vec::new(), block_start: 0, block_targets: zerolm_tensor::sha256::Sha256::new() });
    }

    /// Close the open block, so the last updates are covered too.
    pub fn close_log(&mut self) {
        let h = self.weights_sha256();
        let changed = self.changed_tensors();
        if let Some(log) = self.log.as_mut() {
            if log.block_start < log.events.len() {
                let targets = std::mem::replace(&mut log.block_targets, zerolm_tensor::sha256::Sha256::new()).finish_hex();
                log.checkpoints.push(CheckpointRecord { input_stream: log.stream.clone(), signal: SIGNAL, rule: RULE, changed_tensors: changed.clone(), first_seq: log.block_start, last_seq: log.events.len() - 1, targets_sha256: targets, checkpoint_sha256: h });
                log.block_start = log.events.len();
            }
        }
    }

    /// Names of the tensors an update may change (the memory tensors only when the memory is on).
    pub fn changed_tensors(&self) -> Vec<&'static str> {
        let mut v = CHANGED.to_vec();
        if self.cfg.gdm.is_some_and(|g| !g.off) {
            v.extend(["gdm.u", "gdm.v", "gdm.d", "gdm.a0", "gdm.b0", "gdm.gamma0", "gdm.wa", "gdm.wb"]);
        }
        v
    }

    pub fn events(&self) -> &[UpdateEvent] {
        self.log.as_ref().map(|l| &l.events[..]).unwrap_or(&[])
    }

    pub fn checkpoints(&self) -> &[CheckpointRecord] {
        self.log.as_ref().map(|l| &l.checkpoints[..]).unwrap_or(&[])
    }

    /// With `false`, chunks are scored and the state advances, but no weight changes.
    pub fn set_learning(&mut self, on: bool) {
        self.learning = on;
    }

    pub fn weights_sha256(&self) -> String {
        let mut h = zerolm_tensor::sha256::Sha256::new();
        h.update(b"zerolm-gradient-core-v1\0");
        h.update_f32(&self.p);
        h.finish_hex()
    }

    fn weights_sha256_if_due(&self) -> Option<String> {
        let log = self.log.as_ref()?;
        if (log.events.len() + 1) % log.every == 0 { Some(self.weights_sha256()) } else { None }
    }

    /// One chronological pass over `bytes`; returns the prequential BPB of these bytes.
    pub fn observe_all(&mut self, bytes: &[u8]) -> f64 {
        let mut bits = 0.0;
        for chunk in bytes.chunks(self.cfg.window) {
            bits += self.train_chunk(chunk);
        }
        bits / bytes.len().max(1) as f64
    }

    /// Total prequential bits so far.
    pub fn prequential_bits(&self) -> f64 {
        self.bits
    }

    pub fn prequential_bpb(&self) -> f64 {
        self.bits / self.seen.max(1) as f64
    }

    /// Per-position bits of `bytes` from a fresh state; changes no weight.
    pub fn score(&self, bytes: &[u8]) -> Vec<f32> {
        self.score_detail(bytes).into_iter().map(|d| d[0]).collect()
    }

    /// Per position `[bits, confidence (max probability), 1.0 if the argmax is the byte else 0.0]`.
    pub fn score_detail(&self, bytes: &[u8]) -> Vec<[f32; 3]> {
        let mut h = vec![0.0f32; self.l.d];
        let mut fs = vec![0.0f32; self.fast.len()];
        let mut buf = Vec::with_capacity(bytes.len() + 1);
        buf.push(START_BYTE);
        buf.extend_from_slice(bytes);
        let mut out = Vec::with_capacity(bytes.len());
        for (i, &y) in bytes.iter().enumerate() {
            let s = self.forward_step(&self.p, &h, &fs, &buf[..i + 1]);
            let (am, conf) = s.p.iter().enumerate().fold((0usize, 0.0f32), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
            out.push([-(s.p[y as usize].max(1e-30)).log2(), conf, (am == y as usize) as u8 as f32]);
            h = s.h;
            if let Some(gd) = s.gd {
                fs = gd.s_new;
            }
        }
        out
    }

    /// Per position, call `f(i, probabilities)` with the full next-byte distribution from a fresh
    /// state; changes no weight. Used by the hybrid arms, which mix this with other predictors.
    pub fn score_probs(&self, bytes: &[u8], mut f: impl FnMut(usize, &[f32])) {
        let mut h = vec![0.0f32; self.l.d];
        let mut fs = vec![0.0f32; self.fast.len()];
        let mut buf = Vec::with_capacity(bytes.len() + 1);
        buf.push(START_BYTE);
        buf.extend_from_slice(bytes);
        for i in 0..bytes.len() {
            let s = self.forward_step(&self.p, &h, &fs, &buf[..i + 1]);
            f(i, &s.p);
            h = s.h;
            if let Some(gd) = s.gd {
                fs = gd.s_new;
            }
        }
    }

    /// Held-out BPB from a fresh state; changes no weight.
    pub fn evaluate(&self, bytes: &[u8]) -> f64 {
        let mut h = vec![0.0f32; self.l.d];
        let mut fs = vec![0.0f32; self.fast.len()];
        let mut buf = Vec::with_capacity(bytes.len() + 1);
        buf.push(START_BYTE);
        buf.extend_from_slice(bytes);
        let mut bits = 0.0f64;
        for (i, &y) in bytes.iter().enumerate() {
            let s = self.forward_step(&self.p, &h, &fs, &buf[..i + 1]);
            bits -= (s.p[y as usize].max(1e-30) as f64).log2();
            h = s.h;
            if let Some(gd) = s.gd {
                fs = gd.s_new;
            }
        }
        bits / bytes.len().max(1) as f64
    }

    /// For tests: gradient of the mean loss on `bytes` from a fresh state.
    pub fn gradient_for_test(&self, bytes: &[u8]) -> (f64, Vec<f32>) {
        let h0 = vec![0.0f32; self.l.d];
        let (l, g, _) = self.loss_and_grad(&self.p, &h0, &vec![0.0; self.fast.len()], &[START_BYTE], bytes);
        (l, g)
    }

    pub fn params_mut_for_test(&mut self) -> &mut [f32] {
        &mut self.p
    }

    pub fn loss_for_test(&self, bytes: &[u8]) -> f64 {
        self.loss_and_grad(&self.p, &vec![0.0; self.l.d], &vec![0.0; self.fast.len()], &[START_BYTE], bytes).0
    }
}
