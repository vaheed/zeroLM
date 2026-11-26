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

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub window: usize,
    pub lr: f32,
    pub clip: f32,
    pub beta1: f32,
    pub beta2: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config { window: 32, lr: 3e-3, clip: 1.0, beta1: 0.9, beta2: 0.999 }
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
    total: usize,
}

impl Layout {
    fn new(e: usize, d: usize) -> Self {
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
        Layout { e, d, embed, w, u, b, ro, rb, total: at }
    }
}

pub struct Bptt {
    cfg: Config,
    l: Layout,
    p: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    steps: u64,
    state: Vec<f32>,
    prev: u8,
    /// Total bits and bytes of the prequential pass so far.
    bits: f64,
    seen: usize,
}

/// Cached forward values of one step.
struct Step {
    prev: u8,
    hp: Vec<f32>,
    z: Vec<f32>,
    r: Vec<f32>,
    c: Vec<f32>,
    h: Vec<f32>,
    p: Vec<f32>,
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
        let l = Layout::new(EMBED_DIM, d);
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
        Bptt { cfg, m: vec![0.0; l.total], v: vec![0.0; l.total], steps: 0, state: vec![0.0; d], prev: START_BYTE, bits: 0.0, seen: 0, l, p }
    }

    /// The embedding and GRU tensors as a plastic core (same flat layout).
    pub fn core(&self) -> zerolm_learning::plastic::Core {
        zerolm_learning::plastic::Core { e: self.l.e, d: self.l.d, p: self.p[..self.l.ro].to_vec() }
    }

    /// Readout weights and bias (zero-initialized, trained here).
    pub fn readout(&self) -> (&[f32], &[f32]) {
        (&self.p[self.l.ro..self.l.rb], &self.p[self.l.rb..self.l.total])
    }

    pub fn param_count(&self) -> usize {
        self.l.total
    }

    fn forward_step(&self, params: &[f32], h: &[f32], prev: u8) -> Step {
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
        zerolm_tensor::softmax_inplace(&mut logits);
        Step { prev, hp: h.to_vec(), z, r, c, h: hn, p: logits }
    }

    /// Mean cross entropy (nats) of `bytes` from state `h0`, and the gradient
    /// of that mean with respect to every parameter.
    fn loss_and_grad(&self, params: &[f32], h0: &[f32], prev0: u8, bytes: &[u8]) -> (f64, Vec<f32>, Vec<Step>) {
        let (e, d) = (self.l.e, self.l.d);
        let f = d + e;
        let mut steps: Vec<Step> = Vec::with_capacity(bytes.len());
        let mut h = h0.to_vec();
        let mut prev = prev0;
        let mut loss = 0.0f64;
        for &y in bytes {
            let s = self.forward_step(params, &h, prev);
            loss -= (s.p[y as usize].max(1e-30) as f64).ln();
            h = s.h.clone();
            prev = y;
            steps.push(s);
        }
        let n = bytes.len() as f32;
        let mut g = vec![0.0f32; self.l.total];
        let mut dh = vec![0.0f32; d];
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
            let mut dx: Vec<f32> = dphi[d..].to_vec();
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
        let (loss, mut g, steps) = self.loss_and_grad(&self.p, &self.state, self.prev, bytes);
        let bits = loss * bytes.len() as f64 / std::f64::consts::LN_2;
        self.bits += bits;
        self.seen += bytes.len();
        let norm = g.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32;
        if norm > self.cfg.clip {
            let s = self.cfg.clip / norm;
            g.iter_mut().for_each(|x| *x *= s);
        }
        self.steps += 1;
        let (b1, b2) = (self.cfg.beta1, self.cfg.beta2);
        let (c1, c2) = (1.0 - b1.powi(self.steps as i32), 1.0 - b2.powi(self.steps as i32));
        for (i, &gi) in g.iter().enumerate() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            self.p[i] -= self.cfg.lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
        // Carry the state forward, detached, as computed before the update.
        self.state = steps.last().expect("non-empty chunk").h.clone();
        self.prev = *bytes.last().expect("non-empty chunk");
        bits
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
        let mut prev = START_BYTE;
        let mut out = Vec::with_capacity(bytes.len());
        for &y in bytes {
            let s = self.forward_step(&self.p, &h, prev);
            let (am, conf) = s.p.iter().enumerate().fold((0usize, 0.0f32), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
            out.push([-(s.p[y as usize].max(1e-30)).log2(), conf, (am == y as usize) as u8 as f32]);
            h = s.h;
            prev = y;
        }
        out
    }

    /// Held-out BPB from a fresh state; changes no weight.
    pub fn evaluate(&self, bytes: &[u8]) -> f64 {
        let mut h = vec![0.0f32; self.l.d];
        let mut prev = START_BYTE;
        let mut bits = 0.0f64;
        for &y in bytes {
            let s = self.forward_step(&self.p, &h, prev);
            bits -= (s.p[y as usize].max(1e-30) as f64).log2();
            h = s.h;
            prev = y;
        }
        bits / bytes.len().max(1) as f64
    }

    /// For tests: gradient of the mean loss on `bytes` from a fresh state.
    pub fn gradient_for_test(&self, bytes: &[u8]) -> (f64, Vec<f32>) {
        let h0 = vec![0.0f32; self.l.d];
        let (l, g, _) = self.loss_and_grad(&self.p, &h0, START_BYTE, bytes);
        (l, g)
    }

    pub fn params_mut_for_test(&mut self) -> &mut [f32] {
        &mut self.p
    }

    pub fn loss_for_test(&self, bytes: &[u8]) -> f64 {
        self.loss_and_grad(&self.p, &vec![0.0; self.l.d], START_BYTE, bytes).0
    }
}
