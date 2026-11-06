//! Bake-off metrics. No learning happens here: every candidate is evaluated
//! at its seeded initialization, so the metrics measure *architectural
//! capacity* (memory horizon, stability, cost), not language ability.
//! Held-out generalization and chronological acquisition: see [`online`] (Phase 4).

pub mod isa_ref;
pub mod memory;
pub mod reasoning;
pub mod online;

use serde::{Deserialize, Serialize};
use std::time::Instant;
use zerolm_engine::Model;
use zerolm_nn::Kind;
use zerolm_tensor::{all_finite, max_abs, norm, Rng};

pub const EMBED_DIM: usize = 16;
pub const LAGS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
pub const SENSITIVITY_THRESHOLD: f32 = 1e-3;

#[derive(Debug, Clone)]
pub struct Config {
    pub budget: usize,
    pub seeds: Vec<u64>,
    pub stream_len: usize,
    pub latency_steps: usize,
}

impl Config {
    pub fn full() -> Self {
        Config { budget: 20_000, seeds: vec![1, 2, 3], stream_len: 200_000, latency_steps: 20_000 }
    }
    pub fn quick() -> Self {
        Config { budget: 4_000, seeds: vec![1, 2, 3], stream_len: 5_000, latency_steps: 500 }
    }
}

fn random_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut r = Rng::new(seed ^ 0xA5A5);
    (0..n).map(|_| r.below(256) as u8).collect()
}

fn hidden_sensitivity(a: &[f32], b: &[f32]) -> f32 {
    let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    norm(&diff) / (norm(a) + norm(b) + 1e-12)
}

/// Sensitivity of the hidden state at each lag to the very first byte.
pub fn sensitivity_by_lag(m: &Model, seed: u64) -> Vec<f32> {
    let max_lag = *LAGS.last().unwrap();
    let tail = random_bytes(seed, max_lag);
    let d = m.cell.hidden_dim();
    let (mut a, mut b) = (m.init_state(), m.init_state());
    m.step(&mut a, 0x41, None).unwrap();
    m.step(&mut b, 0x41 ^ 0x55, None).unwrap();
    let mut out = Vec::new();
    for (i, &byte) in tail.iter().enumerate() {
        m.step(&mut a, byte, None).unwrap();
        m.step(&mut b, byte, None).unwrap();
        if LAGS.contains(&(i + 1)) {
            out.push(hidden_sensitivity(&a[..d], &b[..d]));
        }
    }
    out
}

/// Largest lag before sensitivity first falls below the threshold (0 if lag 1 fails).
pub fn horizon(sens: &[f32]) -> usize {
    let mut h = 0;
    for (s, &lag) in sens.iter().zip(LAGS.iter()) {
        if *s >= SENSITIVITY_THRESHOLD {
            h = lag;
        } else {
            break;
        }
    }
    h
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stability {
    pub stream_len: usize,
    pub finite: bool,
    pub max_abs_state: f32,
    pub state_len_start: usize,
    pub state_len_end: usize,
    pub bits_per_byte_untrained: f64,
}

pub fn stability(m: &Model, seed: u64, n: usize) -> Stability {
    let bytes = random_bytes(seed.wrapping_add(7), n);
    let mut st = m.init_state();
    let start = st.len();
    let mut peak = 0.0f32;
    let mut finite = true;
    for chunk in bytes.chunks(1000) {
        finite &= m.feed(&mut st, chunk).is_ok();
        peak = peak.max(max_abs(&st));
    }
    finite &= all_finite(&st);
    let bpb = m.bits_per_byte(&bytes[..bytes.len().min(20_000)]).unwrap_or(f64::NAN);
    Stability { stream_len: n, finite, max_abs_state: peak, state_len_start: start, state_len_end: st.len(), bits_per_byte_untrained: bpb }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    pub p50_us: f64,
    pub p95_us: f64,
    pub tokens_per_sec: f64,
    pub peak_rss_kb: Option<u64>,
}

pub fn peak_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse().ok()
}

pub fn cpu_profile(m: &Model, seed: u64, steps: usize) -> CpuProfile {
    let bytes = random_bytes(seed.wrapping_add(13), steps);
    let mut st = m.init_state();
    let mut lat = Vec::with_capacity(steps);
    let mut logits = vec![0.0; 256];
    let total = Instant::now();
    for &b in &bytes {
        let t = Instant::now();
        m.step(&mut st, b, None).unwrap();
        m.logits(&st, &mut logits);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let secs = total.elapsed().as_secs_f64();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| lat[((lat.len() as f64 - 1.0) * q) as usize];
    CpuProfile { p50_us: pick(0.5), p95_us: pick(0.95), tokens_per_sec: steps as f64 / secs, peak_rss_kb: peak_rss_kb() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ablation {
    pub horizon_intact: usize,
    pub horizon_severed: usize,
    pub shuffle_state_divergence: f32,
}

pub fn ablation(kind: Kind, seed: u64, budget: usize) -> Ablation {
    let intact = Model::new(kind, seed, EMBED_DIM, budget);
    let mut severed = Model::new(kind, seed, EMBED_DIM, budget);
    severed.cell.sever_recurrence();
    let mut shuffled = Model::new(kind, seed, EMBED_DIM, budget);
    shuffled.cell.shuffle_recurrence(seed ^ 0xFEED);
    let bytes = random_bytes(seed, 200);
    let (mut a, mut b) = (intact.init_state(), shuffled.init_state());
    intact.feed(&mut a, &bytes).unwrap();
    shuffled.feed(&mut b, &bytes).unwrap();
    let d = intact.cell.hidden_dim();
    Ablation {
        horizon_intact: horizon(&sensitivity_by_lag(&intact, seed)),
        horizon_severed: horizon(&sensitivity_by_lag(&severed, seed)),
        shuffle_state_divergence: hidden_sensitivity(&a[..d], &b[..d]),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateResult {
    pub kind: String,
    pub eligible: bool,
    pub hidden_dim: usize,
    pub cell_params: usize,
    pub total_params: usize,
    pub checkpoint_bytes: usize,
    pub session_state_bytes: usize,
    pub trace_points: usize,
    pub horizons: Vec<usize>,
    pub median_horizon: usize,
    pub stability: Vec<Stability>,
    pub ablations: Vec<Ablation>,
    pub cpu: Vec<CpuProfile>,
    pub stable: bool,
    pub score: f64,
}

fn median(mut v: Vec<usize>) -> usize {
    v.sort();
    v[v.len() / 2]
}

pub fn evaluate(kind: Kind, cfg: &Config) -> CandidateResult {
    let mut horizons = Vec::new();
    let (mut stab, mut abl, mut cpu) = (Vec::new(), Vec::new(), Vec::new());
    let mut first: Option<Model> = None;
    for &seed in &cfg.seeds {
        let m = Model::new(kind, seed, EMBED_DIM, cfg.budget);
        horizons.push(horizon(&sensitivity_by_lag(&m, seed)));
        stab.push(stability(&m, seed, cfg.stream_len));
        abl.push(ablation(kind, seed, cfg.budget));
        cpu.push(cpu_profile(&m, seed, cfg.latency_steps));
        first.get_or_insert(m);
    }
    let m = first.expect("at least one seed");
    let stable = stab.iter().all(|s| s.finite && s.state_len_start == s.state_len_end && s.max_abs_state <= 16.0);
    CandidateResult {
        kind: kind.name().into(),
        eligible: kind.eligible(),
        hidden_dim: m.cell.hidden_dim(),
        cell_params: m.cell.param_count(),
        total_params: m.param_count(),
        checkpoint_bytes: m.checkpoint_bytes(),
        session_state_bytes: m.cell.state_len() * 4,
        trace_points: m.cell.trace_points(),
        median_horizon: median(horizons.clone()),
        horizons,
        stability: stab,
        ablations: abl,
        cpu,
        stable,
        score: 0.0,
    }
}

/// Deterministic selection score (latency and RSS are reported and gated
/// separately because wall-clock numbers are noisy):
///   0.60 * retention horizon (log scale, normalized to the longest lag)
/// + 0.25 * compactness (1 - session_state_bytes / max over candidates)
/// + 0.15 * traceability (trace points / max over candidates)
/// Only stable, eligible candidates can be selected.
pub fn score_all(results: &mut [CandidateResult]) {
    let max_state = results.iter().map(|r| r.session_state_bytes).max().unwrap_or(1) as f64;
    let max_trace = results.iter().map(|r| r.trace_points).max().unwrap_or(1) as f64;
    let top = (*LAGS.last().unwrap() as f64 + 1.0).log2();
    for r in results.iter_mut() {
        let ret = (r.median_horizon as f64 + 1.0).log2() / top;
        r.score = 0.60 * ret + 0.25 * (1.0 - r.session_state_bytes as f64 / max_state) + 0.15 * (r.trace_points as f64 / max_trace);
    }
}

pub fn select(results: &[CandidateResult]) -> Option<&CandidateResult> {
    results
        .iter()
        .filter(|r| r.eligible && r.stable)
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap().then_with(|| b.kind.cmp(&a.kind)))
}

/// Hard budgets from the CPU profile; failing candidates cannot be selected.
pub const P95_LATENCY_BUDGET_US: f64 = 2_000.0;

pub fn within_cpu_budget(r: &CandidateResult) -> bool {
    r.cpu.iter().all(|c| c.p95_us <= P95_LATENCY_BUDGET_US)
}
