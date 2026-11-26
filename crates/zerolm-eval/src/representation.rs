//! Gate A harness (roadmap Phase 9): can local learning build useful
//! representations from one chronological pass over real text?
//!
//! Arms (same seeded initialization, same exposure bytes, one pass):
//! A0 frozen, A1 random features (readout only), A2 e-prop on embedding,
//! GRU and readout, A3 online backpropagation CONTROL, and A2-perturbed
//! (initial core plus a random-direction change of A2's per-tensor norm).
//!
//! The decisive measurement is the representation probe: freeze an arm's
//! embedding and core, discard its readout, train a fresh readout with the
//! online-learning milestone's frozen delta rule over the probe split `P`,
//! then score the hidden split `H`. `Delta R_x = BPB_probe(A1) - BPB_probe(x)`.

use crate::bptt::{Bptt, Config as BpttConfig};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;
use zerolm_data::{contamination, corpus, manifest::Manifest};
use zerolm_learning::online_predictor::Constants;
use zerolm_learning::plastic::{Core, EpropConfig, Plasticity, PlasticModel};
use zerolm_tensor::fnv1a;

pub const SPEC: &str = "model/specs/representation.json";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpropChoice {
    pub eta: f32,
    pub max_core_update: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    /// Minimum Delta R of the backpropagation control (headroom).
    pub h: f64,
    /// Minimum Delta R of A2.
    pub r: f64,
    pub rho_min: f64,
    pub kappa_min: f64,
    /// Largest allowed late-window prequential BPB degradation.
    pub s: f64,
    /// Largest allowed fraction of probe or hidden 13-word windows found in exposure.
    pub max_shared_window_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sizes {
    pub e1_bytes: usize,
    pub e2_bytes: usize,
    pub probe_bytes: usize,
    pub hidden_bytes: usize,
    pub core_budget: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub version: u32,
    pub frozen: bool,
    pub frozen_on: Option<String>,
    pub rationale: Option<String>,
    pub status: Option<String>,
    pub feedback_seed: u64,
    pub pilot_seeds: Vec<u64>,
    pub main_seeds: Vec<u64>,
    pub smoke: Sizes,
    pub pilot: Sizes,
    pub main: Sizes,
    /// Core budgets the pilot may compare (cell parameters).
    pub core_budgets: Vec<usize>,
    /// Pre-registered e-prop configurations (the pilot may try exactly these).
    pub eprop_pilot_configs: Vec<EpropChoice>,
    pub bptt_pilot_lrs: Vec<f32>,
    pub eprop: Option<EpropChoice>,
    pub bptt_lr: Option<f32>,
    pub novel_k: usize,
    pub thresholds: Option<Thresholds>,
    /// True once the largest affordable core and stream have been tried (turns a V2 failure into fail_pivot).
    pub largest_affordable: bool,
    pub milestones: Vec<usize>,
}

impl Spec {
    pub fn load(root: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(root.join(SPEC)).map_err(|e| format!("{SPEC}: {e}"))?;
        let s: Spec = serde_json::from_str(&text).map_err(|e| format!("{SPEC}: {e}"))?;
        s.validate()?;
        Ok(s)
    }

    pub fn sha256(root: &Path) -> String {
        corpus::sha256_hex(&std::fs::read(root.join(SPEC)).unwrap_or_default())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.main_seeds.len() < 3 {
            return Err("at least 3 main seeds".into());
        }
        if self.main_seeds.iter().any(|s| self.pilot_seeds.contains(s)) {
            return Err("pilot and main seeds must be disjoint".into());
        }
        if self.eprop_pilot_configs.is_empty() || self.bptt_pilot_lrs.is_empty() || self.core_budgets.is_empty() {
            return Err("pilot configuration lists must be non-empty".into());
        }
        if self.frozen && (self.eprop.is_none() || self.bptt_lr.is_none() || self.thresholds.is_none() || self.frozen_on.is_none() || self.rationale.is_none()) {
            return Err("a frozen spec needs the chosen rule, learning rate, thresholds, date, and rationale".into());
        }
        if !self.frozen && (self.eprop.is_some() || self.thresholds.is_some()) {
            return Err("an unfrozen spec must not carry chosen constants or thresholds".into());
        }
        if self.milestones.is_empty() || self.novel_k == 0 {
            return Err("milestones and novel_k are required".into());
        }
        Ok(())
    }
}

/// Data roles for one run.
pub struct Data {
    pub e1: Vec<u8>,
    pub e2: Vec<u8>,
    pub probe: Vec<u8>,
    pub probe_e1: Vec<u8>,
    pub hidden: Vec<u8>,
    pub hidden_e1: Vec<u8>,
    pub manifest_sha256: String,
    pub contamination: Value,
    pub contamination_passed: bool,
}

fn cap(v: Vec<u8>, n: usize) -> Vec<u8> {
    let mut v = v;
    v.truncate(n);
    v
}

/// Build the roles from the manifest. The first source in `exposure_order` is E1, the second E2.
/// `pilot` swaps the hidden role for the development role (pilots never read hidden data).
pub fn load_data(root: &Path, manifest_text: &str, sz: &Sizes, pilot: bool, max_shared: f64) -> Result<Data, String> {
    let m = Manifest::parse(manifest_text)?;
    let problems = corpus::verify(&m, root);
    if !problems.is_empty() {
        return Err(problems.join("; "));
    }
    if m.exposure_order.len() < 2 {
        return Err("the retention test needs two exposure sources".into());
    }
    let (s1, s2) = (&m.exposure_order[0], &m.exposure_order[1]);
    let stream = |src: &str, role| corpus::source_role_stream(&m, root, src, role);
    let held = if pilot { corpus::Role::Dev } else { corpus::Role::Hidden };
    let (e1, e2) = (cap(stream(s1, corpus::Role::Exposure)?, sz.e1_bytes), cap(stream(s2, corpus::Role::Exposure)?, sz.e2_bytes));
    // `probe_bytes` and `hidden_bytes` cap each source's role (the first documents of the role).
    let (probe_e1, probe_e2) = (cap(stream(s1, corpus::Role::Probe)?, sz.probe_bytes), cap(stream(s2, corpus::Role::Probe)?, sz.probe_bytes));
    let (hidden_e1, hidden_e2) = (cap(stream(s1, held)?, sz.hidden_bytes), cap(stream(s2, held)?, sz.hidden_bytes));
    let cat = |a: &[u8], b: &[u8]| [a, b].concat();
    let (probe, hidden) = (cat(&probe_e1, &probe_e2), cat(&hidden_e1, &hidden_e2));
    // Contamination: 13-word windows of probe and held-out text found in exposure.
    let exposure = cat(&e1, &e2);
    let index = contamination::ngram_hashes(&exposure, 13);
    let mut rows = Vec::new();
    let mut passed = true;
    for (name, text) in [("probe", &probe), ("held_out", &hidden)] {
        let o = contamination::overlap(&index, text, 13);
        let frac = o.shared as f64 / o.windows.max(1) as f64;
        passed &= frac <= max_shared;
        rows.push(json!({"role": name, "windows": o.windows, "shared": o.shared, "fraction": frac}));
    }
    Ok(Data { e1, e2, probe, probe_e1, hidden, hidden_e1, manifest_sha256: corpus::sha256_hex(manifest_text.as_bytes()), contamination: json!(rows), contamination_passed: passed })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Frozen,
    RandomFeatures,
    Local,
    Bptt,
}

impl Arm {
    pub fn id(self) -> &'static str {
        match self {
            Arm::Frozen => "frozen",
            Arm::RandomFeatures => "random_features",
            Arm::Local => "local",
            Arm::Bptt => "bptt_control",
        }
    }
}

/// Probe of one core on (`p`, `h`): fresh readout, delta rule, one pass over `p`, BPB on `h`.
pub fn probe(core: &Core, p: &[u8], h: &[u8], delta: &Constants) -> f64 {
    let mut m = PlasticModel::new(core.clone(), Plasticity::ReadoutOnly, *delta, EpropConfig::default());
    m.observe_all(p).expect("finite probe");
    m.evaluate(h)
}

/// Sorted hashes of every `k`-byte window of `exposure`.
pub fn kgram_index(exposure: &[u8], k: usize) -> Vec<u64> {
    let mut v: Vec<u64> = exposure.windows(k).map(fnv1a).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Positions of `h` whose preceding `k` bytes never occur in the exposure index.
pub fn novel_positions(h: &[u8], index: &[u64], k: usize) -> Vec<bool> {
    (0..h.len()).map(|i| i >= k && index.binary_search(&fnv1a(&h[i - k..i])).is_err()).collect()
}

pub fn masked_mean(bits: &[f32], mask: &[bool]) -> f64 {
    let (s, n) = bits.iter().zip(mask).filter(|(_, m)| **m).fold((0.0f64, 0usize), |a, (b, _)| (a.0 + *b as f64, a.1 + 1));
    if n == 0 { f64::NAN } else { s / n as f64 }
}

/// Expected calibration error over 15 equal-width confidence bins.
pub fn ece(detail: &[[f32; 3]]) -> f64 {
    let mut bins = [(0.0f64, 0.0f64, 0usize); 15];
    for d in detail {
        let b = ((d[1] * 15.0) as usize).min(14);
        bins[b].0 += d[1] as f64;
        bins[b].1 += d[2] as f64;
        bins[b].2 += 1;
    }
    let n = detail.len().max(1) as f64;
    bins.iter().filter(|b| b.2 > 0).map(|b| (b.0 - b.1).abs() / n).sum()
}

#[derive(Debug, Clone)]
pub struct Eval {
    pub bytes: usize,
    pub prequential_bpb: f64,
    pub heldout_bpb: f64,
    pub novel_bpb: f64,
    pub ece: f64,
    /// Probe BPB on all-source held-out text, and on E1-source text only.
    pub probe_bpb: f64,
    pub probe_e1_bpb: f64,
}

#[derive(Debug, Clone)]
pub struct ArmRun {
    pub arm: Arm,
    pub seed: u64,
    pub evals: Vec<Eval>,
    pub windows: Vec<f64>,
    pub finite: bool,
    pub max_abs_weight: f32,
    pub final_core: Core,
    pub change_norms: Vec<f64>,
    pub wall_s: f64,
    pub bytes: usize,
    pub bytes_per_s: f64,
    pub peak_rss_kb: Option<u64>,
}

pub struct RunParams {
    pub delta: Constants,
    pub eprop: EpropConfig,
    pub bptt_lr: f32,
    pub budget: usize,
    pub novel_k: usize,
    pub milestones: Vec<usize>,
    pub window: usize,
}

/// Milestones to evaluate for a two-segment stream: the configured ones that fit, the end of E1, and the end.
pub fn milestone_list(configured: &[usize], e1: usize, total: usize) -> Vec<usize> {
    let mut v: Vec<usize> = configured.iter().copied().filter(|m| *m <= total).chain([e1, total]).filter(|m| *m > 0).collect();
    v.sort_unstable();
    v.dedup();
    v
}

enum Learner {
    Plastic(Box<PlasticModel>),
    Bptt(Box<Bptt>),
}

impl Learner {
    fn observe(&mut self, bytes: &[u8]) -> Result<f64, &'static str> {
        match self {
            Learner::Plastic(m) => {
                let mut bits = 0.0;
                for &b in bytes {
                    bits += m.observe(b)?;
                }
                Ok(bits)
            }
            Learner::Bptt(b) => {
                let before = b.prequential_bits();
                b.observe_all(bytes);
                Ok(b.prequential_bits() - before)
            }
        }
    }
    fn core(&self) -> Core {
        match self {
            Learner::Plastic(m) => m.core.clone(),
            Learner::Bptt(b) => b.core(),
        }
    }
    fn detail(&self, h: &[u8]) -> Vec<[f32; 3]> {
        match self {
            Learner::Plastic(m) => m.score_detail(h),
            Learner::Bptt(b) => b.score_detail(h),
        }
    }
}

/// Run one arm through the exposure stream (`e1` then `e2`), evaluating at the milestones.
pub fn run_arm(arm: Arm, seed: u64, d: &Data, rp: &RunParams, index: &[u64], init: &Core) -> ArmRun {
    let t0 = Instant::now();
    let total = d.e1.len() + d.e2.len();
    let stream: Vec<u8> = [&d.e1[..], &d.e2[..]].concat();
    let miles = milestone_list(&rp.milestones, d.e1.len(), total);
    let mut learner = match arm {
        Arm::Frozen => Learner::Plastic(Box::new(PlasticModel::new(init.clone(), Plasticity::Frozen, rp.delta, rp.eprop))),
        Arm::RandomFeatures => Learner::Plastic(Box::new(PlasticModel::new(init.clone(), Plasticity::ReadoutOnly, rp.delta, rp.eprop))),
        Arm::Local => Learner::Plastic(Box::new(PlasticModel::new(init.clone(), Plasticity::Eprop, rp.delta, rp.eprop))),
        Arm::Bptt => Learner::Bptt(Box::new(Bptt::with_budget(seed, BpttConfig { lr: rp.bptt_lr, ..Default::default() }, rp.budget))),
    };
    let (mut evals, mut windows) = (Vec::new(), Vec::new());
    let (mut at, mut bits_all, mut finite) = (0usize, 0.0f64, true);
    let mask = novel_positions(&d.hidden, index, rp.novel_k);
    for &m in &miles {
        while at < m {
            let end = (at + rp.window).min(m);
            match learner.observe(&stream[at..end]) {
                Ok(b) => {
                    bits_all += b;
                    windows.push(b / (end - at) as f64);
                }
                Err(_) => {
                    finite = false;
                    break;
                }
            }
            at = end;
        }
        if !finite {
            break;
        }
        let detail = learner.detail(&d.hidden);
        let bits: Vec<f32> = detail.iter().map(|x| x[0]).collect();
        let core = learner.core();
        let delta = rp.delta;
        let (probe_all, probe_e1) = std::thread::scope(|s| {
            let a = s.spawn(|| probe(&core, &d.probe, &d.hidden, &delta));
            let b = probe(&core, &d.probe_e1, &d.hidden_e1, &delta);
            (a.join().expect("probe thread"), b)
        });
        evals.push(Eval {
            bytes: m,
            prequential_bpb: bits_all / m.max(1) as f64,
            heldout_bpb: bits.iter().map(|b| *b as f64).sum::<f64>() / bits.len().max(1) as f64,
            novel_bpb: masked_mean(&bits, &mask),
            ece: ece(&detail),
            probe_bpb: probe_all,
            probe_e1_bpb: probe_e1,
        });
    }
    let final_core = learner.core();
    let finite = finite && final_core.p.iter().all(|w| w.is_finite());
    let wall_s = t0.elapsed().as_secs_f64();
    ArmRun {
        arm,
        seed,
        evals,
        windows,
        finite,
        max_abs_weight: final_core.p.iter().fold(0.0f32, |a, b| a.max(b.abs())),
        change_norms: final_core.change_norms(init),
        final_core,
        wall_s,
        bytes: at,
        bytes_per_s: at as f64 / wall_s.max(1e-9),
        peak_rss_kb: crate::peak_rss_kb(),
    }
}

/// Probe-only evaluation of a fixed core (A2-perturbed, and the A1/A0 reference cores) at the milestones that need it: the end of E1 and the end.
pub fn probe_core(core: &Core, d: &Data, delta: &Constants) -> (f64, f64) {
    std::thread::scope(|s| {
        let a = s.spawn(|| probe(core, &d.probe, &d.hidden, delta));
        let b = probe(core, &d.probe_e1, &d.hidden_e1, delta);
        (a.join().expect("probe thread"), b)
    })
}

pub fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let sd = if v.len() > 1 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
    (m, sd)
}

/// "Significantly above `bound`": mean minus two standard deviations exceeds it.
pub fn significant_above(v: &[f64], bound: f64) -> bool {
    let (m, sd) = mean_sd(v);
    m - 2.0 * sd > bound
}

/// Per-seed measurements the decision rule reads.
#[derive(Debug, Clone)]
pub struct SeedMeasures {
    pub delta_r_local: f64,
    pub delta_r_bptt: f64,
    pub delta_r_perturbed: f64,
    /// A1 minus A2 end-to-end BPB on novel-context positions at the end.
    pub novel_gain_local: f64,
    pub kappa_local: f64,
    pub finite: bool,
    /// Worst late-window prequential BPB minus the best window, for A2.
    pub late_degradation_local: f64,
}

#[derive(Debug, Clone)]
pub struct Check {
    pub id: &'static str,
    pub passed: bool,
    pub value: f64,
    pub threshold: f64,
    pub detail: String,
}

/// Late-window degradation: worst window in the second half minus the best window overall.
pub fn late_degradation(windows: &[f64]) -> f64 {
    if windows.len() < 4 {
        return 0.0;
    }
    let best = windows.iter().cloned().fold(f64::INFINITY, f64::min);
    let worst = windows[windows.len() / 2..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    worst - best
}

/// Evaluate V2-V3 and P1-P5 on the main seeds. V1 and V4 are checked by the caller.
pub fn decide(m: &[SeedMeasures], t: &Thresholds) -> (Vec<Check>, Vec<Check>) {
    let col = |f: fn(&SeedMeasures) -> f64| -> Vec<f64> { m.iter().map(f).collect() };
    let (dl, db, dp) = (col(|s| s.delta_r_local), col(|s| s.delta_r_bptt), col(|s| s.delta_r_perturbed));
    let (ml, mb, mp) = (mean_sd(&dl).0, mean_sd(&db).0, mean_sd(&dp).0);
    let validity = vec![
        Check { id: "V2", passed: significant_above(&db, 0.0) && mb >= t.h, value: mb, threshold: t.h, detail: "backpropagation control gain is significant and at least h".into() },
        Check { id: "V3", passed: !significant_above(&dp, 0.0), value: mp, threshold: 0.0, detail: "A2-perturbed gain is not significantly positive".into() },
    ];
    let rho = if mb.abs() > 1e-12 { ml / mb } else { f64::NAN };
    let kappas = col(|s| s.kappa_local);
    let novel = col(|s| s.novel_gain_local);
    let late = col(|s| s.late_degradation_local);
    let worst_late = late.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let criteria = vec![
        Check { id: "P1", passed: significant_above(&dl, 0.0) && ml >= t.r && ml > mp, value: ml, threshold: t.r, detail: "A2 probe gain is significant, at least r, and above the perturbed control".into() },
        Check { id: "P2", passed: rho >= t.rho_min, value: rho, threshold: t.rho_min, detail: "fraction of the backpropagation gain captured".into() },
        Check { id: "P3", passed: significant_above(&novel, 0.0), value: mean_sd(&novel).0, threshold: 0.0, detail: "A2 beats A1 end-to-end on novel-context positions".into() },
        Check { id: "P4", passed: mean_sd(&kappas).0 >= t.kappa_min, value: mean_sd(&kappas).0, threshold: t.kappa_min, detail: "retention of the E1 probe gain after E2".into() },
        Check { id: "P5", passed: m.iter().all(|s| s.finite) && worst_late <= t.s, value: worst_late, threshold: t.s, detail: "no non-finite value; late-window degradation bounded".into() },
    ];
    (validity, criteria)
}

/// Map checks to an outcome. `v1_v4` are the pipeline and baseline validity checks.
pub fn outcome(validity: &[Check], criteria: &[Check], largest_affordable: bool) -> (&'static str, Vec<&'static str>) {
    let failed_v: Vec<&'static str> = validity.iter().filter(|c| !c.passed).map(|c| c.id).collect();
    if !failed_v.is_empty() {
        let only_headroom = failed_v == ["V2"];
        return if only_headroom && largest_affordable { ("fail_pivot", failed_v) } else { ("inconclusive", failed_v) };
    }
    let failed_p: Vec<&'static str> = criteria.iter().filter(|c| !c.passed).map(|c| c.id).collect();
    if failed_p.is_empty() { ("pass", criteria.iter().map(|c| c.id).collect()) } else { ("fail_pivot", failed_p) }
}
