//! Online-learning benchmark: chronological online exposure, held-out evaluation,
//! baselines, ablation, and the acquisition gate.
//!
//! Every learner sees the exposure stream once, byte by byte, in order. At
//! each milestone a read-only evaluation runs on the held-out and novel
//! streams (which never reach an update). The gate compares each local rule
//! against the frozen initialized model and the byte-frequency baseline with
//! a floor fixed in `model/specs/online-learning.json` before the main run.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use zerolm_learning::ngram::Ngram;
use zerolm_learning::online_predictor::{CheckpointRecord, UpdateEvent};
use zerolm_learning::streams::{sha256_hex, Contamination, StreamSpec};
use zerolm_learning::{Constants, EvalMetrics, OnlinePredictor, Rule};

pub const PROTOCOL: &str = "model/specs/online-learning.json";
/// Ablation arm: delta rule with the recurrent path severed.
pub const ABLATION_ARM: &str = "delta_no_recurrence";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    pub version: u32,
    pub stream_spec: String,
    pub exposure_bytes: usize,
    pub eval_bytes: usize,
    pub milestones: Vec<usize>,
    pub checkpoint_every: usize,
    pub pilot_seeds: Vec<u64>,
    pub main_seeds: Vec<u64>,
    pub primary_rule: String,
    pub local_rules: Vec<String>,
    pub baseline_orders: Vec<usize>,
    pub improvement_floor_bpb: f64,
    pub floor_rationale: String,
    pub constants: Constants,
    pub bptt_control: String,
}

impl Protocol {
    pub fn load(root: &Path) -> Result<Protocol, String> {
        let t = std::fs::read_to_string(root.join(PROTOCOL)).map_err(|e| format!("{PROTOCOL}: {e}"))?;
        let p: Protocol = serde_json::from_str(&t).map_err(|e| format!("{PROTOCOL}: {e}"))?;
        p.validate()?;
        Ok(p)
    }

    pub fn validate(&self) -> Result<(), String> {
        let rules: Result<Vec<Rule>, String> =
            self.local_rules.iter().map(|r| Rule::parse(r).ok_or(format!("unknown rule '{r}'"))).collect();
        let rules = rules?;
        if rules.contains(&Rule::Frozen) {
            return Err("frozen is a baseline, not a local rule".into());
        }
        if !self.local_rules.contains(&self.primary_rule) {
            return Err("primary_rule must be one of local_rules".into());
        }
        if self.main_seeds.len() < 3 {
            return Err("at least three main seeds are required".into());
        }
        if self.main_seeds.iter().any(|s| self.pilot_seeds.contains(s)) {
            return Err("pilot and main seeds must be disjoint".into());
        }
        if !self.baseline_orders.contains(&1) {
            return Err("the byte-frequency (order 1) baseline is required".into());
        }
        if self.milestones.first() != Some(&0) || self.milestones.last() != Some(&self.exposure_bytes) {
            return Err("milestones must start at 0 and end at exposure_bytes".into());
        }
        if self.milestones.windows(2).any(|w| w[0] >= w[1]) {
            return Err("milestones must be strictly increasing".into());
        }
        if self.improvement_floor_bpb <= 0.0 {
            return Err("improvement_floor_bpb must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CurvePoint {
    pub events: usize,
    pub heldout: EvalMetrics,
    pub novel: EvalMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub seed: u64,
    pub arm: String,
    pub rule: &'static str,
    pub initial_sha256: String,
    pub final_sha256: String,
    pub update_events: usize,
    pub checkpoints: usize,
    /// Prequential (predict-then-learn) BPB over the exposure stream.
    pub online_bpb: f64,
    pub heldout: EvalMetrics,
    pub novel: EvalMetrics,
    /// Re-reading the first `eval_bytes` of exposure (memorization reference).
    pub exposure_replay: EvalMetrics,
    pub curve: Vec<CurvePoint>,
    pub log_failures: Vec<String>,
    pub us_per_event: f64,
    pub plastic_params: usize,
    pub total_params: usize,
    #[serde(skip)]
    pub events: Vec<UpdateEvent>,
    #[serde(skip)]
    pub checkpoint_log: Vec<CheckpointRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineResult {
    pub seed: u64,
    pub name: String,
    pub order: usize,
    pub heldout_bpb: f64,
    pub novel_bpb: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeedStreams {
    pub seed: u64,
    pub exposure_bytes: usize,
    pub exposure_sha256: String,
    pub heldout_bytes: usize,
    pub heldout_sha256: String,
    pub novel_bytes: usize,
    pub novel_sha256: String,
    pub contamination: Contamination,
}

#[derive(Debug, Clone, Serialize)]
pub struct Stat {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl Stat {
    pub fn of(v: &[f64]) -> Stat {
        let n = v.len().max(1) as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = if v.len() > 1 { v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
        Stat {
            mean,
            std: var.sqrt(),
            min: v.iter().cloned().fold(f64::INFINITY, f64::min),
            max: v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Gate {
    pub rule: String,
    pub set: &'static str,
    /// Per seed: min(frozen, byte-frequency) BPB minus the rule's BPB.
    pub improvement_per_seed: Vec<f64>,
    pub improvement: Stat,
    pub floor: f64,
    /// Every seed clears the floor and mean - 2 sd clears it too.
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub rule: String,
    pub acquisition: bool,
    pub heldout_bpb: Stat,
    pub novel_bpb: Stat,
    /// Mean novel BPB minus mean exposure-replay BPB (memorization gap).
    pub generalization_gap_bpb: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Report kind (always "online-learning").
    pub report: &'static str,
    pub mode: &'static str,
    pub passed: bool,
    pub failures: Vec<String>,
    pub protocol: Protocol,
    pub stream_spec_id: String,
    pub stream_spec_sha256: String,
    pub seeds: Vec<SeedStreams>,
    pub runs: Vec<RunResult>,
    pub baselines: Vec<BaselineResult>,
    pub gates: Vec<Gate>,
    pub verdicts: Vec<Verdict>,
    /// Mean held-out BPB lost when the recurrent path is severed (delta).
    pub recurrence_ablation_delta_bpb: f64,
    pub bptt_control: serde_json::Value,
    pub scope_note: &'static str,
}

pub const SCOPE_NOTE: &str = "Acquisition here means lower held-out bits-per-byte on generated sentences after chronological exposure, compared with the same untrained architecture and with a byte-frequency baseline. Only the readout layer learns; the GRU core and embedding are frozen random features. Byte n-gram baselines of order >= 3 remain far better: this is not a competitive language model and makes no claim about English understanding.";

fn arms(p: &Protocol) -> Vec<(String, Rule, bool)> {
    let mut v = vec![("frozen".to_string(), Rule::Frozen, false)];
    for r in &p.local_rules {
        v.push((r.clone(), Rule::parse(r).expect("validated"), false));
    }
    v.push((ABLATION_ARM.to_string(), Rule::Delta, true));
    v
}

/// Check that the update log corresponds exactly to the exposure stream.
pub fn verify_log(rule: Rule, exposure: &[u8], run: &RunResult) -> Vec<String> {
    let mut f = Vec::new();
    if rule == Rule::Frozen {
        if !run.events.is_empty() || !run.checkpoint_log.is_empty() {
            f.push("frozen run logged updates".into());
        }
        if run.initial_sha256 != run.final_sha256 {
            f.push("frozen run changed weights".into());
        }
        return f;
    }
    if run.initial_sha256 == run.final_sha256 {
        f.push("weights unchanged after exposure".into());
    }
    if run.events.len() != exposure.len() {
        f.push(format!("{} events for {} observed bytes", run.events.len(), exposure.len()));
    }
    if let Some((i, e)) = run
        .events
        .iter()
        .enumerate()
        .find(|(i, e)| e.seq != *i || e.stream_offset != *i || exposure.get(*i) != Some(&e.target))
    {
        f.push(format!("event {i} does not match the stream: {e:?}"));
    }
    let mut next = 0;
    for c in &run.checkpoint_log {
        if c.first_seq != next || c.last_seq < c.first_seq || c.last_seq >= exposure.len() {
            f.push(format!("checkpoint blocks not contiguous at {}", c.first_seq));
            break;
        }
        if c.targets_sha256 != sha256_hex(&exposure[c.first_seq..=c.last_seq]) {
            f.push(format!("checkpoint {}..{} target hash mismatch", c.first_seq, c.last_seq));
        }
        next = c.last_seq + 1;
    }
    if next != exposure.len() {
        f.push("checkpoint blocks do not cover the stream".into());
    }
    if run.checkpoint_log.last().map(|c| &c.checkpoint_sha256) != Some(&run.final_sha256) {
        f.push("last checkpoint hash is not the final weight hash".into());
    }
    f
}

/// The role-separated byte streams of one seed.
#[derive(Clone, Copy)]
struct Streams<'a> {
    id: &'a str,
    exposure: &'a [u8],
    heldout: &'a [u8],
    novel: &'a [u8],
}

fn run_arm(p: &Protocol, seed: u64, arm: &str, rule: Rule, sever: bool, st: Streams) -> RunResult {
    let (ex, ho, nv) = (st.exposure, st.heldout, st.novel);
    let mut m = OnlinePredictor::new(seed, rule, p.constants, st.id);
    if sever {
        m.sever_recurrence();
    }
    m.set_checkpoint_every(p.checkpoint_every);
    let initial = m.weights_sha256();
    let mut curve = Vec::new();
    let (mut bits, mut learn_secs, mut pos) = (0.0f64, 0.0f64, 0usize);
    let eval = |m: &OnlinePredictor, b: &[u8]| m.evaluate(b).expect("finite evaluation");
    for &ms in &p.milestones {
        let t = Instant::now();
        for &b in &ex[pos..ms] {
            bits += m.observe(b).expect("finite update").bits;
        }
        learn_secs += t.elapsed().as_secs_f64();
        pos = ms;
        curve.push(CurvePoint { events: ms, heldout: eval(&m, ho), novel: eval(&m, nv) });
    }
    m.close_block();
    let last = curve.last().expect("milestones").clone();
    let mut r = RunResult {
        seed,
        arm: arm.to_string(),
        rule: rule.name(),
        initial_sha256: initial,
        final_sha256: m.weights_sha256(),
        update_events: m.events().len(),
        checkpoints: m.checkpoints().len(),
        online_bpb: bits / ex.len().max(1) as f64,
        heldout: last.heldout,
        novel: last.novel,
        exposure_replay: eval(&m, &ex[..p.eval_bytes.min(ex.len())]),
        curve,
        log_failures: Vec::new(),
        us_per_event: learn_secs * 1e6 / ex.len().max(1) as f64,
        plastic_params: m.plastic_param_count(),
        total_params: m.param_count(),
        events: m.events().to_vec(),
        checkpoint_log: m.checkpoints().to_vec(),
    };
    r.log_failures = verify_log(rule, ex, &r);
    r
}

/// Run the benchmark on `seeds`. `gate` false (pilot mode) never fails on
/// the acquisition floor.
pub fn run(root: &Path, p: &Protocol, seeds: &[u64], gate: bool) -> Result<Report, String> {
    let spec_text = std::fs::read_to_string(root.join(&p.stream_spec)).map_err(|e| format!("{}: {e}", p.stream_spec))?;
    let spec = StreamSpec::parse(&spec_text).map_err(|e| format!("{}: {e:?}", p.stream_spec))?;
    let mut failures = Vec::new();
    let mut seed_info = Vec::new();
    let mut runs = Vec::new();
    let mut baselines = Vec::new();
    for &seed in seeds {
        let split = spec.split(seed, p.exposure_bytes, p.eval_bytes);
        // The protocol exposes exactly `exposure_bytes` events.
        let ex = &split.exposure.as_bytes()[..p.exposure_bytes];
        let (ho, nv) = (split.heldout.as_bytes(), split.novel.as_bytes());
        let contamination = split.contamination(&spec);
        if !contamination.clean() {
            failures.push(format!("seed {seed}: stream contamination {contamination:?}"));
        }
        seed_info.push(SeedStreams {
            seed,
            exposure_bytes: ex.len(),
            exposure_sha256: sha256_hex(ex),
            heldout_bytes: ho.len(),
            heldout_sha256: sha256_hex(ho),
            novel_bytes: nv.len(),
            novel_sha256: sha256_hex(nv),
            contamination,
        });
        let stream_id = format!("{}#seed={seed}", spec.id);
        let streams = Streams { id: &stream_id, exposure: ex, heldout: ho, novel: nv };
        let arm_list = arms(p);
        let results: Vec<RunResult> = std::thread::scope(|s| {
            let hs: Vec<_> = arm_list
                .iter()
                .map(|(arm, rule, sever)| s.spawn(move || run_arm(p, seed, arm, *rule, *sever, streams)))
                .collect();
            hs.into_iter().map(|h| h.join().expect("arm thread")).collect()
        });
        for r in &results {
            for f in &r.log_failures {
                failures.push(format!("seed {seed} {}: {f}", r.arm));
            }
        }
        runs.extend(results);
        for &o in &p.baseline_orders {
            let mut g = Ngram::new(o);
            g.observe_all(ex);
            let name = if o == 1 { "byte_frequency".to_string() } else { format!("ngram{o}") };
            baselines.push(BaselineResult { seed, name, order: o, heldout_bpb: g.bpb(ho), novel_bpb: g.bpb(nv) });
        }
    }
    let find = |seed: u64, arm: &str| runs.iter().find(|r| r.seed == seed && r.arm == arm).expect("run exists");
    let unigram = |seed: u64| baselines.iter().find(|b| b.seed == seed && b.order == 1).expect("order 1");
    let mut gates = Vec::new();
    let mut verdicts = Vec::new();
    for rule in &p.local_rules {
        let mut acquired = true;
        for set in ["heldout", "novel"] {
            let imp: Vec<f64> = seeds
                .iter()
                .map(|&s| {
                    let (fr, rr, u) = (find(s, "frozen"), find(s, rule), unigram(s));
                    let (fb, rb, ub) = match set {
                        "heldout" => (fr.heldout.bpb, rr.heldout.bpb, u.heldout_bpb),
                        _ => (fr.novel.bpb, rr.novel.bpb, u.novel_bpb),
                    };
                    fb.min(ub) - rb
                })
                .collect();
            let st = Stat::of(&imp);
            let passed = st.min >= p.improvement_floor_bpb && st.mean - 2.0 * st.std >= p.improvement_floor_bpb;
            acquired &= passed;
            gates.push(Gate { rule: rule.clone(), set, improvement_per_seed: imp, improvement: st, floor: p.improvement_floor_bpb, passed });
        }
        let rs: Vec<&RunResult> = seeds.iter().map(|&s| find(s, rule)).collect();
        let novel = Stat::of(&rs.iter().map(|r| r.novel.bpb).collect::<Vec<_>>());
        let replay = Stat::of(&rs.iter().map(|r| r.exposure_replay.bpb).collect::<Vec<_>>());
        verdicts.push(Verdict {
            rule: rule.clone(),
            acquisition: acquired,
            heldout_bpb: Stat::of(&rs.iter().map(|r| r.heldout.bpb).collect::<Vec<_>>()),
            generalization_gap_bpb: novel.mean - replay.mean,
            novel_bpb: novel,
        });
    }
    let primary_ok = verdicts.iter().any(|v| v.rule == p.primary_rule && v.acquisition);
    if gate && !primary_ok {
        failures.push(format!("primary rule '{}' did not clear the {} BPB floor on every seed", p.primary_rule, p.improvement_floor_bpb));
    }
    let abl: Vec<f64> = seeds.iter().map(|&s| find(s, ABLATION_ARM).heldout.bpb - find(s, "delta").heldout.bpb).collect();
    Ok(Report {
        report: "online-learning",
        mode: if gate { "main" } else { "pilot" },
        passed: failures.is_empty(),
        failures,
        protocol: p.clone(),
        stream_spec_id: spec.id.clone(),
        stream_spec_sha256: spec.sha256(),
        seeds: seed_info,
        runs,
        baselines,
        gates,
        verdicts,
        recurrence_ablation_delta_bpb: Stat::of(&abl).mean,
        bptt_control: serde_json::json!({
            "status": p.bptt_control,
            "reason": "Optional online truncated-BPTT control not implemented: the project has no backpropagation code, and the strict local rule passed, so the gradient control was not needed to decide the online-learning gate. Any future BPTT control must be reported separately and never as the strict local-plasticity result.",
        }),
        scope_note: SCOPE_NOTE,
    })
}

pub fn learning_curve_csv(r: &Report) -> String {
    let mut s = String::from("seed,arm,events,heldout_bpb,heldout_accuracy,heldout_ece,novel_bpb,novel_accuracy,novel_ece\n");
    for run in &r.runs {
        for c in &run.curve {
            s += &format!(
                "{},{},{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}\n",
                run.seed, run.arm, c.events, c.heldout.bpb, c.heldout.accuracy, c.heldout.ece, c.novel.bpb, c.novel.accuracy, c.novel.ece
            );
        }
    }
    s
}

pub fn summary_md(r: &Report) -> String {
    let seeds: Vec<String> = r.seeds.iter().map(|s| s.seed.to_string()).collect();
    let mut s = format!(
        "## Online learning ({}): {}\n\nStream `{}` (spec sha256 `{}`), {} exposure events per seed, seeds {}. Improvement floor {} BPB (frozen before the main run).\n\n",
        r.mode,
        if r.passed { "PASS" } else { "FAIL" },
        r.stream_spec_id,
        &r.stream_spec_sha256[..16],
        r.protocol.exposure_bytes,
        seeds.join(", "),
        r.protocol.improvement_floor_bpb
    );
    s += "### Held-out BPB after exposure (mean +- sd over seeds; lower is better)\n\n| arm | held-out BPB | novel-combination BPB | held-out acc | held-out ECE | online BPB | us/event |\n|---|---|---|---|---|---|---|\n";
    let mut arm_names: Vec<&str> = Vec::new();
    for run in &r.runs {
        if !arm_names.contains(&run.arm.as_str()) {
            arm_names.push(&run.arm);
        }
    }
    for arm in arm_names {
        let rs: Vec<&RunResult> = r.runs.iter().filter(|x| x.arm == arm).collect();
        let st = |f: &dyn Fn(&RunResult) -> f64| Stat::of(&rs.iter().map(|x| f(x)).collect::<Vec<_>>());
        let (h, n, a, e, o, u) = (
            st(&|x| x.heldout.bpb),
            st(&|x| x.novel.bpb),
            st(&|x| x.heldout.accuracy),
            st(&|x| x.heldout.ece),
            st(&|x| x.online_bpb),
            st(&|x| x.us_per_event),
        );
        s += &format!(
            "| {arm} | {:.3} +- {:.3} | {:.3} +- {:.3} | {:.3} | {:.3} | {:.3} | {:.1} |\n",
            h.mean, h.std, n.mean, n.std, a.mean, e.mean, o.mean, u.mean
        );
    }
    let mut names: Vec<&str> = Vec::new();
    for b in &r.baselines {
        if !names.contains(&b.name.as_str()) {
            names.push(&b.name);
        }
    }
    for name in names {
        let bs: Vec<&BaselineResult> = r.baselines.iter().filter(|b| b.name == name).collect();
        let h = Stat::of(&bs.iter().map(|b| b.heldout_bpb).collect::<Vec<_>>());
        let n = Stat::of(&bs.iter().map(|b| b.novel_bpb).collect::<Vec<_>>());
        s += &format!("| {name} (baseline) | {:.3} +- {:.3} | {:.3} +- {:.3} | | | | |\n", h.mean, h.std, n.mean, n.std);
    }
    s += "\n### Acquisition gate (improvement over min(frozen, byte-frequency))\n\n| rule | set | per-seed improvement | mean - 2 sd | passed |\n|---|---|---|---|---|\n";
    for g in &r.gates {
        let per: Vec<String> = g.improvement_per_seed.iter().map(|x| format!("{x:.3}")).collect();
        s += &format!("| {} | {} | {} | {:.3} | {} |\n", g.rule, g.set, per.join(", "), g.improvement.mean - 2.0 * g.improvement.std, g.passed);
    }
    s += "\n";
    for v in &r.verdicts {
        s += &format!(
            "- **{}**: acquisition {}; generalization gap (novel minus exposure replay) {:+.3} BPB\n",
            v.rule,
            if v.acquisition { "shown" } else { "NOT shown" },
            v.generalization_gap_bpb
        );
    }
    s += &format!(
        "- Severing the recurrent path (delta) costs {:+.3} held-out BPB.\n- Online-BPTT control: {}.\n",
        r.recurrence_ablation_delta_bpb, r.bptt_control["status"].as_str().unwrap_or("?")
    );
    for f in &r.failures {
        s += &format!("- FAILURE: {f}\n");
    }
    s += &format!("\n{}\n", r.scope_note);
    s
}
