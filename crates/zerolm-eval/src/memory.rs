//! Memory benchmark: weight-resident associative memory.
//!
//! Per seed: >= 1,000 generated facts go through the live update API
//! (`teach`), then confirm / contradict / supersede / forget operations,
//! an atomic save, a fresh load (restart), and evaluation that uses only the
//! loaded checkpoint tensors. Exact-key recall, paraphrase retrieval (trivial
//! and non-trivial for this encoder), and abstention on negatives are scored
//! separately. Also: interference and capacity curves, a precision/recall
//! sweep over the read threshold, fast-weight capacity, latency, causal
//! ablations, compaction, corrupt/interrupted checkpoints, and a check that
//! no fact exists outside the checkpoint tensors.

use crate::online::Stat;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;
use zerolm_memory::checkpoint::{self, Fault, LoadSource};
use zerolm_memory::{Ablation, MemoryConfig, MemoryError, MemoryStore, Outcome};
use zerolm_tensor::{shuffle, Rng};

pub const SPEC: &str = "model/specs/memory.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub exact_recall_min: f64,
    pub exact_precision_min: f64,
    pub paraphrase_recall_min: f64,
    pub paraphrase_precision_min: f64,
    pub negative_false_accept_max: f64,
    pub ablation_recall_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub version: u32,
    pub config: MemoryConfig,
    pub facts: usize,
    pub milestones: Vec<usize>,
    pub probe: usize,
    pub confirm_every: usize,
    pub contradict: usize,
    pub supersede: usize,
    pub forget: usize,
    pub negatives_per_category: usize,
    pub capacity_levels: Vec<usize>,
    pub fast_loads: Vec<usize>,
    pub threshold_sweep: Vec<f32>,
    pub pilot_seeds: Vec<u64>,
    pub main_seeds: Vec<u64>,
    pub gates: Gates,
    pub gate_rationale: String,
}

impl Spec {
    pub fn load(root: &Path) -> Result<Spec, String> {
        let t = std::fs::read_to_string(root.join(SPEC)).map_err(|e| format!("{SPEC}: {e}"))?;
        let s: Spec = serde_json::from_str(&t).map_err(|e| format!("{SPEC}: {e}"))?;
        s.validate()?;
        Ok(s)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        if self.facts < 1000 {
            return Err("the roadmap requires at least 1,000 facts".into());
        }
        if self.main_seeds.len() < 3 || self.main_seeds.iter().any(|s| self.pilot_seeds.contains(s)) {
            return Err("need >= 3 main seeds, disjoint from pilot seeds".into());
        }
        if self.milestones.windows(2).any(|w| w[0] >= w[1]) || self.milestones.last() != Some(&self.facts) {
            return Err("milestones must increase and end at facts".into());
        }
        if self.contradict + self.supersede + self.forget > self.facts || self.probe > self.facts {
            return Err("mutation and probe counts exceed facts".into());
        }
        if self.facts + self.supersede > self.config.slots {
            return Err("facts + supersessions must fit in the slots".into());
        }
        let mut lv = vec![self.facts];
        lv.extend(&self.capacity_levels);
        if lv.windows(2).any(|w| w[0] >= w[1]) || self.capacity_levels.last() != Some(&self.config.slots) {
            return Err("capacity_levels must increase from facts and end at slots".into());
        }
        if self.threshold_sweep.windows(2).any(|w| w[0] >= w[1]) {
            return Err("threshold_sweep must increase".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- facts

pub const ATTRIBUTES: [&str; 12] =
    ["color", "home", "friend", "food", "size", "shape", "sound", "owner", "maker", "origin", "enemy", "pet"];
pub const FACTS_PER_ENTITY: usize = 3;

/// Paraphrase categories. `trivial` ones normalize to the same key under this
/// encoder (case/punctuation/word order), so they test the normalizer, not
/// retrieval robustness; they are reported but not gated.
pub const PARAPHRASES: [(&str, bool); 6] =
    [("surface", true), ("reorder", true), ("possessive", false), ("filler", false), ("question", false), ("typo", false)];
pub const NEGATIVES: [&str; 3] = ["near_miss_attribute", "near_miss_entity", "unrelated"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    pub entity: String,
    pub attr: String,
    pub value: String,
}

impl Fact {
    pub fn key(&self) -> String {
        format!("{} {}", self.entity, self.attr)
    }
}

fn pseudo(rng: &mut Rng, syllables: usize) -> String {
    const C: &[u8] = b"bdfgklmnprstvz";
    const V: &[u8] = b"aeiou";
    let mut s = String::new();
    for _ in 0..syllables {
        s.push(C[rng.below(C.len())] as char);
        s.push(V[rng.below(V.len())] as char);
        if rng.below(3) == 0 {
            s.push(C[rng.below(C.len())] as char);
        }
    }
    s
}

pub struct FactSet {
    pub facts: Vec<Fact>,
    /// Fresh words (never used as entities, attributes, or values' words).
    pub unseen_entities: Vec<String>,
    pub unseen_attrs: Vec<String>,
    words: BTreeSet<String>,
}

impl FactSet {
    pub fn generate(seed: u64, n: usize, negatives: usize) -> FactSet {
        let mut rng = Rng::from_tag(seed, "memory-facts");
        let mut words: BTreeSet<String> = ATTRIBUTES.iter().map(|s| s.to_string()).collect();
        let fresh = |rng: &mut Rng, words: &mut BTreeSet<String>, syl: usize| loop {
            let w = pseudo(rng, syl);
            if w.len() >= 4 && words.insert(w.clone()) {
                return w;
            }
        };
        let mut facts = Vec::with_capacity(n);
        while facts.len() < n {
            let entity = fresh(&mut rng, &mut words, 3);
            let mut attrs = ATTRIBUTES.to_vec();
            shuffle(&mut attrs, rng.next_u64());
            for a in attrs.into_iter().take(FACTS_PER_ENTITY) {
                if facts.len() == n {
                    break;
                }
                let value = if rng.below(2) == 0 {
                    fresh(&mut rng, &mut words, 2)
                } else {
                    format!("{} {}", fresh(&mut rng, &mut words, 2), fresh(&mut rng, &mut words, 2))
                };
                facts.push(Fact { entity: entity.clone(), attr: a.to_string(), value });
            }
        }
        let unseen_entities = (0..negatives).map(|_| fresh(&mut rng, &mut words, 3)).collect();
        let unseen_attrs = (0..negatives).map(|_| fresh(&mut rng, &mut words, 2)).collect();
        FactSet { facts, unseen_entities, unseen_attrs, words }
    }

    /// A value word that is new and differs from `old`.
    pub fn new_value(&mut self, rng: &mut Rng) -> String {
        loop {
            let w = pseudo(rng, 2);
            if w.len() >= 4 && self.words.insert(w.clone()) {
                return w;
            }
        }
    }

    pub fn paraphrase(&self, cat: &str, f: &Fact, rng: &mut Rng) -> String {
        let (e, a) = (&f.entity, &f.attr);
        match cat {
            "surface" => format!("{}, {}?", capitalize(e), a.to_uppercase()),
            "reorder" => format!("{a} {e}"),
            "possessive" => format!("{e}'s {a}"),
            "filler" => format!("the {a} of {e}"),
            "question" => format!("what is the {a} of {e}?"),
            "typo" => loop {
                let i = 1 + rng.below(e.len() - 2);
                let t = format!("{}{}", &e[..i], &e[i + 1..]);
                if !self.words.contains(&t) {
                    return format!("{t} {a}");
                }
            },
            _ => unreachable!("unknown paraphrase category {cat}"),
        }
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

// ---------------------------------------------------------------- scoring

/// One evaluated query. `expected = None` means the memory should abstain.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryRecord {
    pub category: &'static str,
    pub query: String,
    pub score: f32,
    pub coverage: f32,
    pub decoded: String,
    pub accepted: bool,
    pub pooled_ok: bool,
    pub expected: Option<String>,
}

impl QueryRecord {
    /// Accepted at coverage threshold `tau` (the pooled-score threshold is
    /// fixed at `read_threshold`, recorded in `pooled_ok`).
    fn accepted_at(&self, tau: f32) -> bool {
        self.pooled_ok && self.coverage >= tau && !self.decoded.is_empty()
    }

}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Score {
    pub queries: usize,
    /// Positives: correct answers / queries. Negatives: unused (0).
    pub recall: f64,
    /// Positives: correct answers / answered.
    pub precision: f64,
    pub abstain_rate: f64,
    /// Negatives: answered / queries.
    pub false_accept_rate: f64,
}

/// Scores with the decision re-applied at coverage threshold `tau` (for sweeps).
pub fn score(records: &[&QueryRecord], tau: f32) -> Score {
    score_by(records, |r| r.accepted_at(tau))
}

/// Scores with the circuit's own accept decision.
pub fn score_circuit(records: &[&QueryRecord]) -> Score {
    score_by(records, |r| r.accepted)
}

fn score_by(records: &[&QueryRecord], accepted: impl Fn(&QueryRecord) -> bool) -> Score {
    let n = records.len();
    if n == 0 {
        return Score::default();
    }
    let hit = |r: &QueryRecord| accepted(r) && r.expected.as_deref() == Some(r.decoded.as_str());
    let acc = records.iter().filter(|r| accepted(r)).count();
    let pos: Vec<_> = records.iter().filter(|r| r.expected.is_some()).collect();
    let neg = n - pos.len();
    let hits = pos.iter().filter(|r| hit(r)).count();
    let pos_acc = pos.iter().filter(|r| accepted(r)).count();
    let neg_acc = acc - pos_acc;
    Score {
        queries: n,
        recall: if pos.is_empty() { 0.0 } else { hits as f64 / pos.len() as f64 },
        precision: if pos_acc == 0 { 1.0 } else { hits as f64 / pos_acc as f64 },
        abstain_rate: (n - acc) as f64 / n as f64,
        false_accept_rate: if neg == 0 { 0.0 } else { neg_acc as f64 / neg as f64 },
    }
}

fn run_query(m: &MemoryStore, category: &'static str, text: &str, expected: Option<&str>) -> QueryRecord {
    let a = m.query(text).expect("non-empty query");
    QueryRecord {
        category,
        query: text.to_string(),
        score: a.score,
        coverage: a.coverage,
        pooled_ok: a.score >= m.config().read_threshold,
        decoded: a.decoded,
        accepted: a.value.is_some(),
        expected: expected.map(str::to_string),
    }
}

// ---------------------------------------------------------------- results

#[derive(Debug, Clone, Serialize)]
pub struct CurvePoint {
    pub facts_stored: usize,
    /// Exact-key recall over every fact stored so far.
    pub exact: Score,
    /// Non-trivial paraphrases of the first `probe` facts (interference probe).
    pub probe_paraphrase: Score,
    /// Mean (best - runner-up) cosine over exact probe queries.
    pub probe_mean_margin: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CategoryScore {
    pub category: &'static str,
    pub kind: &'static str,
    pub score: Score,
}

#[derive(Debug, Clone, Serialize)]
pub struct SweepPoint {
    pub threshold: f32,
    pub exact: Score,
    pub paraphrase: Score,
    pub negatives: Score,
}

#[derive(Debug, Clone, Serialize)]
pub struct FastPoint {
    pub associations: usize,
    pub recall: f64,
    /// Recall of the first association after all `associations` writes.
    pub first_recalled: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Latency {
    pub write_p50_us: f64,
    pub write_p95_us: f64,
    pub query_p50_us: f64,
    pub query_p95_us: f64,
    pub save_ms: f64,
    pub load_ms: f64,
    pub checkpoint_bytes: usize,
    pub active_facts_at_query: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeedResult {
    pub seed: u64,
    pub facts: usize,
    pub facts_sha256: String,
    pub initial_core_sha256: String,
    pub final_core_sha256: String,
    pub initial_memory_sha256: String,
    pub final_memory_sha256: String,
    pub checkpoint_sha256: String,
    pub checkpoint_files: Vec<String>,
    pub tensor_groups: Vec<GroupDigest>,
    pub categories: Vec<CategoryScore>,
    pub exact: Score,
    pub paraphrase: Score,
    pub negatives: Score,
    pub interference: Vec<CurvePoint>,
    pub capacity: Vec<CurvePoint>,
    pub sweep: Vec<SweepPoint>,
    pub fast_capacity: Vec<FastPoint>,
    pub ablations: Vec<(String, Score)>,
    /// Each block of the compiled read circuit zeroed in turn.
    pub circuit_ablations: Vec<CircuitAblation>,
    pub reference_parity: Parity,
    pub latency: Latency,
    pub checks: Vec<Check>,
    pub update_events: usize,
}

/// The independent reference interpreter rerun on every evaluated query.
#[derive(Debug, Clone, Serialize)]
pub struct Parity {
    pub queries: usize,
    /// Queries where any output differs in any bit.
    pub mismatches: usize,
    pub bundle_sha256: String,
    pub circuit_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CircuitAblation {
    pub block: String,
    pub exact: Score,
    pub negatives: Score,
}

#[derive(Debug, Clone, Serialize)]
pub struct GroupDigest {
    pub tensor: String,
    pub group: String,
    pub changed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Report kind (always "memory").
    pub report: &'static str,
    pub mode: &'static str,
    pub passed: bool,
    pub failures: Vec<String>,
    pub spec: Spec,
    pub spec_sha256: String,
    pub seeds: Vec<SeedResult>,
    pub summary: Summary,
    pub scope_note: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub exact_recall: Stat,
    pub exact_precision: Stat,
    pub paraphrase_recall: Stat,
    pub paraphrase_precision: Stat,
    pub negative_false_accept: Stat,
    pub query_p95_us: Stat,
}

pub const SCOPE_NOTE: &str = "Memory here is slot-structured tensors in the model checkpoint read by a programmed attention computation; the key encoder is a frozen seeded random-feature embedding bag (case-, punctuation- and word-order-invariant, no learned synonymy). Values are reconstructed through the core byte embedding. Facts are generated pseudo-word triples, not natural language. This measures persistent storage and retrieval, not understanding or reasoning.";

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn us(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e6
}

struct Eval<'a> {
    set: &'a FactSet,
    spec: &'a Spec,
    /// key -> expected value after all mutations (None = forgotten).
    expected: &'a BTreeMap<String, Option<String>>,
    seed: u64,
}

impl Eval<'_> {
    /// Every query of the full evaluation, in a fixed order.
    fn queries(&self) -> Vec<(&'static str, String, Option<String>)> {
        let mut out = Vec::new();
        let mut rng = Rng::from_tag(self.seed, "memory-paraphrase");
        for f in &self.set.facts {
            let exp = self.expected[&f.key()].clone();
            let cat = if exp.is_some() { "exact" } else { "forgotten" };
            out.push((cat, f.key(), exp.clone()));
            if exp.is_none() {
                continue;
            }
            for (c, _) in PARAPHRASES {
                out.push((c, self.set.paraphrase(c, f, &mut rng), exp.clone()));
            }
        }
        let n = self.spec.negatives_per_category;
        let taught: BTreeSet<(String, String)> = self.set.facts.iter().map(|f| (f.entity.clone(), f.attr.clone())).collect();
        let mut i = 0;
        let mut k = 0;
        while k < n && i < self.set.facts.len() * ATTRIBUTES.len() {
            let f = &self.set.facts[(i * 7919) % self.set.facts.len()];
            let a = ATTRIBUTES[i % ATTRIBUTES.len()];
            i += 1;
            if !taught.contains(&(f.entity.clone(), a.to_string())) {
                out.push(("near_miss_attribute", format!("{} {a}", f.entity), None));
                k += 1;
            }
        }
        for (j, e) in self.set.unseen_entities.iter().enumerate().take(n) {
            out.push(("near_miss_entity", format!("{e} {}", ATTRIBUTES[j % ATTRIBUTES.len()]), None));
        }
        for (e, a) in self.set.unseen_entities.iter().zip(&self.set.unseen_attrs).take(n) {
            out.push(("unrelated", format!("{a} {e}"), None));
        }
        out
    }

    fn run(&self, m: &MemoryStore, times: Option<&mut Vec<f64>>) -> Vec<QueryRecord> {
        let qs = self.queries();
        let mut out = Vec::with_capacity(qs.len());
        let mut t = Vec::new();
        for (c, q, e) in &qs {
            let t0 = Instant::now();
            out.push(run_query(m, c, q, e.as_deref()));
            t.push(us(t0));
        }
        if let Some(times) = times {
            *times = t;
        }
        out
    }
}

pub fn kind_of(cat: &str) -> &'static str {
    match cat {
        "exact" => "exact",
        c if PARAPHRASES.iter().any(|(p, t)| *p == c && *t) => "paraphrase_trivial",
        c if PARAPHRASES.iter().any(|(p, _)| *p == c) => "paraphrase",
        _ => "negative",
    }
}

fn by_kind<'a>(r: &'a [QueryRecord], kind: &str) -> Vec<&'a QueryRecord> {
    r.iter().filter(|q| kind_of(q.category) == kind).collect()
}

fn categories(r: &[QueryRecord], tau: f32) -> Vec<CategoryScore> {
    let mut names: Vec<&'static str> = vec!["exact"];
    names.extend(PARAPHRASES.iter().map(|(c, _)| *c));
    names.push("forgotten");
    names.extend(NEGATIVES);
    names
        .into_iter()
        .map(|c| CategoryScore { category: c, kind: kind_of(c), score: score(&r.iter().filter(|q| q.category == c).collect::<Vec<_>>(), tau) })
        .collect()
}

fn curve_point(m: &MemoryStore, set: &FactSet, expected: &dyn Fn(&Fact) -> Option<String>, stored: usize, probe: usize, seed: u64) -> CurvePoint {
    let tau = m.config().coverage_threshold;
    let exact: Vec<_> = set.facts[..stored.min(set.facts.len())].iter().map(|f| run_query(m, "exact", &f.key(), expected(f).as_deref())).collect();
    let mut rng = Rng::from_tag(seed, "memory-probe");
    let mut para = Vec::new();
    let mut margin = 0.0;
    let probe_facts = &set.facts[..probe.min(stored)];
    for f in probe_facts {
        let a = m.query(&f.key()).expect("key");
        margin += (a.score - a.runner_up.max(0.0)) as f64;
        for (c, trivial) in PARAPHRASES {
            if !trivial {
                para.push(run_query(m, c, &set.paraphrase(c, f, &mut rng), expected(f).as_deref()));
            }
        }
    }
    CurvePoint {
        facts_stored: stored,
        exact: score(&exact.iter().collect::<Vec<_>>(), tau),
        probe_paraphrase: score(&para.iter().collect::<Vec<_>>(), tau),
        probe_mean_margin: margin / probe_facts.len().max(1) as f64,
    }
}

/// Rerun every evaluated query on the reference interpreter from the exported
/// read bundle and compare every output bit for bit with the engine.
fn reference_parity(m: &MemoryStore, ev: &Eval<'_>) -> Result<Parity, String> {
    use crate::isa_ref::{self, T};
    let bundle = m.read_bundle();
    let (mut queries, mut mismatches) = (0, 0);
    for (_, q, _) in ev.queries() {
        let (k, words) = m.query_inputs(&q).map_err(|e| e.to_string())?;
        let engine = m.read_outputs(&k, &words).map_err(|e| e.to_string())?;
        let inputs = [("q", T { r: 1, c: k.len(), d: k.clone() }), ("qw", T { r: words.len(), c: k.len(), d: words.concat() })];
        let reference = isa_ref::run(&bundle, &inputs)?;
        let same = engine.values.iter().all(|(n, b)| {
            reference.get(n).map(|t| t.r == b.rows && t.c == b.cols && t.d.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits())).unwrap_or(false)
        });
        queries += 1;
        mismatches += !same as usize;
    }
    Ok(Parity {
        queries,
        mismatches,
        bundle_sha256: zerolm_engine::isa::bundle::sha256(&bundle),
        circuit_sha256: m.read_circuit().sha256(),
    })
}

fn check(checks: &mut Vec<Check>, name: &'static str, passed: bool, detail: impl Into<String>) {
    checks.push(Check { name, passed, detail: detail.into() });
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut f: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    f.sort();
    f
}

/// Printable-ASCII runs of length >= 4 (where any plaintext would appear).
fn printable_runs(bytes: &[u8]) -> Vec<String> {
    bytes
        .split(|b| !(0x20..0x7f).contains(b))
        .filter(|r| r.len() >= 4)
        .map(|r| String::from_utf8_lossy(r).into_owned())
        .collect()
}

fn capacity_full(e: Result<Outcome, MemoryError>) -> bool {
    matches!(e, Err(MemoryError::CapacityFull))
}

/// Run one seed end to end. `work` is a scratch directory for checkpoints.
pub fn run_seed(spec: &Spec, seed: u64, work: &Path) -> Result<SeedResult, String> {
    let cfg = spec.config;
    let tau = cfg.coverage_threshold;
    let stream = format!("memory:synthetic-facts-v1:seed={seed}");
    let mut set = FactSet::generate(seed, spec.facts, spec.negatives_per_category);
    let facts_sha256 = {
        let mut all = String::new();
        for f in &set.facts {
            all += &f.key();
            all.push('\t');
            all += &f.value;
            all.push('\n');
        }
        zerolm_learning::streams::sha256_hex(all.as_bytes())
    };
    let _ = std::fs::remove_dir_all(work);
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    let dir = work.join("checkpoint");
    let mut checks = Vec::new();
    let me = |e: MemoryError| e.to_string();

    let mut m = MemoryStore::new(cfg).map_err(me)?;
    let core0 = m.digest("core.");
    let mem0 = m.digest("memory.");
    let tensors0 = m.tensors();

    // 1. Teach every fact through the live API; interference curve.
    let mut write_us = Vec::new();
    let mut interference = Vec::new();
    let original = |f: &Fact| Some(f.value.clone());
    let mut next_milestone = 0;
    let mut stored_ok = 0;
    for (i, f) in set.facts.iter().enumerate() {
        let t0 = Instant::now();
        let o = m.teach(&f.key(), &f.value, &stream).map_err(me)?;
        write_us.push(us(t0));
        stored_ok += matches!(o, Outcome::Stored { .. }) as usize;
        if spec.milestones.get(next_milestone) == Some(&(i + 1)) {
            interference.push(curve_point(&m, &set, &original, i + 1, spec.probe, seed));
            next_milestone += 1;
        }
    }
    check(&mut checks, "all_facts_stored", stored_ok == spec.facts, format!("{stored_ok}/{} stored as new slots", spec.facts));
    m.save(&dir).map_err(me)?;

    // 2. Mutations: confirm, contradict, supersede, forget (disjoint sets).
    let mut idx: Vec<usize> = (0..spec.facts).collect();
    shuffle(&mut idx, Rng::from_tag(seed, "memory-mutations").next_u64());
    let (contra, rest) = idx.split_at(spec.contradict);
    let (sup, rest) = rest.split_at(spec.supersede);
    let forget = &rest[..spec.forget];
    let mut expected: BTreeMap<String, Option<String>> = set.facts.iter().map(|f| (f.key(), Some(f.value.clone()))).collect();
    let mut vrng = Rng::from_tag(seed, "memory-new-values");
    let mut ok = 0;
    for i in (0..spec.facts).step_by(spec.confirm_every.max(1)) {
        ok += matches!(m.confirm(&set.facts[i].key(), &stream).map_err(me)?, Outcome::Confirmed { .. }) as usize;
    }
    let n_confirm = spec.facts.div_ceil(spec.confirm_every.max(1));
    check(&mut checks, "confirm", ok == n_confirm, format!("{ok}/{n_confirm} confirmed"));
    ok = 0;
    for &i in contra {
        let v = set.new_value(&mut vrng);
        ok += matches!(m.teach(&set.facts[i].key(), &v, &stream).map_err(me)?, Outcome::Conflict { .. }) as usize;
    }
    let contra_kept = contra.iter().filter(|&&i| m.query(&set.facts[i].key()).ok().and_then(|a| a.value) == Some(set.facts[i].value.clone())).count();
    let contra_flag = contra.iter().filter(|&&i| m.query(&set.facts[i].key()).map(|a| a.contradictions == 1).unwrap_or(false)).count();
    check(
        &mut checks,
        "contradiction",
        ok == contra.len() && contra_kept == contra.len() && contra_flag == contra.len(),
        format!("{ok} conflicts reported, {contra_kept} originals kept, {contra_flag} flagged of {}", contra.len()),
    );
    ok = 0;
    for &i in sup {
        let v = set.new_value(&mut vrng);
        ok += matches!(m.correct(&set.facts[i].key(), &v, &stream).map_err(me)?, Outcome::Superseded { .. }) as usize;
        expected.insert(set.facts[i].key(), Some(v));
    }
    check(&mut checks, "supersession", ok == sup.len(), format!("{ok}/{} superseded", sup.len()));
    ok = 0;
    for &i in forget {
        ok += matches!(m.forget(&set.facts[i].key(), &stream).map_err(me)?, Outcome::Forgotten { .. }) as usize;
        expected.insert(set.facts[i].key(), None);
    }
    check(&mut checks, "deletion", ok == forget.len(), format!("{ok}/{} forgotten", forget.len()));
    let core_after_writes = m.digest("core.");
    let tensors1 = m.tensors();

    let ev = Eval { set: &set, spec, expected: &expected, seed };
    let before_restart = ev.run(&m, None);
    let t0 = Instant::now();
    let checkpoint_sha256 = m.save(&dir).map_err(me)?;
    let save_ms = us(t0) / 1e3;
    let update_events = m.events().len();
    let events_bound = m.events().iter().all(|e| e.checkpoint_sha256.is_some());
    check(&mut checks, "events_logged", events_bound && update_events > spec.facts, format!("{update_events} events, all bound to a saved checkpoint: {events_bound}"));
    let stats_before = m.stats();
    drop(m);

    // 3. Restart: a fresh store from the checkpoint only.
    let t0 = Instant::now();
    let (m, info) = MemoryStore::load(&dir).map_err(me)?;
    let load_ms = us(t0) / 1e3;
    check(&mut checks, "restart_from_current", info.source == LoadSource::Current && info.checkpoint_sha256 == checkpoint_sha256, format!("{:?}", info.source));
    let mut query_us = Vec::new();
    let records = ev.run(&m, Some(&mut query_us));
    check(&mut checks, "restart_identical_answers", records == before_restart, "every answer and score bit-identical before and after restart");
    let reference_parity = reference_parity(&m, &ev)?;
    check(
        &mut checks,
        "reference_interpreter_parity",
        reference_parity.mismatches == 0 && reference_parity.queries == records.len(),
        format!("{}/{} queries reproduced bit for bit by the reference interpreter from the exported bundle", reference_parity.queries - reference_parity.mismatches, reference_parity.queries),
    );
    let core1 = m.digest("core.");
    check(
        &mut checks,
        "core_byte_identical",
        core0 == core_after_writes && core0 == core1,
        "core.* tensors identical after memory-only writes and after reload",
    );
    let tensor_groups: Vec<GroupDigest> = tensors0
        .iter()
        .zip(&tensors1)
        .map(|(a, b)| GroupDigest { tensor: a.name.clone(), group: a.group.clone(), changed: a.data.iter().zip(&b.data).any(|(x, y)| x.to_bits() != y.to_bits()) })
        .collect();
    let changed_core = tensor_groups.iter().filter(|g| g.tensor.starts_with("core.") && g.changed).count();
    let changed_semantic = tensor_groups.iter().filter(|g| g.tensor.starts_with("memory.semantic.") && g.changed).count();
    check(&mut checks, "writes_land_in_memory_tensors", changed_core == 0 && changed_semantic == 4, format!("{changed_core} core tensors changed, {changed_semantic}/4 semantic tensors changed"));

    // 4. Facts live only in checkpoint tensors.
    let files = files_in(&dir);
    let only_ckpt = files.iter().all(|f| f == checkpoint::CURRENT || f == checkpoint::PREVIOUS);
    // Structural text (header JSON, tensor names) is present in an empty
    // model's checkpoint too; only strings absent from it can be a leak.
    let baseline = printable_runs(&MemoryStore::new(cfg).map_err(me)?.checkpoint_bytes());
    let mut leaks = BTreeSet::new();
    for f in &files {
        let runs = printable_runs(&std::fs::read(dir.join(f)).map_err(|e| e.to_string())?);
        for fact in &set.facts {
            for s in [&fact.value, &fact.entity] {
                if runs.iter().any(|r| r.contains(s.as_str())) && !baseline.iter().any(|r| r.contains(s.as_str())) {
                    leaks.insert(s.clone());
                }
            }
        }
    }
    let leaked = leaks.len();
    check(&mut checks, "facts_only_in_tensors", only_ckpt && leaked == 0, format!("files {files:?}; {leaked} plaintext fact strings found {leaks:?}"));
    let mut erased = 0;
    let t = m.tensors();
    let values = t.iter().find(|t| t.name == zerolm_memory::persistent_memory::VALUES).expect("values tensor");
    let meta = t.iter().find(|t| t.name == zerolm_memory::persistent_memory::META).expect("meta tensor");
    for s in 0..cfg.slots {
        if meta.data[s * meta.cols] == zerolm_memory::persistent_memory::DELETED {
            erased += values.data[s * values.cols..(s + 1) * values.cols].iter().all(|v| *v == 0.0) as usize;
        }
    }
    check(&mut checks, "deleted_values_erased", erased == stats_before.deleted && erased >= forget.len(), format!("{erased} deleted slots, all value rows zero"));

    // 5. Scores at the frozen threshold. The sweep re-applies the decision in
    // the evaluator; at the frozen threshold it must equal the circuit's.
    let same_decision = records.iter().all(|r| r.accepted == r.accepted_at(tau));
    check(&mut checks, "evaluator_matches_circuit_decision", same_decision, "re-applied accept decision equals the circuit's accept output on every query");
    let cats = categories(&records, tau);
    let exact = score(&by_kind(&records, "exact"), tau);
    let paraphrase = score(&by_kind(&records, "paraphrase"), tau);
    let negatives = score(&by_kind(&records, "negative"), tau);
    let forgotten = cats.iter().find(|c| c.category == "forgotten").expect("forgotten").score;
    let leaked_old = records
        .iter()
        .filter(|q| q.category == "forgotten" && q.accepted)
        .filter(|q| set.facts.iter().any(|f| f.key() == q.query && f.value == q.decoded))
        .count();
    check(
        &mut checks,
        "forgotten_value_unrecoverable",
        leaked_old == 0 && forgotten.false_accept_rate <= spec.gates.negative_false_accept_max,
        format!("{:.4} of forgotten keys answered; {leaked_old} answers returned the forgotten value", forgotten.false_accept_rate),
    );
    let sweep = spec
        .threshold_sweep
        .iter()
        .map(|&th| SweepPoint {
            threshold: th,
            exact: score(&by_kind(&records, "exact"), th),
            paraphrase: score(&by_kind(&records, "paraphrase"), th),
            negatives: score(&by_kind(&records, "negative"), th),
        })
        .collect();

    // 6. Causal ablations (each on a fresh load).
    let mut ablations = Vec::new();
    for (name, a) in [("zero_values", Ablation::ZeroValues), ("zero_keys", Ablation::ZeroKeys), ("shuffle_keys", Ablation::ShuffleKeys(seed))] {
        let (mut x, _) = MemoryStore::load(&dir).map_err(me)?;
        x.ablate(a);
        let r: Vec<QueryRecord> = set
            .facts
            .iter()
            .filter_map(|f| expected[&f.key()].as_ref().map(|v| run_query(&x, "exact", &f.key(), Some(v))))
            .collect();
        let s = score(&r.iter().collect::<Vec<_>>(), tau);
        check(&mut checks, "ablation_damages_recall", s.recall <= spec.gates.ablation_recall_max, format!("{name}: exact recall {:.4}", s.recall));
        ablations.push((name.to_string(), s));
    }

    // 6b. Circuit-block ablations: verification is what makes the memory
    // abstain, so zeroing it must keep exact answers and accept negatives;
    // every other block carries the answers themselves.
    let mut circuit_ablations = Vec::new();
    let blocks = m.read_circuit().blocks();
    for block in blocks.iter().filter(|b| *b != "support") {
        let (mut x, _) = MemoryStore::load(&dir).map_err(me)?;
        x.ablate_circuit(block).map_err(|e| e.to_string())?;
        let r = ev.run(&x, None);
        let (ex, ng) = (score_circuit(&by_kind(&r, "exact")), score_circuit(&by_kind(&r, "negative")));
        let ok = if block == "verify" {
            ex.recall >= exact.recall && ng.false_accept_rate > spec.gates.negative_false_accept_max
        } else {
            ex.recall <= spec.gates.ablation_recall_max
        };
        check(&mut checks, "circuit_block_ablation", ok, format!("{block}: exact recall {:.4} (intact {:.4}), negative false accepts {:.4}", ex.recall, exact.recall, ng.false_accept_rate));
        circuit_ablations.push(CircuitAblation { block: block.clone(), exact: ex, negatives: ng });
    }

    // 7. Compaction preserves answers and frees superseded/deleted slots.
    let mut m = m;
    let exact_keys: Vec<String> = set.facts.iter().map(|f| f.key()).collect();
    let before: Vec<_> = exact_keys.iter().map(|k| m.query(k).map(|a| a.value).unwrap_or(None)).collect();
    let st = m.stats();
    let freed = m.compact(&stream);
    let after: Vec<_> = exact_keys.iter().map(|k| m.query(k).map(|a| a.value).unwrap_or(None)).collect();
    check(&mut checks, "compaction", freed == st.superseded + st.deleted && before == after && m.stats().free == cfg.slots - st.active, format!("freed {freed} slots; answers unchanged: {}", before == after));

    // 8. Capacity: fill every slot, then a write must be refused cleanly.
    let mut capacity = Vec::new();
    let mut crng = Rng::from_tag(seed, "memory-capacity");
    let extra_needed = cfg.slots - m.stats().active;
    let mut extra: Vec<Fact> = Vec::with_capacity(extra_needed);
    while extra.len() < extra_needed {
        let e = set.new_value(&mut crng);
        let v = set.new_value(&mut crng);
        extra.push(Fact { entity: e, attr: ATTRIBUTES[extra.len() % ATTRIBUTES.len()].into(), value: v });
    }
    let mut lvl = 0;
    let active0 = m.stats().active;
    let current = |f: &Fact| expected[&f.key()].clone();
    let survivors = FactSet {
        facts: set.facts.iter().filter(|f| expected[&f.key()].is_some()).cloned().collect(),
        unseen_entities: Vec::new(),
        unseen_attrs: Vec::new(),
        words: set.words.clone(),
    };
    for (j, f) in extra.iter().enumerate() {
        if !matches!(m.teach(&f.key(), &f.value, &stream).map_err(me)?, Outcome::Stored { .. }) {
            return Err(format!("capacity fill: {} was not stored", f.key()));
        }
        // After compaction every slot in use holds an active fact.
        let filled = active0 + j + 1;
        while lvl < spec.capacity_levels.len() && filled >= spec.capacity_levels[lvl] {
            capacity.push(curve_point(&m, &survivors, &current, survivors.facts.len(), spec.probe, seed));
            capacity.last_mut().expect("pushed").facts_stored = filled;
            lvl += 1;
        }
    }
    let digest = m.digest("memory.");
    let refused = capacity_full(m.teach("overflow fact", "nope", &stream));
    check(&mut checks, "capacity_refuses_cleanly", refused && m.digest("memory.") == digest && m.stats().free == 0, format!("{} slots full; overflow refused: {refused}", cfg.slots));

    // 9. Corrupt and interrupted checkpoints.
    let cdir = work.join("corrupt");
    let mut c = MemoryStore::new(cfg).map_err(me)?;
    c.teach("alpha key", "one", &stream).map_err(me)?;
    let first = c.save(&cdir).map_err(me)?;
    c.teach("beta key", "two", &stream).map_err(me)?;
    c.save(&cdir).map_err(me)?;
    let cur = cdir.join(checkpoint::CURRENT);
    let mut bytes = std::fs::read(&cur).map_err(|e| e.to_string())?;
    let mid = bytes.len() / 3;
    bytes[mid] ^= 0x04;
    std::fs::write(&cur, &bytes).map_err(|e| e.to_string())?;
    let flip_ok = matches!(MemoryStore::load(&cdir), Ok((_, i)) if i.source == LoadSource::Previous && i.checkpoint_sha256 == first);
    std::fs::write(cdir.join(checkpoint::PREVIOUS), &bytes[..bytes.len() / 2]).map_err(|e| e.to_string())?;
    let both_fail = MemoryStore::load(&cdir).is_err();
    let mut faults_ok = 0;
    for fault in [Fault::DuringTempWrite, Fault::BeforeRename, Fault::BetweenRenames] {
        let fdir = work.join(format!("fault-{fault:?}"));
        let mut s = MemoryStore::new(cfg).map_err(me)?;
        s.teach("alpha key", "one", &stream).map_err(me)?;
        let committed = s.save(&fdir).map_err(me)?;
        s.teach("beta key", "two", &stream).map_err(me)?;
        let crashed = s.save_with_fault(&fdir, fault).is_err();
        faults_ok += matches!(MemoryStore::load(&fdir), Ok((r, i)) if crashed && i.checkpoint_sha256 == committed
            && r.query("beta key").map(|a| a.value.is_none()).unwrap_or(false)) as usize;
    }
    check(&mut checks, "corrupt_checkpoint_recovery", flip_ok && both_fail && faults_ok == 3, format!("bit flip -> previous: {flip_ok}; both corrupt -> refused: {both_fail}; interrupted writes recovered: {faults_ok}/3"));

    // 10. W_FAST capacity and decay.
    let mut fast_capacity = Vec::new();
    for &n in &spec.fast_loads {
        let mut f = MemoryStore::new(cfg).map_err(me)?;
        let items = &set.facts[..n.min(set.facts.len())];
        for it in items {
            f.associate_fast(&it.key(), &it.value, &stream).map_err(me)?;
        }
        let hits: Vec<bool> = items.iter().map(|it| f.recall_fast(&it.key()).map(|d| d.bytes == it.value.as_bytes()).unwrap_or(false)).collect();
        fast_capacity.push(FastPoint { associations: n, recall: hits.iter().filter(|h| **h).count() as f64 / n as f64, first_recalled: hits[0] });
    }

    let latency = Latency {
        write_p50_us: pct(&mut write_us.clone(), 0.5),
        write_p95_us: pct(&mut write_us, 0.95),
        query_p50_us: pct(&mut query_us.clone(), 0.5),
        query_p95_us: pct(&mut query_us, 0.95),
        save_ms,
        load_ms,
        checkpoint_bytes: std::fs::metadata(&cur).map(|m| m.len() as usize).unwrap_or(0),
        active_facts_at_query: stats_before.active,
    };
    let final_memory_sha256 = m.digest("memory.");
    let _ = std::fs::remove_dir_all(work);
    Ok(SeedResult {
        seed,
        facts: spec.facts,
        facts_sha256,
        initial_core_sha256: core0,
        final_core_sha256: m.digest("core."),
        initial_memory_sha256: mem0,
        final_memory_sha256,
        checkpoint_sha256,
        checkpoint_files: files,
        tensor_groups,
        categories: cats,
        exact,
        paraphrase,
        negatives,
        interference,
        capacity,
        sweep,
        fast_capacity,
        ablations,
        circuit_ablations,
        reference_parity,
        latency,
        checks,
        update_events,
    })
}

/// Run every seed (in parallel threads; results are seed-deterministic).
pub fn run(root: &Path, spec: &Spec, seeds: &[u64], gate: bool, work: &Path) -> Result<Report, String> {
    let spec_sha256 = zerolm_learning::streams::sha256_hex(&std::fs::read(root.join(SPEC)).map_err(|e| e.to_string())?);
    let results: Vec<Result<SeedResult, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&seed| {
                let w: PathBuf = work.join(format!("seed-{seed}"));
                s.spawn(move || run_seed(spec, seed, &w))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("seed thread panicked".into()))).collect()
    });
    let seeds_out: Vec<SeedResult> = results.into_iter().collect::<Result<_, _>>()?;
    let g = &spec.gates;
    let mut failures = Vec::new();
    for r in &seeds_out {
        for c in r.checks.iter().filter(|c| !c.passed) {
            failures.push(format!("seed {}: {} failed ({})", r.seed, c.name, c.detail));
        }
        let mut gate_fail = |ok: bool, what: String| {
            if !ok {
                failures.push(format!("seed {}: {what}", r.seed));
            }
        };
        gate_fail(r.exact.recall >= g.exact_recall_min, format!("exact recall {:.4} < {}", r.exact.recall, g.exact_recall_min));
        gate_fail(r.exact.precision >= g.exact_precision_min, format!("exact precision {:.4} < {}", r.exact.precision, g.exact_precision_min));
        gate_fail(r.paraphrase.recall >= g.paraphrase_recall_min, format!("paraphrase recall {:.4} < {}", r.paraphrase.recall, g.paraphrase_recall_min));
        gate_fail(r.paraphrase.precision >= g.paraphrase_precision_min, format!("paraphrase precision {:.4} < {}", r.paraphrase.precision, g.paraphrase_precision_min));
        gate_fail(r.negatives.false_accept_rate <= g.negative_false_accept_max, format!("negative false accept {:.4} > {}", r.negatives.false_accept_rate, g.negative_false_accept_max));
    }
    let stat = |f: &dyn Fn(&SeedResult) -> f64| Stat::of(&seeds_out.iter().map(f).collect::<Vec<_>>());
    let summary = Summary {
        exact_recall: stat(&|r| r.exact.recall),
        exact_precision: stat(&|r| r.exact.precision),
        paraphrase_recall: stat(&|r| r.paraphrase.recall),
        paraphrase_precision: stat(&|r| r.paraphrase.precision),
        negative_false_accept: stat(&|r| r.negatives.false_accept_rate),
        query_p95_us: stat(&|r| r.latency.query_p95_us),
    };
    Ok(Report {
        report: "memory",
        mode: if gate { "main" } else { "pilot" },
        passed: gate && failures.is_empty(),
        failures,
        spec: spec.clone(),
        spec_sha256,
        seeds: seeds_out,
        summary,
        scope_note: SCOPE_NOTE,
    })
}

pub fn summary_md(r: &Report) -> String {
    let mut s = format!("## Weight-resident memory ({} mode)\n\n", r.mode);
    s += &format!("**{}**", if r.passed { "PASSED" } else if r.mode == "pilot" { "PILOT (not gated)" } else { "FAILED" });
    s += &format!(" | seeds {:?} | {} facts per seed | pooled threshold {} | coverage threshold {}\n\n", r.seeds.iter().map(|x| x.seed).collect::<Vec<_>>(), r.spec.facts, r.spec.config.read_threshold, r.spec.config.coverage_threshold);
    let st = |x: &Stat| format!("{:.4} ± {:.4}", x.mean, x.std);
    s += "| metric (after restart) | mean ± sd |\n|---|---|\n";
    s += &format!("| exact-key recall | {} |\n| exact-key precision | {} |\n", st(&r.summary.exact_recall), st(&r.summary.exact_precision));
    s += &format!("| paraphrase recall (non-trivial) | {} |\n| paraphrase precision | {} |\n", st(&r.summary.paraphrase_recall), st(&r.summary.paraphrase_precision));
    s += &format!("| negative false-accept rate | {} |\n| query p95 (µs) | {:.1} |\n\n", st(&r.summary.negative_false_accept), r.summary.query_p95_us.mean);
    if let Some(x) = r.seeds.first() {
        s += &format!("### Seed {} by category\n\n| category | kind | queries | recall | precision | abstain | false accept |\n|---|---|---|---|---|---|---|\n", x.seed);
        for c in &x.categories {
            let k = &c.score;
            s += &format!("| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} |\n", c.category, c.kind, k.queries, k.recall, k.precision, k.abstain_rate, k.false_accept_rate);
        }
        s += "\n| ablation | exact recall | precision |\n|---|---|---|\n";
        for (n, a) in &x.ablations {
            s += &format!("| {n} | {:.4} | {:.4} |\n", a.recall, a.precision);
        }
        s += "\n| W_FAST associations | recall | first recalled |\n|---|---|---|\n";
        for f in &x.fast_capacity {
            s += &format!("| {} | {:.3} | {} |\n", f.associations, f.recall, f.first_recalled);
        }
        let l = &x.latency;
        s += &format!("\nLatency: write p50/p95 {:.0}/{:.0} µs, query p50/p95 {:.0}/{:.0} µs at {} facts, save {:.1} ms, load {:.1} ms, checkpoint {} bytes.\n\n", l.write_p50_us, l.write_p95_us, l.query_p50_us, l.query_p95_us, l.active_facts_at_query, l.save_ms, l.load_ms, l.checkpoint_bytes);
        s += "| check | passed | detail |\n|---|---|---|\n";
        for c in &x.checks {
            s += &format!("| {} | {} | {} |\n", c.name, if c.passed { "yes" } else { "**no**" }, c.detail);
        }
    }
    if !r.failures.is_empty() {
        s += "\n### Failures\n\n";
        for f in &r.failures {
            s += &format!("- {f}\n");
        }
    }
    s += &format!("\n_{}_\n", r.scope_note);
    s
}

pub fn curves_csv(r: &Report) -> (String, String, String) {
    let mut inter = String::from("seed,curve,facts_stored,exact_recall,exact_precision,probe_paraphrase_recall,probe_paraphrase_precision,probe_mean_margin\n");
    let mut sweep = String::from("seed,coverage_threshold,exact_recall,exact_precision,paraphrase_recall,paraphrase_precision,negative_false_accept\n");
    let mut fast = String::from("seed,associations,recall,first_recalled\n");
    for x in &r.seeds {
        for (name, c) in [("interference", &x.interference), ("capacity", &x.capacity)] {
            for p in c {
                inter += &format!("{},{name},{},{:.5},{:.5},{:.5},{:.5},{:.5}\n", x.seed, p.facts_stored, p.exact.recall, p.exact.precision, p.probe_paraphrase.recall, p.probe_paraphrase.precision, p.probe_mean_margin);
            }
        }
        for p in &x.sweep {
            sweep += &format!("{},{},{:.5},{:.5},{:.5},{:.5},{:.5}\n", x.seed, p.threshold, p.exact.recall, p.exact.precision, p.paraphrase.recall, p.paraphrase.precision, p.negatives.false_accept_rate);
        }
        for p in &x.fast_capacity {
            fast += &format!("{},{},{:.5},{}\n", x.seed, p.associations, p.recall, p.first_recalled);
        }
    }
    (inter, sweep, fast)
}
