//! Phase 6: relational memory of bound role-filler structures and controlled
//! inference operators.
//!
//! Statements are parsed by `zerolm_engine::semantic_state` into structures,
//! encoded as bundles of role-filler bindings (`zerolm_engine::binding`), and
//! written as rows of the tensor `memory.relations.slots`. No sentence or
//! name string is stored: names exist only as their seeded vectors in
//! `memory.relations.codebook`.
//!
//! Questions are answered by the operators in [`operators`], which only use
//! vector probes (thresholded attention over slots), unbinding, cleanup
//! against the codebook, and binding (number arithmetic). The symbolic
//! [`oracle`] and the [`lookup`] baseline exist for evaluation only and are
//! never called by the reasoner.

pub mod circuits;
pub mod lookup;
pub mod operators;
pub mod oracle;

use serde::{Deserialize, Serialize};
use std::path::Path;
use zerolm_engine::binding::{bind, bundle, sim_raw, unbind, Hv, Space};
use zerolm_engine::semantic_state::{parse, Fact, Pred, Rule, SemanticState, Term, Utterance};
use zerolm_memory::checkpoint::{self, CheckpointError, Tensor};
use zerolm_tensor::sha256::Sha256;
use zerolm_tensor::{fnv1a, Matrix};

pub const SLOTS: &str = "memory.relations.slots";
pub const META: &str = "memory.relations.meta";
pub const CODEBOOK: &str = "memory.relations.codebook";
pub const STATE: &str = "memory.relations.state";
pub const ROLES_T: &str = "core.reasoner.roles";
pub const RELATIONS_T: &str = "core.reasoner.relations";
pub const NUMBER_T: &str = "core.reasoner.number_one";

pub const ROLES: [&str; 5] = ["subj", "rel", "obj", "quant", "scope"];
pub const RELATIONS: [&str; 10] = ["isa", "can", "in", "prop", "gt", "before", "has", "more", "implies", "something"];

/// Slot metadata columns.
pub mod field {
    pub const ACTIVE: usize = 0;
    /// +1 asserted, -1 negated.
    pub const POLARITY: usize = 1;
    pub const CONFIDENCE: usize = 2;
    /// Predicate code (see `pred_code`), 9 for rules.
    pub const KIND: usize = 3;
    pub const SEQ_LO: usize = 4;
    pub const SEQ_HI: usize = 5;
    pub const PROV: usize = 6; // 6, 7, 8
    pub const N_ROLES: usize = 9;
    pub const COUNT: usize = 10;
}

pub fn pred_code(p: &Pred) -> f32 {
    match p {
        Pred::Isa(_) => 1.0,
        Pred::Can(_) => 2.0,
        Pred::In(_) => 3.0,
        Pred::Prop(_) => 4.0,
        Pred::Greater { .. } => 5.0,
        Pred::Before(_) => 6.0,
        Pred::Count { .. } => 7.0,
        Pred::More { .. } => 8.0,
    }
}
pub const RULE_KIND: f32 = 9.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasonerConfig {
    pub seed: u64,
    /// Complex dimension of every vector.
    pub dim: usize,
    pub slots: usize,
    pub symbols: usize,
    /// A probe of k role-filler pairs matches a slot when sim >= k - margin.
    pub match_margin: f32,
    /// Minimum cosine for cleanup of an unbound filler to a codebook symbol.
    pub cleanup_min: f32,
    /// Cosine at which a new symbol vector is the same codebook entry.
    pub symbol_same: f32,
    /// Minimum cosine for resolving "it" from the discourse state.
    pub pronoun_min: f32,
    pub pronoun_decay: f32,
    /// Confidence of a fact whose subject was resolved from a pronoun.
    pub pronoun_confidence: f32,
    /// Maximum chain length of any operator.
    pub max_depth: usize,
    /// Numbers are decoded in [-number_range, number_range].
    pub number_range: i64,
}

impl Default for ReasonerConfig {
    fn default() -> Self {
        ReasonerConfig {
            seed: zerolm_engine::bootstrap::DEFAULT_SEED,
            dim: 512,
            slots: 4096,
            symbols: 2048,
            match_margin: 0.5,
            cleanup_min: 0.3,
            symbol_same: 0.9,
            pronoun_min: 0.4,
            pronoun_decay: 0.5,
            pronoun_confidence: 0.9,
            max_depth: 8,
            number_range: 1000,
        }
    }
}

impl ReasonerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.dim < 64 || self.slots == 0 || self.symbols == 0 || self.slots >= 1 << 24 {
            return Err("dim >= 64, slots in 1..2^24, symbols > 0".into());
        }
        if !(0.0..1.0).contains(&self.match_margin) || !(0.0..1.0).contains(&self.cleanup_min) || !(0.0..=1.0).contains(&self.symbol_same) {
            return Err("thresholds out of range".into());
        }
        if self.max_depth == 0 || self.number_range <= 0 {
            return Err("max_depth and number_range must be positive".into());
        }
        Ok(())
    }
}

/// Inference operators; each can be disabled for a causal ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Ops {
    pub class_inclusion: bool,
    pub transitivity: bool,
    pub attribute_inheritance: bool,
    pub comparison: bool,
    pub arithmetic: bool,
    pub temporal_ordering: bool,
    pub negation: bool,
    pub conditionals: bool,
}

impl Ops {
    pub const ALL: Ops = Ops {
        class_inclusion: true,
        transitivity: true,
        attribute_inheritance: true,
        comparison: true,
        arithmetic: true,
        temporal_ordering: true,
        negation: true,
        conditionals: true,
    };
    pub const NAMES: [&'static str; 8] =
        ["class_inclusion", "transitivity", "attribute_inheritance", "comparison", "arithmetic", "temporal_ordering", "negation", "conditionals"];

    pub fn without(name: &str) -> Ops {
        let mut o = Ops::ALL;
        match name {
            "class_inclusion" => o.class_inclusion = false,
            "transitivity" => o.transitivity = false,
            "attribute_inheritance" => o.attribute_inheritance = false,
            "comparison" => o.comparison = false,
            "arithmetic" => o.arithmetic = false,
            "temporal_ordering" => o.temporal_ordering = false,
            "negation" => o.negation = false,
            "conditionals" => o.conditionals = false,
            other => panic!("unknown operator {other}"),
        }
        o
    }
}

/// Pathway ablations (evaluation only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Pathway {
    /// Roles collapse to the identity: structures become plain superpositions.
    Binding,
    /// The discourse state stays zero, so "it" never resolves.
    RecurrentState,
    /// Relation slots read as zero: probes match nothing.
    Retrieval,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Verdict {
    Yes,
    No,
    Unknown,
    Contradiction,
    Number(i64),
    Choice(String),
}

impl Verdict {
    pub fn text(&self) -> String {
        match self {
            Verdict::Yes => "yes".into(),
            Verdict::No => "no".into(),
            Verdict::Unknown => "unknown".into(),
            Verdict::Contradiction => "contradiction".into(),
            Verdict::Number(n) => n.to_string(),
            Verdict::Choice(s) => s.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    pub verdict: Verdict,
    /// True when the answer needed an operator beyond one direct retrieval.
    pub derived: bool,
    /// Slots of the premises the answer rests on (sorted, unique).
    pub premises: Vec<usize>,
    /// Product of the premises' confidences (1 for an unknown answer).
    pub confidence: f32,
    pub operators: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Observed {
    Stored { slot: usize },
    Duplicate { slot: usize },
    /// "it" could not be resolved from the discourse state; nothing stored.
    Unresolved,
    /// Not in the bounded language; nothing stored.
    Unparsed,
    Answer(Answer),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReasonerError {
    Full(&'static str),
    Config(String),
    Format(String),
    CoreMismatch(String),
    Checkpoint(CheckpointError),
}

impl std::fmt::Display for ReasonerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReasonerError::Full(what) => write!(f, "{what} capacity exhausted"),
            ReasonerError::Config(e) => write!(f, "config: {e}"),
            ReasonerError::Format(e) => write!(f, "checkpoint content: {e}"),
            ReasonerError::CoreMismatch(t) => write!(f, "frozen tensor {t} differs from its seeded initialization"),
            ReasonerError::Checkpoint(e) => write!(f, "checkpoint: {e}"),
        }
    }
}

impl From<CheckpointError> for ReasonerError {
    fn from(e: CheckpointError) -> Self {
        ReasonerError::Checkpoint(e)
    }
}

/// One logged memory write (fields required by `update-policy.json`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WriteEvent {
    pub seq: u64,
    pub input_stream: String,
    pub signal: &'static str,
    pub rule: &'static str,
    pub changed_tensors: Vec<&'static str>,
    pub slot: usize,
    pub rows_sha256: String,
    pub checkpoint_sha256: Option<String>,
}

pub struct Reasoner {
    pub cfg: ReasonerConfig,
    pub ops: Ops,
    space: Space,
    roles: Vec<Hv>,
    rels: Vec<Hv>,
    slots: Matrix,
    meta: Matrix,
    codebook: Matrix,
    n_slots: usize,
    n_sym: usize,
    clock: u64,
    state: SemanticState,
    binding_ablated: bool,
    retrieval_ablated: bool,
    events: Vec<WriteEvent>,
    unsaved_from: usize,
    /// Compiled relational circuits (`W_PROG`).
    circuits: circuits::Circuits,
    /// C5 output for the current memory state (session state, recomputed after writes).
    graph_cache: std::cell::RefCell<Option<circuits::Graph>>,
}

pub(crate) const SUBJ: usize = 0;
pub(crate) const REL: usize = 1;
pub(crate) const OBJ: usize = 2;
pub(crate) const QUANT: usize = 3;
pub(crate) const SCOPE: usize = 4;

pub(crate) fn rel_index(name: &str) -> usize {
    RELATIONS.iter().position(|r| *r == name).expect("known relation")
}

const P24: u64 = 1 << 24;

impl Reasoner {
    pub fn new(cfg: ReasonerConfig) -> Result<Self, ReasonerError> {
        cfg.validate().map_err(ReasonerError::Config)?;
        let space = Space::new(cfg.seed, cfg.dim);
        let roles = ROLES.iter().map(|r| space.role(r)).collect();
        let rels = RELATIONS.iter().map(|r| space.atom("rel", r)).collect();
        Ok(Reasoner {
            ops: Ops::ALL,
            roles,
            rels,
            slots: Matrix::zeros(cfg.slots, 2 * cfg.dim),
            meta: Matrix::zeros(cfg.slots, field::COUNT),
            codebook: Matrix::zeros(cfg.symbols, 2 * cfg.dim),
            n_slots: 0,
            n_sym: 0,
            clock: 0,
            state: SemanticState::new(cfg.dim, cfg.pronoun_decay),
            binding_ablated: false,
            retrieval_ablated: false,
            events: Vec::new(),
            unsaved_from: 0,
            circuits: circuits::Circuits::new(&cfg),
            graph_cache: std::cell::RefCell::new(None),
            space,
            cfg,
        })
    }

    pub fn ablate(&mut self, p: Pathway) {
        match p {
            Pathway::Binding => self.binding_ablated = true,
            Pathway::RecurrentState => {
                self.state.enabled = false;
                self.state.reset();
            }
            Pathway::Retrieval => self.retrieval_ablated = true,
        }
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    pub fn events(&self) -> &[WriteEvent] {
        &self.events
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn n_symbols(&self) -> usize {
        self.n_sym
    }

    pub fn slot_meta(&self, slot: usize, f: usize) -> f32 {
        self.meta.data[slot * field::COUNT + f]
    }

    pub(crate) fn rel(&self, name: &str) -> &Hv {
        &self.rels[rel_index(name)]
    }

    pub(crate) fn role(&self, r: usize) -> Hv {
        if self.binding_ablated {
            // Identity phasor: binding with it is a no-op.
            Hv::from_angles(&vec![0.0; self.cfg.dim])
        } else {
            self.roles[r].clone()
        }
    }

    pub(crate) fn encode(&self, parts: &[(usize, Hv)]) -> Hv {
        let items: Vec<Hv> = parts.iter().map(|(r, f)| bind(&self.role(*r), f)).collect();
        bundle(&items)
    }

    // ---- Codebook ----

    pub(crate) fn code(&self, i: usize) -> Hv {
        Hv::from_data(self.cfg.dim, self.codebook.row(i))
    }

    /// Codebook index of a symbol vector (cosine >= symbol_same).
    pub(crate) fn find_symbol(&self, v: &Hv) -> Option<usize> {
        self.cleanup_with(v, self.cfg.symbol_same)
    }

    /// Nearest codebook entry with cosine >= min.
    pub(crate) fn cleanup_with(&self, v: &Hv, min: f32) -> Option<usize> {
        let d = self.cfg.dim;
        let vn = sim_raw(&v.data, &v.data, d).sqrt();
        if vn == 0.0 {
            return None;
        }
        let mut best: Option<(usize, f32)> = None;
        for i in 0..self.n_sym {
            let row = self.codebook.row(i);
            let c = sim_raw(row, &v.data, d) / vn; // rows are unit phasors: |row| = 1
            if c >= min && best.map(|b| c > b.1).unwrap_or(true) {
                best = Some((i, c));
            }
        }
        best.map(|b| b.0)
    }

    pub(crate) fn cleanup(&self, v: &Hv) -> Option<usize> {
        self.cleanup_with(v, self.cfg.cleanup_min)
    }

    /// Index of a name's symbol if it was ever stored.
    pub fn known(&self, name: &str) -> Option<usize> {
        self.find_symbol(&self.space.symbol(name))
    }

    fn intern(&mut self, name: &str) -> Result<usize, ReasonerError> {
        let v = self.space.symbol(name);
        if let Some(i) = self.find_symbol(&v) {
            return Ok(i);
        }
        if self.n_sym == self.cfg.symbols {
            return Err(ReasonerError::Full("codebook"));
        }
        let i = self.n_sym;
        let c = self.codebook.cols;
        self.codebook.data[i * c..(i + 1) * c].copy_from_slice(&v.data);
        self.n_sym += 1;
        Ok(i)
    }

    // ---- Probes ----

    /// Slots matching every given role-filler pair (sim >= k - margin).
    pub(crate) fn matches(&self, parts: &[(usize, Hv)]) -> Vec<usize> {
        if self.retrieval_ablated {
            return Vec::new();
        }
        let q = self.encode(parts);
        let need = parts.len() as f32 - self.cfg.match_margin;
        (0..self.n_slots)
            .filter(|&s| self.slot_meta(s, field::ACTIVE) == 1.0)
            .filter(|&s| sim_raw(self.slots.row(s), &q.data, self.cfg.dim) >= need)
            .collect()
    }

    pub(crate) fn unbind_slot(&self, slot: usize, role: usize) -> Hv {
        unbind(&Hv::from_data(self.cfg.dim, self.slots.row(slot)), &self.role(role))
    }

    // ---- Language interface ----

    /// Relation vector and role-filler parts of a predicate for subject `s`.
    pub(crate) fn pred_parts(&self, s: &Hv, pred: &Pred) -> Vec<(usize, Hv)> {
        let sym = |n: &str| self.space.symbol(n);
        let mut p = vec![(SUBJ, s.clone())];
        match pred {
            Pred::Isa(c) => p.extend([(REL, self.rel("isa").clone()), (OBJ, sym(c))]),
            Pred::Can(v) => p.extend([(REL, self.rel("can").clone()), (OBJ, sym(v))]),
            Pred::In(y) => p.extend([(REL, self.rel("in").clone()), (OBJ, sym(y))]),
            Pred::Prop(x) => p.extend([(REL, self.rel("prop").clone()), (OBJ, sym(x))]),
            Pred::Greater { dim, other } => p.extend([(REL, bind(self.rel("gt"), &sym(dim))), (OBJ, sym(other))]),
            Pred::Before(y) => p.extend([(REL, self.rel("before").clone()), (OBJ, sym(y))]),
            Pred::Count { thing, n } => p.extend([(REL, bind(self.rel("has"), &sym(thing))), (QUANT, self.space.number(*n))]),
            Pred::More { thing, delta, other } => {
                p.extend([(REL, bind(self.rel("more"), &sym(thing))), (OBJ, sym(other)), (QUANT, self.space.number(*delta))])
            }
        }
        p
    }

    fn pred_symbols(pred: &Pred) -> Vec<&str> {
        match pred {
            Pred::Isa(x) | Pred::Can(x) | Pred::In(x) | Pred::Prop(x) | Pred::Before(x) => vec![x],
            Pred::Greater { dim, other } => vec![dim, other],
            Pred::Count { thing, .. } => vec![thing],
            Pred::More { thing, other, .. } => vec![thing, other],
        }
    }

    /// Resolve a subject term to a codebook vector; `None` if "it" is unresolved.
    pub(crate) fn resolve(&self, t: &Term) -> Option<(Hv, bool)> {
        match t {
            Term::Name(n) => Some((self.space.symbol(n), false)),
            Term::Pronoun => {
                let cands: Vec<Hv> = (0..self.n_sym).map(|i| self.code(i)).collect();
                self.state.resolve(cands.iter().enumerate(), self.cfg.pronoun_min).map(|(i, _)| (cands[i].clone(), true))
            }
        }
    }

    fn write(&mut self, parts: &[(usize, Hv)], polarity: f32, kind: f32, confidence: f32, stream: &str) -> Result<Observed, ReasonerError> {
        let v = self.encode(parts);
        let full = parts.len() as f32 - self.cfg.match_margin;
        for s in 0..self.n_slots {
            if self.slot_meta(s, field::POLARITY) == polarity
                && self.slot_meta(s, field::N_ROLES) == parts.len() as f32
                && sim_raw(self.slots.row(s), &v.data, self.cfg.dim) >= full
            {
                return Ok(Observed::Duplicate { slot: s });
            }
        }
        if self.n_slots == self.cfg.slots {
            return Err(ReasonerError::Full("relation slots"));
        }
        let s = self.n_slots;
        self.n_slots += 1;
        self.clock += 1;
        let c = self.slots.cols;
        self.slots.data[s * c..(s + 1) * c].copy_from_slice(&v.data);
        let h = fnv1a(stream.as_bytes());
        let m = (1u64 << 22) - 1;
        let row = &mut self.meta.data[s * field::COUNT..(s + 1) * field::COUNT];
        row[field::ACTIVE] = 1.0;
        row[field::POLARITY] = polarity;
        row[field::CONFIDENCE] = confidence;
        row[field::KIND] = kind;
        row[field::SEQ_LO] = (self.clock % P24) as f32;
        row[field::SEQ_HI] = (self.clock / P24 % P24) as f32;
        row[field::PROV] = (h & m) as f32;
        row[field::PROV + 1] = ((h >> 22) & m) as f32;
        row[field::PROV + 2] = (h >> 44) as f32;
        row[field::N_ROLES] = parts.len() as f32;
        let mut sh = Sha256::new();
        sh.update_f32(self.slots.row(s));
        sh.update_f32(self.meta.row(s));
        self.events.push(WriteEvent {
            seq: self.clock,
            input_stream: stream.into(),
            signal: "user_statement",
            rule: "relation_write",
            changed_tensors: vec![SLOTS, META, CODEBOOK],
            slot: s,
            rows_sha256: sh.finish_hex(),
            checkpoint_sha256: None,
        });
        Ok(Observed::Stored { slot: s })
    }

    pub fn teach_fact(&mut self, f: &Fact, stream: &str) -> Result<Observed, ReasonerError> {
        let Some((s, via_pronoun)) = self.resolve(&f.subj) else { return Ok(Observed::Unresolved) };
        if let Term::Name(n) = &f.subj {
            self.intern(n)?;
        }
        for n in Self::pred_symbols(&f.pred) {
            self.intern(n)?;
        }
        let parts = self.pred_parts(&s, &f.pred);
        let conf = if via_pronoun { self.cfg.pronoun_confidence } else { 1.0 };
        let out = self.write(&parts, if f.negated { -1.0 } else { 1.0 }, pred_code(&f.pred), conf, stream)?;
        self.state.observe_subject(&s);
        Ok(out)
    }

    pub fn teach_rule(&mut self, r: &Rule, stream: &str) -> Result<Observed, ReasonerError> {
        for n in [&r.premise, &r.conclusion].into_iter().chain(r.scope.as_ref()) {
            self.intern(n)?;
        }
        let scope = match &r.scope {
            Some(x) => self.space.symbol(x),
            None => self.rel("something").clone(),
        };
        let parts = vec![
            (SUBJ, self.space.symbol(&r.premise)),
            (REL, self.rel("implies").clone()),
            (OBJ, self.space.symbol(&r.conclusion)),
            (SCOPE, scope),
        ];
        self.write(&parts, 1.0, RULE_KIND, 1.0, stream)
    }

    /// Process one line: statements are stored, questions are answered.
    pub fn observe(&mut self, text: &str, stream: &str) -> Result<Observed, ReasonerError> {
        match parse(text) {
            None => Ok(Observed::Unparsed),
            Some(Utterance::Fact(f)) => self.teach_fact(&f, stream),
            Some(Utterance::Rule(r)) => self.teach_rule(&r, stream),
            Some(Utterance::Query(q)) => Ok(Observed::Answer(self.answer(&q))),
        }
    }

    /// Answer a question line (`None` if it is not a question in the language).
    pub fn ask(&self, text: &str) -> Option<Answer> {
        match parse(text) {
            Some(Utterance::Query(q)) => Some(self.answer(&q)),
            _ => None,
        }
    }

    // ---- Checkpoint ----

    pub fn tensors(&self) -> Vec<Tensor> {
        let d2 = 2 * self.cfg.dim;
        let rows = |v: &[Hv]| v.iter().flat_map(|h| h.data.clone()).collect::<Vec<f32>>();
        let fresh = Reasoner::new(self.cfg).expect("valid config");
        let mut t = vec![
            Tensor::new(ROLES_T, "binding_roles", ROLES.len(), d2, rows(&fresh.roles)),
            Tensor::new(RELATIONS_T, "binding_roles", RELATIONS.len(), d2, rows(&fresh.rels)),
            Tensor::new(NUMBER_T, "binding_roles", 1, d2, self.space.number(1).data),
            Tensor::new(SLOTS, "relation_memory", self.slots.rows, d2, self.slots.data.clone()),
            Tensor::new(META, "relation_memory", self.meta.rows, field::COUNT, self.meta.data.clone()),
            Tensor::new(CODEBOOK, "relation_memory", self.codebook.rows, d2, self.codebook.data.clone()),
            Tensor::new(
                STATE,
                "relation_memory",
                1,
                4,
                vec![(self.clock % P24) as f32, (self.clock / P24 % P24) as f32, self.n_slots as f32, self.n_sym as f32],
            ),
        ];
        // Compiled circuit constants (W_PROG), frozen like the role phasors.
        for c in [&fresh.circuits.graph, &fresh.circuits.reach] {
            for p in &c.tensors {
                t.push(Tensor::new(&c.qualified(&p.name), zerolm_wsp::compiler::GROUP, p.rows, p.cols, p.data.clone()));
            }
        }
        t
    }

    pub fn save(&mut self, dir: &Path) -> Result<String, ReasonerError> {
        let bytes = checkpoint::encode(&serde_json::json!({ "config": self.cfg }), &self.tensors());
        let sha = checkpoint::save_atomic(dir, &bytes)?;
        for e in &mut self.events[self.unsaved_from..] {
            e.checkpoint_sha256 = Some(sha.clone());
        }
        self.unsaved_from = self.events.len();
        Ok(sha)
    }

    pub fn load(dir: &Path) -> Result<Self, ReasonerError> {
        let l = checkpoint::load_latest(dir)?;
        let cfg: ReasonerConfig =
            serde_json::from_value(l.meta.get("config").cloned().unwrap_or_default()).map_err(|e| ReasonerError::Format(e.to_string()))?;
        let mut r = Reasoner::new(cfg)?;
        let want = r.tensors();
        if l.tensors.len() != want.len() {
            return Err(ReasonerError::Format("tensor count".into()));
        }
        for (got, w) in l.tensors.into_iter().zip(want) {
            if (got.name.as_str(), got.rows, got.cols) != (w.name.as_str(), w.rows, w.cols) {
                return Err(ReasonerError::Format(format!("unexpected tensor {}", got.name)));
            }
            match got.name.as_str() {
                name if name == ROLES_T || name == RELATIONS_T || name == NUMBER_T || name.starts_with("prog.") => {
                    if got.data.iter().zip(&w.data).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        return Err(ReasonerError::CoreMismatch(got.name));
                    }
                }
                SLOTS => r.slots.data = got.data,
                META => r.meta.data = got.data,
                CODEBOOK => r.codebook.data = got.data,
                STATE => {
                    r.clock = got.data[0] as u64 + got.data[1] as u64 * P24;
                    r.n_slots = got.data[2] as usize;
                    r.n_sym = got.data[3] as usize;
                }
                other => return Err(ReasonerError::Format(format!("unexpected tensor {other}"))),
            }
        }
        if r.n_slots > cfg.slots || r.n_sym > cfg.symbols {
            return Err(ReasonerError::Format("counts exceed capacity".into()));
        }
        Ok(r)
    }
}
