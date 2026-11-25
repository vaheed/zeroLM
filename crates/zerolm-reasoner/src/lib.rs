//! Relational memory of bound role-filler structures and controlled
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
mod lexer;
mod grammar;
mod grammar_seed;
mod parser_seed;
mod parser;
mod frame;
mod render;
pub mod lookup;
pub mod operators;
pub mod oracle;

use serde::{Deserialize, Serialize};
use std::path::Path;
use zerolm_engine::binding::Hv;
use zerolm_engine::semantic_state::SemanticState;
use crate::frame::{Frame, FORM_COUNT, FORM_FACT, FORM_HOLDS, FORM_RULE, FORM_WHICH};
use zerolm_memory::checkpoint::{self, CheckpointError, Tensor};
use zerolm_tensor::sha256::Sha256;
use zerolm_tensor::{fnv1a_label, Matrix};

pub const SLOTS: &str = "memory.relations.slots";
pub const META: &str = "memory.relations.meta";
pub const CODEBOOK: &str = "memory.relations.codebook";
pub const STATE: &str = "memory.relations.state";
pub const ROLES_T: &str = "core.reasoner.roles";
pub const RELATIONS_T: &str = "core.reasoner.relations";
pub const NUMBER_DIGITS_T: &str = "core.reasoner.number_digits";
pub const NUMBER_CODEBOOK_T: &str = "core.reasoner.number_codebook";
pub const ATOM_TABLE_T: &str = "core.reasoner.atom_table";

pub const ROLES: [&str; 5] = ["subj", "rel", "obj", "quant", "scope"];
pub const RELATIONS: [&str; 10] = ["isa", "can", "in", "prop", "gt", "before", "has", "more", "implies", "something"];

/// Slot metadata columns.
pub mod field {
    pub const ACTIVE: usize = 0;
    /// +1 asserted, -1 negated.
    pub const POLARITY: usize = 1;
    pub const CONFIDENCE: usize = 2;
    /// Predicate code (the template table's predicate code), 9 for rules.
    pub const KIND: usize = 3;
    pub const SEQ_LO: usize = 4;
    pub const SEQ_HI: usize = 5;
    pub const PROV: usize = 6; // 6, 7, 8
    pub const N_ROLES: usize = 9;
    pub const COUNT: usize = 10;
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
    numbers: circuits::Numbers,
    grammar: grammar_seed::Grammar,
    /// Compiled atom encoder (`atom-encode`) with its table.
    atoms: circuits::Atoms,
    roles: Vec<Hv>,
    rels: Vec<Hv>,
    /// Atoms of the seeded dimension names (input of `frame-bind`).
    dim_atoms: Matrix,
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
    /// Evaluation only: recorded circuit runs for reference replay.
    recorder: circuits::Recorder,
}

// Role positions (subject, relation, object, quantity, scope) follow `ROLES`; `frame-bind` fills them in that order.

pub(crate) fn rel_index(name: &str) -> usize {
    RELATIONS.iter().position(|r| *r == name).expect("known relation")
}

const P24: u64 = 1 << 24;

impl Reasoner {
    pub fn new(cfg: ReasonerConfig) -> Result<Self, ReasonerError> {
        cfg.validate().map_err(ReasonerError::Config)?;
        let atoms = circuits::Atoms::new(&cfg);
        let roles = ROLES.iter().map(|r| atoms.encode("role", r)).collect();
        let rels = RELATIONS.iter().map(|r| atoms.encode("rel", r)).collect();
        let grammar = grammar_seed::Grammar::new();
        let dim_atoms = Matrix { rows: grammar.dimensions.len(), cols: 2 * cfg.dim, data: grammar.dimensions.iter().flat_map(|d| atoms.encode("sym", d).data).collect() };
        Ok(Reasoner {
            dim_atoms,
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
            recorder: Default::default(),
            atoms,
            numbers: circuits::Numbers::new(&cfg),
            grammar,
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

    /// The vector of a name, computed by the compiled atom encoder.
    pub fn symbol(&self, name: &str) -> Hv {
        self.atoms.encode("sym", name)
    }

    /// The compiled number encoder and decoder.
    pub fn numbers(&self) -> &circuits::Numbers {
        &self.numbers
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

    // ---- Codebook ----

    pub(crate) fn code(&self, i: usize) -> Hv {
        Hv::from_data(self.cfg.dim, self.codebook.row(i))
    }

    /// Codebook index of a symbol vector (cosine >= symbol_same), by the
    /// `symbol-cleanup` circuit.
    pub(crate) fn find_symbol(&self, v: &Hv) -> Option<usize> {
        self.symbol_cleanup(v, circuits::CleanupMode::Same)
    }

    /// Index of a name's symbol if it was ever stored.
    pub fn known(&self, name: &str) -> Option<usize> {
        self.find_symbol(&self.symbol(name))
    }

    /// Intern a symbol vector: `symbol-cleanup` decides whether it is new (and
    /// `gate` is 1); the host appends the circuit's row at the codebook counter.
    fn intern(&mut self, v: &Hv, gate: f32) -> Result<(), ReasonerError> {
        let c = self.symbol_cleanup_full(v, circuits::CleanupMode::Same, gate);
        if c.added {
            if self.n_sym == self.cfg.symbols {
                return Err(ReasonerError::Full("codebook"));
            }
            let (i, w) = (self.n_sym, self.codebook.cols);
            self.codebook.data[i * w..(i + 1) * w].copy_from_slice(&c.row);
            self.n_sym += 1;
        }
        Ok(())
    }

    // ---- Language interface ----

    /// Resolve "it": the codebook symbol nearest the discourse state
    /// (`symbol-cleanup`, pronoun threshold). Returns (vector, resolved).
    /// An unresolved "it" is the zero vector, which no circuit matches.
    pub(crate) fn resolve_pronoun(&self) -> (Hv, f32) {
        match self.symbol_cleanup(&self.state.context, circuits::CleanupMode::Pronoun) {
            Some(i) => (self.code(i), 1.0),
            None => (Hv::zeros(self.cfg.dim), 0.0),
        }
    }

    /// Store a statement: `relation-probe` decides (resolved and not a
    /// duplicate) and computes the slot and metadata rows; the host appends
    /// them at the slot counter and logs the write.
    fn write(&mut self, parts: &[(usize, Hv)], polarity: f32, kind: f32, confidence: f32, gate: f32, stream: &str) -> Result<Observed, ReasonerError> {
        let seq = self.clock + 1;
        let h = fnv1a_label(stream);
        let m = (1u64 << 22) - 1;
        let mut record = vec![0.0f32; field::COUNT];
        record[field::ACTIVE] = 1.0;
        record[field::POLARITY] = polarity;
        record[field::CONFIDENCE] = confidence;
        record[field::KIND] = kind;
        record[field::SEQ_LO] = (seq % P24) as f32;
        record[field::SEQ_HI] = (seq / P24 % P24) as f32;
        record[field::PROV] = (h & m) as f32;
        record[field::PROV + 1] = ((h >> 22) & m) as f32;
        record[field::PROV + 2] = (h >> 44) as f32;
        record[field::N_ROLES] = parts.len() as f32;
        let st = self.store(parts, &record, gate);
        if self.state.enabled && kind != RULE_KIND {
            self.state.context = st.context.clone();
        }
        if !st.stored {
            return Ok(match st.dup_slot {
                Some(s) if gate == 1.0 => Observed::Duplicate { slot: s },
                _ => Observed::Unresolved,
            });
        }
        if self.n_slots == self.cfg.slots {
            return Err(ReasonerError::Full("relation slots"));
        }
        let s = self.n_slots;
        self.n_slots += 1;
        self.clock = seq;
        let c = self.slots.cols;
        self.slots.data[s * c..(s + 1) * c].copy_from_slice(&st.row);
        self.meta.data[s * field::COUNT..(s + 1) * field::COUNT].copy_from_slice(&st.meta_row);
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

    /// Store a bound statement (`frame-bind` chose the fillers and the codebook
    /// entries; `relation-probe` decides the write).
    fn teach(&mut self, f: &Frame, stream: &str) -> Result<Observed, ReasonerError> {
        let mut parts = f.parts.clone();
        let (mut gate, mut confidence) = (1.0, 1.0);
        if f.form == FORM_FACT && f.pronoun {
            let (s, resolved) = self.resolve_pronoun();
            parts[0].1 = s;
            (gate, confidence) = (resolved, self.cfg.pronoun_confidence);
        }
        for v in &f.interns {
            self.intern(v, gate)?;
        }
        self.write(&parts, f.polarity, f.kind, confidence, gate, stream)
    }

    /// Process one line: statements are stored, questions are answered.
    /// Recognition, binding, and every decision run in compiled programs; the
    /// form code they emit picks the fixed circuit sequence.
    pub fn observe(&mut self, text: &str, stream: &str) -> Result<Observed, ReasonerError> {
        let Some(f) = self.parse_frame(text) else { return Ok(Observed::Unparsed) };
        match f.form {
            FORM_FACT | FORM_RULE => self.teach(&f, stream),
            FORM_HOLDS | FORM_COUNT | FORM_WHICH => Ok(Observed::Answer(self.answer_frame(&f))),
            _ => Ok(Observed::Unparsed),
        }
    }

    /// Answer a question line (`None` if it is not a question in the language).
    pub fn ask(&self, text: &str) -> Option<Answer> {
        let f = self.parse_frame(text)?;
        matches!(f.form, FORM_HOLDS | FORM_COUNT | FORM_WHICH).then(|| self.answer_frame(&f))
    }

    // ---- Checkpoint ----

    pub fn tensors(&self) -> Vec<Tensor> {
        let d2 = 2 * self.cfg.dim;
        let rows = |v: &[Hv]| v.iter().flat_map(|h| h.data.clone()).collect::<Vec<f32>>();
        let fresh = Reasoner::new(self.cfg).expect("valid config");
        let mut t = vec![
            Tensor::new(ROLES_T, "binding_roles", ROLES.len(), d2, rows(&fresh.roles)),
            Tensor::new(RELATIONS_T, "binding_roles", RELATIONS.len(), d2, rows(&fresh.rels)),
            Tensor::new(NUMBER_DIGITS_T, "binding_roles", fresh.numbers.digits.rows, d2, fresh.numbers.digits.data.clone()),
            Tensor::new(NUMBER_CODEBOOK_T, "binding_roles", fresh.numbers.codebook.rows, d2, fresh.numbers.codebook.data.clone()),
            Tensor::new(ATOM_TABLE_T, "binding_roles", self.atoms.table.rows, d2, self.atoms.table.data.clone()),
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
        t.extend(fresh.grammar.tensors.iter().cloned());
        // Compiled circuit constants (W_PROG), frozen like the role phasors.
        for c in [&fresh.atoms.circuit, &fresh.numbers.encoder, &fresh.numbers.decoder, &fresh.grammar.lookup, &fresh.grammar.rewrite].into_iter().chain(fresh.grammar.parser.all()).chain(fresh.circuits.all()) {
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
                _ if w.group != "relation_memory" => {
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
