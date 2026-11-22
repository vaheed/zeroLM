//! The compiled relational circuits (roadmap C5, C6) bound to this
//! reasoner's tensors. The host binds the used prefix of the relation slots
//! and the codebook, runs the programs, and reads their outputs.
//!
//! - `relation-graph` (C5) decodes the memory once per memory state; the
//!   result is cached session state, recomputed after any write.
//! - `relation-reach` (C6) replaces breadth-first search: reachability and
//!   shortest-proof premises as a fixed-step recurrence.

use crate::{field, Reasoner, ReasonerConfig, ROLES};
use std::collections::BTreeMap;
use zerolm_engine::binding::{Hv, Space};
use zerolm_engine::isa::{self, bundle, Bindings, Buf, Outputs, View};
use zerolm_wsp::circuits::atom_encode::Params as AtomParams;
use zerolm_wsp::circuits::decide::CountParams;
use zerolm_wsp::circuits::probe::{CleanupParams, ProbeParams};
use zerolm_wsp::circuits::number::{DecodeParams as NumberDecodeParams, EncodeParams as NumberEncodeParams};
use zerolm_wsp::circuits::relational::{Params, OPERATORS};
use zerolm_wsp::{compile, Circuit, Compiled, Spec};

pub const GRAPH_ID: &str = "relation-graph";
pub const REACH_ID: &str = "relation-reach";

fn params(cfg: &ReasonerConfig) -> Params {
    Params {
        dim: cfg.dim,
        meta_cols: field::COUNT,
        max_depth: cfg.max_depth,
        match_margin: cfg.match_margin,
        cleanup_min: cfg.cleanup_min,
        symbol_same: cfg.symbol_same,
    }
}

pub fn graph_spec(cfg: &ReasonerConfig) -> Spec {
    Spec { id: GRAPH_ID.into(), version: 1, seed_entry: "relation-graph-circuit".into(), description: String::new(), circuit: Circuit::RelationGraph(params(cfg)) }
}

pub fn reach_spec(cfg: &ReasonerConfig) -> Spec {
    Spec { id: REACH_ID.into(), version: 1, seed_entry: "relation-reach-circuit".into(), description: String::new(), circuit: Circuit::RelationReach(params(cfg)) }
}

pub const JOIN_ID: &str = "relation-join";
pub const LEVELS_ID: &str = "relation-levels";
pub const RULES_ID: &str = "relation-rules";
pub const COUNTS_ID: &str = "relation-counts";
pub const VERDICT_ID: &str = "relation-verdict";

/// The spec of one decision circuit (`zerolm_wsp::circuits::decide`).
pub fn decide_spec(cfg: &ReasonerConfig, id: &str) -> Spec {
    let p = params(cfg);
    let circuit = match id {
        JOIN_ID => Circuit::RelationJoin(p),
        LEVELS_ID => Circuit::RelationLevels(p),
        RULES_ID => Circuit::RelationRules(p),
        COUNTS_ID => Circuit::RelationCounts(CountParams { relational: p, range: cfg.number_range as u32 }),
        VERDICT_ID => Circuit::RelationVerdict(p),
        other => panic!("no decision circuit {other}"),
    };
    Spec { id: id.into(), version: 1, seed_entry: "relation-decide-circuits".into(), description: String::new(), circuit }
}

pub const ATOM_ID: &str = "atom-encode";

/// Atom encoder width: names up to `ATOM_BYTES - 1` bytes (longer names are cut).
pub const ATOM_BYTES: usize = 128;
/// Rows of `core.reasoner.atom_table`.
pub const ATOM_BUCKETS: usize = 1024;

pub fn atom_spec(cfg: &ReasonerConfig) -> Spec {
    Spec {
        id: ATOM_ID.into(),
        version: 1,
        seed_entry: "atom-encode-circuit".into(),
        description: String::new(),
        circuit: Circuit::AtomEncode(AtomParams { name_bytes: ATOM_BYTES, buckets: ATOM_BUCKETS, dim: cfg.dim }),
    }
}

/// The seeded random hash table of the atom encoder (`core.reasoner.atom_table`).
pub fn atom_table(cfg: &ReasonerConfig) -> zerolm_tensor::Matrix {
    let mut rng = zerolm_tensor::Rng::from_tag(cfg.seed, "binding.atom_table");
    let mut m = zerolm_tensor::Matrix::zeros(ATOM_BUCKETS, 2 * cfg.dim);
    m.data.iter_mut().for_each(|x| *x = rng.uniform(1.0));
    m
}

/// The compiled atom encoder with its table and a cache of its outputs.
pub struct Atoms {
    pub circuit: Compiled,
    pub table: zerolm_tensor::Matrix,
    cache: std::cell::RefCell<std::collections::HashMap<String, Hv>>,
    recorder: Recorder,
    dim: usize,
}

impl Atoms {
    pub fn new(cfg: &ReasonerConfig) -> Self {
        Atoms {
            circuit: compile(&atom_spec(cfg)).expect("valid reasoning config compiles"),
            table: atom_table(cfg),
            cache: Default::default(),
            recorder: Default::default(),
            dim: cfg.dim,
        }
    }

    /// The unit phasor of `namespace:name`, computed by the circuit
    /// (memoized: the cache holds circuit outputs, keyed by the same bytes).
    pub fn encode(&self, namespace: &str, name: &str) -> Hv {
        let key = format!("{namespace}:{name}");
        if let Some(h) = self.cache.borrow().get(&key) {
            return h.clone();
        }
        let mut row = vec![0.0f32; ATOM_BYTES];
        for (slot, b) in row[1..].iter_mut().zip(key.bytes()) {
            *slot = b as f32;
        }
        let w = with_consts(&self.circuit, vec![("table", view(&self.table, self.table.rows))]);
        let out = run(&self.circuit, vec![("name", Buf::row(row))], &w, &self.recorder);
        let h = Hv::from_data(self.dim, &out.get("atom").data);
        self.cache.borrow_mut().insert(key, h.clone());
        h
    }
}

pub const NUMBER_ENCODE_ID: &str = "number-encode";
pub const NUMBER_DECODE_ID: &str = "number-decode";
/// Encoder width: a sign and up to 7 digits (the parser accepts 0..=1,000,000).
pub const NUMBER_BYTES: usize = 8;
/// Decimal places of `core.reasoner.number_digits`.
pub const NUMBER_PLACES: usize = 7;

pub fn number_encode_spec(cfg: &ReasonerConfig) -> Spec {
    Spec {
        id: NUMBER_ENCODE_ID.into(),
        version: 1,
        seed_entry: "number-encode-circuit".into(),
        description: String::new(),
        circuit: Circuit::NumberEncode(NumberEncodeParams { numeral_bytes: NUMBER_BYTES, places: NUMBER_PLACES, dim: cfg.dim }),
    }
}

pub fn number_decode_spec(cfg: &ReasonerConfig) -> Spec {
    Spec {
        id: NUMBER_DECODE_ID.into(),
        version: 1,
        seed_entry: "number-decode-circuit".into(),
        description: String::new(),
        circuit: Circuit::NumberDecode(NumberDecodeParams { range: cfg.number_range as u32, dim: cfg.dim }),
    }
}

/// A number decoded by `number-decode`: its value and the clean codebook row.
#[derive(Debug, Clone)]
pub struct Decoded {
    pub value: i64,
    pub clean: Hv,
}

/// The compiled number encoder and decoder with their seeded tables
/// (`core.reasoner.number_digits`, `core.reasoner.number_codebook`).
pub struct Numbers {
    pub encoder: Compiled,
    pub decoder: Compiled,
    pub digits: zerolm_tensor::Matrix,
    pub codebook: zerolm_tensor::Matrix,
    cache: std::cell::RefCell<std::collections::HashMap<i64, Hv>>,
    recorder: Recorder,
    dim: usize,
}

impl Numbers {
    pub fn new(cfg: &ReasonerConfig) -> Self {
        let space = Space::new(cfg.seed, cfg.dim);
        let d2 = 2 * cfg.dim;
        let rows = (2 * cfg.number_range + 1) as usize;
        Numbers {
            encoder: compile(&number_encode_spec(cfg)).expect("valid reasoning config compiles"),
            decoder: compile(&number_decode_spec(cfg)).expect("valid reasoning config compiles"),
            digits: zerolm_tensor::Matrix { rows: 10 * NUMBER_PLACES, cols: d2, data: space.digit_table(NUMBER_PLACES) },
            codebook: zerolm_tensor::Matrix { rows, cols: d2, data: space.codebook(cfg.number_range) },
            cache: Default::default(),
            recorder: Default::default(),
            dim: cfg.dim,
        }
    }

    fn run(&self, c: &Compiled, input: (&str, Buf), table: (&str, View<'_>)) -> Outputs {
        run(c, vec![input], &with_consts(c, vec![table]), &self.recorder)
    }

    /// The phasor of an integer from its decimal numeral bytes, computed by
    /// `number-encode` (memoized by the integer).
    pub fn encode(&self, n: i64) -> Hv {
        if let Some(h) = self.cache.borrow().get(&n) {
            return h.clone();
        }
        // The parser hands over an integer; its canonical decimal numeral is
        // the circuit input (bytes copied, none inspected).
        let numeral = n.to_string();
        assert!(numeral.len() <= NUMBER_BYTES, "numeral wider than the number-encode input");
        let mut row = vec![0.0f32; NUMBER_BYTES];
        for (slot, b) in row.iter_mut().zip(numeral.bytes()) {
            *slot = b as f32;
        }
        let out = self.run(&self.encoder, ("numeral", Buf::row(row)), ("digits", self.digits_view()));
        let h = Hv::from_data(self.dim, &out.get("number").data);
        self.cache.borrow_mut().insert(n, h.clone());
        h
    }

    /// Clean up a noisy number vector with `number-decode`.
    pub fn decode(&self, v: &Hv) -> Decoded {
        let out = self.run(&self.decoder, ("v", Buf::row(v.data.clone())), ("numbers", view(&self.codebook, self.codebook.rows)));
        Decoded { value: out.get("value").data[0] as i64, clean: Hv::from_data(self.dim, &out.get("clean").data) }
    }

    fn digits_view(&self) -> View<'_> {
        view(&self.digits, self.digits.rows)
    }
}

pub struct Circuits {
    pub graph: Compiled,
    pub reach: Compiled,
    pub probe: Compiled,
    pub cleanup: Compiled,
    pub join: Compiled,
    pub levels: Compiled,
    pub rules: Compiled,
    pub counts: Compiled,
    pub verdict: Compiled,
    pub lexer: Compiled,
}

impl Circuits {
    pub fn new(cfg: &ReasonerConfig) -> Self {
        Circuits {
            lexer: zerolm_wsp::shipped("language-lex").expect("shipped lexer compiles"),
            graph: compile(&graph_spec(cfg)).expect("valid reasoning config compiles"),
            reach: compile(&reach_spec(cfg)).expect("valid reasoning config compiles"),
            probe: compile(&probe_spec(cfg)).expect("valid reasoning config compiles"),
            cleanup: compile(&cleanup_spec(cfg)).expect("valid reasoning config compiles"),
            join: compile(&decide_spec(cfg, JOIN_ID)).expect("valid reasoning config compiles"),
            levels: compile(&decide_spec(cfg, LEVELS_ID)).expect("valid reasoning config compiles"),
            rules: compile(&decide_spec(cfg, RULES_ID)).expect("valid reasoning config compiles"),
            counts: compile(&decide_spec(cfg, COUNTS_ID)).expect("valid reasoning config compiles"),
            verdict: compile(&decide_spec(cfg, VERDICT_ID)).expect("valid reasoning config compiles"),
        }
    }

    /// Every compiled program, in checkpoint order.
    pub fn all(&self) -> [&Compiled; 10] {
        [&self.graph, &self.reach, &self.probe, &self.cleanup, &self.join, &self.levels, &self.rules, &self.counts, &self.verdict, &self.lexer]
    }
}

/// C5 output for one memory state.
#[derive(Debug, Clone)]
pub struct Graph {
    pub key: (usize, usize, u64, bool, bool),
    pub subj_sim: Buf,
    pub obj_t: Buf,
    pub pos: Buf,
    pub neg: Buf,
}

fn view(m: &zerolm_tensor::Matrix, rows: usize) -> View<'_> {
    View::new(rows, m.cols, &m.data[..rows * m.cols])
}

/// Records circuit runs for reference replay (evaluation only).
#[derive(Default)]
pub struct Recorder {
    runs: std::cell::RefCell<Option<Vec<Recorded>>>,
    /// The last bundle of each program, reused while its weights are unchanged.
    last: std::cell::RefCell<BTreeMap<String, std::sync::Arc<Vec<u8>>>>,
}

impl Recorder {
    pub fn set(&self, on: bool) {
        *self.runs.borrow_mut() = if on { Some(Vec::new()) } else { None };
        self.last.borrow_mut().clear();
    }

    pub fn take(&self) -> Vec<Recorded> {
        self.runs.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default()
    }
}

/// The caller's external tensors followed by the program's compiled constants.
pub(crate) fn with_consts<'a>(c: &'a Compiled, mut ext: Vec<(&'a str, View<'a>)>) -> Vec<(&'a str, View<'a>)> {
    ext.extend(c.tensors.iter().map(|t| (t.name.as_str(), t.view())));
    ext
}

/// Bind, run, and (when recording) export one program run.
pub(crate) fn run(c: &Compiled, inputs: Vec<(&str, Buf)>, weights: &[(&str, View<'_>)], rec: &Recorder) -> Outputs {
    let mut b = Bindings::new();
    for (n, t) in &inputs {
        b = b.input(n, t.clone());
    }
    for (n, v) in weights {
        b = b.weight(n, *v);
    }
    let out = isa::run(&c.program, b).unwrap_or_else(|e| panic!("{} runs on its own memory: {e}", c.id));
    if let Some(runs) = rec.runs.borrow_mut().as_mut() {
        let encoded = bundle::encode(&c.program, weights);
        let mut last = rec.last.borrow_mut();
        let shared = match last.get(&c.id) {
            Some(p) if **p == encoded => p.clone(),
            _ => {
                let p = std::sync::Arc::new(encoded);
                last.insert(c.id.clone(), p.clone());
                p
            }
        };
        runs.push(Recorded { program: c.id.clone(), bundle: shared, inputs: inputs.into_iter().map(|(n, t)| (n.to_string(), t)).collect(), outputs: out.clone() });
    }
    out
}

pub const PROBE_ID: &str = "relation-probe";
pub const CLEANUP_ID: &str = "symbol-cleanup";

pub fn probe_spec(cfg: &ReasonerConfig) -> Spec {
    Spec {
        id: PROBE_ID.into(),
        version: 1,
        seed_entry: "relation-probe-circuits".into(),
        description: String::new(),
        circuit: Circuit::RelationProbe(ProbeParams { dim: cfg.dim, meta_cols: field::COUNT, n_roles_col: field::N_ROLES, match_margin: cfg.match_margin, discourse_decay: cfg.pronoun_decay }),
    }
}

pub fn cleanup_spec(cfg: &ReasonerConfig) -> Spec {
    Spec {
        id: CLEANUP_ID.into(),
        version: 1,
        seed_entry: "relation-probe-circuits".into(),
        description: String::new(),
        circuit: Circuit::SymbolCleanup(CleanupParams { dim: cfg.dim, cleanup_min: cfg.cleanup_min, symbol_same: cfg.symbol_same, pronoun_min: cfg.pronoun_min }),
    }
}

/// Outputs of one `relation-probe` (statement write) run.
#[derive(Debug, Clone)]
pub struct Store {
    /// The circuit decided to store the statement (resolved and not a duplicate).
    pub stored: bool,
    /// The slot row and metadata row to append (zero when nothing is stored).
    pub row: Vec<f32>,
    pub meta_row: Vec<f32>,
    /// The lowest duplicate slot, if the statement is already stored.
    pub dup_slot: Option<usize>,
    /// The discourse state after the statement.
    pub context: Hv,
}

/// Outputs of one `symbol-cleanup` run.
#[derive(Debug, Clone)]
pub struct Cleaned {
    pub found: Option<usize>,
    /// The circuit decided to add `row` to the codebook (a new symbol).
    pub added: bool,
    pub row: Vec<f32>,
}

/// Which threshold `symbol-cleanup` applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupMode {
    Same,
    Pronoun,
}

impl Reasoner {
    /// Run `relation-probe` for a statement's role-filler `parts` with the
    /// metadata `record` it would get; `gate` is 1 when its subject resolved.
    /// An empty memory is bound as one all-zero inactive row (it never
    /// matches), so every tensor has at least one row.
    pub(crate) fn store(&self, parts: &[(usize, Hv)], record: &[f32], gate: f32) -> Store {
        let d2 = 2 * self.cfg.dim;
        let fillers: Vec<f32> = parts.iter().flat_map(|(_, f)| f.data.iter().copied()).collect();
        let sel: Vec<f32> = parts.iter().flat_map(|(r, _)| (0..ROLES.len()).map(move |j| if j == *r { 1.0 } else { 0.0 })).collect();
        let roles = self.roles_tensor();
        let rows = self.n_slots.max(1);
        let zero_meta = vec![0.0f32; rows * field::COUNT];
        let meta = if self.n_slots > 0 { view(&self.meta, rows) } else { View::new(rows, field::COUNT, &zero_meta) };
        let w = with_consts(&self.circuits.probe, vec![("slots", view(&self.slots, rows)), ("meta", meta), ("roles", View::new(ROLES.len(), d2, &roles))]);
        let inputs = vec![
            ("fillers", Buf::new(parts.len(), d2, fillers)),
            ("role_sel", Buf::new(parts.len(), ROLES.len(), sel)),
            ("record", Buf::row(record.to_vec())),
            ("gate", Buf::scalar(gate)),
            ("context", Buf::row(self.state.context.data.clone())),
        ];
        let o = self.exec(&self.circuits.probe, inputs, &w);
        Store {
            stored: o.scalar("stored") == 1.0,
            row: o.get("row").data.clone(),
            meta_row: o.get("meta_row").data.clone(),
            dup_slot: (o.scalar("dup") == 1.0).then(|| o.scalar("dup_slot") as usize),
            context: Hv::from_data(self.cfg.dim, &o.get("context_out").data),
        }
    }

    /// Run `symbol-cleanup`. With `gate` 1, a vector with no codebook entry
    /// is returned as the row to add. An empty codebook is bound as one zero
    /// row (cosine 0, never found).
    pub(crate) fn symbol_cleanup_full(&self, v: &Hv, mode: CleanupMode, gate: f32) -> Cleaned {
        let m = match mode {
            CleanupMode::Same => [0.0, 1.0, 0.0],
            CleanupMode::Pronoun => [0.0, 0.0, 1.0],
        };
        let rows = self.n_sym.max(1);
        let zero = vec![0.0f32; 2 * self.cfg.dim];
        let book = if self.n_sym > 0 { view(&self.codebook, rows) } else { View::new(1, 2 * self.cfg.dim, &zero) };
        let w = with_consts(&self.circuits.cleanup, vec![("codebook", book)]);
        let inputs = vec![("v", Buf::row(v.data.clone())), ("mode", Buf::row(m.to_vec())), ("gate", Buf::scalar(gate))];
        let o = self.exec(&self.circuits.cleanup, inputs, &w);
        Cleaned { found: (o.scalar("found") == 1.0).then(|| o.scalar("best") as usize), added: o.scalar("added") == 1.0, row: o.get("row").data.clone() }
    }

    /// The codebook index nearest to `v` if its cosine reaches the mode's threshold.
    pub(crate) fn symbol_cleanup(&self, v: &Hv, mode: CleanupMode) -> Option<usize> {
        self.symbol_cleanup_full(v, mode, 0.0).found
    }
}

/// One recorded circuit run (evaluation only): the exported bundle (program
/// and every weight exactly as bound), the inputs, and the engine's outputs.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub program: String,
    /// Shared by consecutive runs of one program with identical weights.
    pub bundle: std::sync::Arc<Vec<u8>>,
    pub inputs: Vec<(String, Buf)>,
    pub outputs: Outputs,
}

impl Reasoner {
    fn roles_tensor(&self) -> Vec<f32> {
        (0..ROLES.len()).flat_map(|r| self.role(r).data).collect()
    }

    /// Start (or stop) recording every circuit run for reference replay.
    pub fn record_circuits(&self, on: bool) {
        for r in [&self.recorder, &self.atoms.recorder, &self.numbers.recorder] {
            r.set(on);
        }
    }

    /// Take the runs recorded so far.
    pub fn take_recorded(&self) -> Vec<Recorded> {
        [&self.recorder, &self.atoms.recorder, &self.numbers.recorder].iter().flat_map(|r| r.take()).collect()
    }

    /// Bind, run, and (when recording) export one program run.
    pub(crate) fn exec(&self, c: &Compiled, inputs: Vec<(&str, Buf)>, weights: &[(&str, View<'_>)]) -> Outputs {
        run(c, inputs, weights, &self.recorder)
    }

    /// The decoded relation graph for the current memory state (cached).
    pub(crate) fn graph(&self) -> std::cell::Ref<'_, Graph> {
        let key = (self.n_slots, self.n_sym, self.clock, self.binding_ablated, self.retrieval_ablated);
        let stale = self.graph_cache.borrow().as_ref().map(|g| g.key != key).unwrap_or(true);
        if stale {
            let roles = self.roles_tensor();
            let zero_meta = vec![0.0f32; self.n_slots * field::COUNT];
            let meta = if self.retrieval_ablated { View::new(self.n_slots, field::COUNT, &zero_meta) } else { view(&self.meta, self.n_slots) };
            let mut w: Vec<(&str, View<'_>)> = vec![
                ("slots", view(&self.slots, self.n_slots)),
                ("meta", meta),
                ("codebook", view(&self.codebook, self.n_sym)),
                ("roles", View::new(ROLES.len(), 2 * self.cfg.dim, &roles)),
            ];
            w.extend(self.circuits.graph.tensors.iter().map(|t| (t.name.as_str(), t.view())));
            let o = self.exec(&self.circuits.graph, Vec::new(), &w);
            let take = |n: &str| o.values.get(n).cloned().expect("graph output");
            *self.graph_cache.borrow_mut() = Some(Graph { key, subj_sim: take("subj_sim"), obj_t: take("obj_t"), pos: take("pos"), neg: take("neg") });
        }
        std::cell::Ref::map(self.graph_cache.borrow(), |g| g.as_ref().expect("graph computed"))
    }

    /// Operator flags in `OPERATORS` order (evaluation ablations).
    fn op_flags(&self) -> [bool; 8] {
        let o = self.ops;
        [o.class_inclusion, o.transitivity, o.attribute_inheritance, o.comparison, o.arithmetic, o.temporal_ordering, o.negation, o.conditionals]
    }

    /// The program's compiled constants, with the enable constant of every
    /// disabled operator bound as zero (evaluation ablations).
    fn op_consts<'a>(&self, c: &'a Compiled, zero: &'a [f32; 1]) -> Vec<(&'a str, View<'a>)> {
        let disabled: Vec<String> = OPERATORS.iter().zip(self.op_flags()).filter(|(_, on)| !on).map(|(op, _)| format!("enable_{op}")).collect();
        c.tensors.iter().map(|t| (t.name.as_str(), if disabled.iter().any(|d| *d == t.name) { View::new(1, 1, zero) } else { t.view() })).collect()
    }

    /// Run a graph-reading circuit (C6 or a decision circuit) with the memory,
    /// the C5 outputs, and `extra` model tensors bound. `pol` is the name C6
    /// gives its polarity row; the decision circuits read `pos` and `neg`.
    fn run_on_graph(&self, c: &Compiled, inputs: Vec<(&str, Buf)>, extra: Vec<(&str, View<'_>)>) -> Outputs {
        let g = self.graph();
        let roles = self.roles_tensor();
        let mut w: Vec<(&str, View<'_>)> = vec![
            ("slots", view(&self.slots, self.n_slots)),
            ("codebook", view(&self.codebook, self.n_sym)),
            ("roles", View::new(ROLES.len(), 2 * self.cfg.dim, &roles)),
            ("subj_sim", g.subj_sim.view()),
            ("obj_t", g.obj_t.view()),
        ];
        if c.id == REACH_ID {
            w.push(("pol", g.pos.view()));
        } else {
            w.push(("pos", g.pos.view()));
            w.push(("neg", g.neg.view()));
        }
        w.extend(extra);
        let zero = [0.0f32];
        w.extend(self.op_consts(c, &zero));
        self.exec(c, inputs, &w)
    }

    /// C6 from vector `x` towards `y` over positive `rel` edges for operator
    /// `op` (a disabled operator keeps only direct edges). The caller has
    /// checked that the memory and codebook are not empty.
    pub(crate) fn run_reach(&self, x: &Hv, y: &Hv, rel: &Hv, op: &str) -> Outputs {
        let opi = OPERATORS.iter().position(|o| *o == op).expect("known operator");
        let mut sel = vec![0.0f32; OPERATORS.len()];
        sel[opi] = 1.0;
        let inputs = vec![("x", Buf::row(x.data.clone())), ("y", Buf::row(y.data.clone())), ("rel", Buf::row(rel.data.clone())), ("operator", Buf::row(sel))];
        self.run_on_graph(&self.circuits.reach, inputs, Vec::new())
    }

    /// `relation-join`: a negative `rel` edge from x (plus `src`'s reached
    /// nodes) to y (plus `tgt`'s), with `src`/`tgt` C6 outputs or `None`.
    pub(crate) fn run_join(&self, x: &Hv, y: &Hv, rel: &Hv, src: Option<&Outputs>, tgt: Option<&Outputs>) -> Outputs {
        let none = |_: ()| (Buf::zeros(1, self.n_sym), Buf::zeros(self.n_sym, self.n_slots));
        let side = |o: Option<&Outputs>| o.map(|o| (o.get("reached").clone(), o.get("proofs").clone())).unwrap_or_else(|| none(()));
        let ((sm, sp), (tm, tp)) = (side(src), side(tgt));
        let inputs = vec![
            ("x", Buf::row(x.data.clone())),
            ("y", Buf::row(y.data.clone())),
            ("rel", Buf::row(rel.data.clone())),
            ("src_more", sm),
            ("src_proofs", sp),
            ("tgt_more", tm),
            ("tgt_proofs", tp),
        ];
        self.run_on_graph(&self.circuits.join, inputs, Vec::new())
    }

    /// `relation-levels` ("can").
    pub(crate) fn run_levels(&self, x: &Hv, v: &Hv, rel_can: &Hv, rel_isa: &Hv) -> Outputs {
        let inputs = vec![("x", Buf::row(x.data.clone())), ("v", Buf::row(v.data.clone())), ("rel_can", Buf::row(rel_can.data.clone())), ("rel_isa", Buf::row(rel_isa.data.clone()))];
        self.run_on_graph(&self.circuits.levels, inputs, Vec::new())
    }

    /// `relation-rules` (properties and conditionals).
    pub(crate) fn run_rules(&self, x: &Hv, p: &Hv, rel_prop: &Hv, rel_implies: &Hv, something: &Hv) -> Outputs {
        let inputs = vec![
            ("x", Buf::row(x.data.clone())),
            ("p", Buf::row(p.data.clone())),
            ("rel_prop", Buf::row(rel_prop.data.clone())),
            ("rel_implies", Buf::row(rel_implies.data.clone())),
            ("something", Buf::row(something.data.clone())),
        ];
        self.run_on_graph(&self.circuits.rules, inputs, Vec::new())
    }

    /// `relation-counts` (arithmetic).
    pub(crate) fn run_counts(&self, x: &Hv, rel_has: &Hv, rel_more: &Hv) -> Outputs {
        let inputs = vec![("x", Buf::row(x.data.clone())), ("rel_has", Buf::row(rel_has.data.clone())), ("rel_more", Buf::row(rel_more.data.clone()))];
        let book = &self.numbers().codebook;
        self.run_on_graph(&self.circuits.counts, inputs, vec![("numbers", view(book, book.rows))])
    }

    /// `relation-verdict`: findings to a one-hot verdict and premises.
    pub(crate) fn run_verdict(&self, pos_found: f32, pos_proof: &Buf, neg_found: f32, neg_proof: &Buf) -> Outputs {
        let inputs = vec![("pos_found", Buf::scalar(pos_found)), ("neg_found", Buf::scalar(neg_found)), ("pos_proof", pos_proof.clone()), ("neg_proof", neg_proof.clone())];
        let w = with_consts(&self.circuits.verdict, Vec::new());
        self.exec(&self.circuits.verdict, inputs, &w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerolm_engine::binding::{cosine, sim};

    #[test]
    fn atoms_are_unit_phasors_distinct_and_deterministic() {
        let cfg = ReasonerConfig::default();
        let a = Atoms::new(&cfg);
        let names = ["cat", "cats", "tom", "tomas", "a", "b", "mammal", "animal", "x1", "x2", "x10"];
        let vs: Vec<Hv> = names.iter().map(|n| a.encode("sym", n)).collect();
        for v in &vs {
            for k in 0..cfg.dim {
                let m = v.data[k] * v.data[k] + v.data[cfg.dim + k] * v.data[cfg.dim + k];
                assert!((m - 1.0).abs() < 1e-5, "component {k} has |z|^2 = {m}");
            }
        }
        for i in 0..vs.len() {
            for j in 0..i {
                assert!(cosine(&vs[i], &vs[j]).abs() < 0.15, "{} vs {}: {}", names[i], names[j], cosine(&vs[i], &vs[j]));
            }
        }
        assert!((sim(&vs[0], &vs[0]) - 1.0).abs() < 1e-5);
        let again = Atoms::new(&cfg).encode("sym", "cat");
        assert_eq!(again.data, vs[0].data);
        assert!(cosine(&a.encode("role", "cat"), &vs[0]).abs() < 0.15, "namespaces differ");
    }

    #[test]
    fn number_circuits_encode_numerals_and_decode_by_cleanup() {
        let cfg = ReasonerConfig::default();
        let nums = Numbers::new(&cfg);
        let space = Space::new(cfg.seed, cfg.dim);
        for n in [0, 1, 7, 10, 42, 100, 907, 1000, -1, -2, -58, -1000, 123_456, 1_000_000, -999_999] {
            let v = nums.encode(n);
            assert!((sim(&v, &space.number(n)) - 1.0).abs() < 1e-4, "{n}: {}", sim(&v, &space.number(n)));
            if n.abs() <= cfg.number_range {
                let d = nums.decode(&v);
                assert_eq!(d.value, n);
                assert_eq!(d.clean.data, space.number(n).data, "clean row is the codebook row");
            }
        }
        // Addition is binding: 38 + (-45) = -7, decoded from clean vectors.
        let (a, b) = (nums.decode(&nums.encode(38)), nums.decode(&nums.encode(-45)));
        assert_eq!(nums.decode(&zerolm_engine::binding::bind(&a.clean, &b.clean)).value, -7);
        // A noisy superposition still decodes to its dominant number.
        let mut noisy = nums.encode(321);
        noisy.add_assign(&Atoms::new(&cfg).encode("sym", "noise"));
        assert_eq!(nums.decode(&noisy).value, 321);
    }

    #[test]
    fn number_circuit_blocks_carry_what_they_claim() {
        let cfg = ReasonerConfig::default();
        let space = Space::new(cfg.seed, cfg.dim);
        let mut nums = Numbers::new(&cfg);
        nums.encoder.ablate_block("sign").unwrap();
        // Without the sign block, the output is zeroed (the sign mask is 0).
        assert!(nums.encode(-12).is_zero());
        let mut nums = Numbers::new(&cfg);
        nums.encoder.ablate_block("position").unwrap();
        // Without places every digit selects all 7 of its rows: a 49-term
        // superposition in which number(12) is one term (cosine about 1/7).
        assert!(cosine(&nums.encode(12), &space.number(12)) < 0.5);
        let mut nums = Numbers::new(&cfg);
        nums.decoder.ablate_block("readout").unwrap();
        assert_eq!(nums.decode(&space.number(12)).value, 0, "no values to read out");
        let nums = Numbers::new(&cfg);
        assert!((sim(&nums.encode(-12), &space.number(-12)) - 1.0).abs() < 1e-4);
    }
}
