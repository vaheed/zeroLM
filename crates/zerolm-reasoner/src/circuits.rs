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
use zerolm_engine::binding::Hv;
use zerolm_engine::isa::{self, bundle, Bindings, Buf, Outputs, View};
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

pub struct Circuits {
    pub graph: Compiled,
    pub reach: Compiled,
}

impl Circuits {
    pub fn new(cfg: &ReasonerConfig) -> Self {
        Circuits {
            graph: compile(&graph_spec(cfg)).expect("valid reasoning config compiles"),
            reach: compile(&reach_spec(cfg)).expect("valid reasoning config compiles"),
        }
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

/// Reached symbols (codebook index) -> premise slots of the shortest proof.
pub type Reach = BTreeMap<usize, Vec<usize>>;

fn view(m: &zerolm_tensor::Matrix, rows: usize) -> View<'_> {
    View::new(rows, m.cols, &m.data[..rows * m.cols])
}

/// One recorded circuit run (evaluation only): the exported bundle (program
/// and every weight exactly as bound), the inputs, and the engine's outputs.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub program: String,
    pub bundle: Vec<u8>,
    pub inputs: Vec<(String, Buf)>,
    pub outputs: Outputs,
}

impl Reasoner {
    fn roles_tensor(&self) -> Vec<f32> {
        (0..ROLES.len()).flat_map(|r| self.role(r).data).collect()
    }

    /// Start (or stop) recording every circuit run for reference replay.
    pub fn record_circuits(&self, on: bool) {
        *self.recorder.borrow_mut() = if on { Some(Vec::new()) } else { None };
    }

    /// Take the runs recorded so far.
    pub fn take_recorded(&self) -> Vec<Recorded> {
        self.recorder.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Bind, run, and (when recording) export one program run.
    fn exec(&self, c: &Compiled, inputs: Vec<(&str, Buf)>, weights: &[(&str, View<'_>)]) -> Outputs {
        let mut b = Bindings::new();
        for (n, t) in &inputs {
            b = b.input(n, t.clone());
        }
        for (n, v) in weights {
            b = b.weight(n, *v);
        }
        let out = isa::run(&c.program, b).unwrap_or_else(|e| panic!("{} runs on its own memory: {e}", c.id));
        if let Some(rec) = self.recorder.borrow_mut().as_mut() {
            rec.push(Recorded {
                program: c.id.clone(),
                bundle: bundle::encode(&c.program, weights),
                inputs: inputs.into_iter().map(|(n, t)| (n.to_string(), t)).collect(),
                outputs: out.clone(),
            });
        }
        out
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

    /// Run C6 from symbol `from` over positive `rel` edges for operator `op`
    /// (a disabled operator's enable block is bound as zero: direct edges only).
    pub(crate) fn run_reach(&self, from: usize, to: Option<usize>, rel: &Hv, op: &str) -> Option<Outputs> {
        if self.n_slots == 0 || self.n_sym == 0 {
            return None;
        }
        let g = self.graph();
        let roles = self.roles_tensor();
        let opi = OPERATORS.iter().position(|o| *o == op).expect("known operator");
        let mut sel = vec![0.0f32; OPERATORS.len()];
        sel[opi] = 1.0;
        let x = self.code(from);
        let y = to.map(|t| self.code(t)).unwrap_or_else(|| x.clone());
        let inputs = vec![("x", Buf::row(x.data)), ("y", Buf::row(y.data)), ("rel", Buf::row(rel.data.clone())), ("operator", Buf::row(sel))];
        let mut w: Vec<(&str, View<'_>)> = vec![
            ("slots", view(&self.slots, self.n_slots)),
            ("codebook", view(&self.codebook, self.n_sym)),
            ("roles", View::new(ROLES.len(), 2 * self.cfg.dim, &roles)),
            ("subj_sim", g.subj_sim.view()),
            ("obj_t", g.obj_t.view()),
            ("pol", g.pos.view()),
        ];
        let zero = [0.0f32];
        let disabled: Vec<String> =
            OPERATORS.iter().zip(self.op_flags()).filter(|(_, on)| !on).map(|(op, _)| format!("enable_{op}")).collect();
        for t in &self.circuits.reach.tensors {
            let off = disabled.iter().any(|d| *d == t.name);
            w.push((t.name.as_str(), if off { View::new(1, 1, &zero) } else { t.view() }));
        }
        Some(self.exec(&self.circuits.reach, inputs, &w))
    }

    /// Every symbol reachable from `from` with its proof premises.
    pub(crate) fn reach(&self, from: usize, rel: &Hv, op: &str) -> Reach {
        let Some(o) = self.run_reach(from, None, rel, op) else { return Reach::new() };
        let reached = &o.get("reached").data;
        let proofs = o.get("proofs");
        let mut out = Reach::new();
        for (j, &r) in reached.iter().enumerate() {
            if r == 1.0 {
                let row = &proofs.data[j * proofs.cols..(j + 1) * proofs.cols];
                out.insert(j, row.iter().enumerate().filter(|(_, &v)| v > 0.0).map(|(s, _)| s).collect());
            }
        }
        out
    }
}
