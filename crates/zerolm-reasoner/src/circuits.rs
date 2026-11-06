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
use zerolm_engine::isa::{self, Bindings, Buf, Outputs, View};
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

impl Reasoner {
    fn roles_tensor(&self) -> Vec<f32> {
        (0..ROLES.len()).flat_map(|r| self.role(r).data).collect()
    }

    /// The decoded relation graph for the current memory state (cached).
    pub(crate) fn graph(&self) -> std::cell::Ref<'_, Graph> {
        let key = (self.n_slots, self.n_sym, self.clock, self.binding_ablated, self.retrieval_ablated);
        let stale = self.graph_cache.borrow().as_ref().map(|g| g.key != key).unwrap_or(true);
        if stale {
            let roles = self.roles_tensor();
            let zero_meta = vec![0.0f32; self.n_slots * field::COUNT];
            let meta = if self.retrieval_ablated { View::new(self.n_slots, field::COUNT, &zero_meta) } else { view(&self.meta, self.n_slots) };
            let mut b = Bindings::new()
                .weight("slots", view(&self.slots, self.n_slots))
                .weight("meta", meta)
                .weight("codebook", view(&self.codebook, self.n_sym))
                .weight("roles", View::new(ROLES.len(), 2 * self.cfg.dim, &roles));
            for t in &self.circuits.graph.tensors {
                b = b.weight(&t.name, t.view());
            }
            let o = isa::run(&self.circuits.graph.program, b).expect("relation-graph runs on its own memory");
            let take = |n: &str| o.values.get(n).cloned().expect("graph output");
            *self.graph_cache.borrow_mut() = Some(Graph { key, subj_sim: take("subj_sim"), obj_t: take("obj_t"), pos: take("pos"), neg: take("neg") });
        }
        std::cell::Ref::map(self.graph_cache.borrow(), |g| g.as_ref().expect("graph computed"))
    }

    fn op_enabled(&self, op: &str) -> bool {
        let o = self.ops;
        match op {
            "class_inclusion" => o.class_inclusion,
            "transitivity" => o.transitivity,
            "attribute_inheritance" => o.attribute_inheritance,
            "comparison" => o.comparison,
            "arithmetic" => o.arithmetic,
            "temporal_ordering" => o.temporal_ordering,
            "negation" => o.negation,
            "conditionals" => o.conditionals,
            other => panic!("unknown operator {other}"),
        }
    }

    /// Run C6 from symbol `from` over positive `rel` edges for operator `op`
    /// (a disabled operator's block is bound as zero: direct edges only).
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
        let mut b = Bindings::new()
            .input("x", Buf::row(x.data))
            .input("y", Buf::row(y.data))
            .input("rel", Buf::row(rel.data.clone()))
            .input("operator", Buf::row(sel))
            .weight("slots", view(&self.slots, self.n_slots))
            .weight("codebook", view(&self.codebook, self.n_sym))
            .weight("roles", View::new(ROLES.len(), 2 * self.cfg.dim, &roles))
            .weight("subj_sim", g.subj_sim.view())
            .weight("obj_t", g.obj_t.view())
            .weight("pol", g.pos.view());
        let zero = [0.0f32];
        for t in &self.circuits.reach.tensors {
            let disabled = t.name.strip_prefix("enable_").map(|op| !self.op_enabled(op)).unwrap_or(false);
            b = b.weight(&t.name, if disabled { View::new(1, 1, &zero) } else { t.view() });
        }
        Some(isa::run(&self.circuits.reach.program, b).expect("relation-reach runs on its own memory"))
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
