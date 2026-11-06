//! Phase 6 relational inference as circuits (roadmap C5, C6).
//!
//! **C5 `relation_graph`** decodes the relation memory (bound role-filler
//! bundles, one row per statement) into slot x symbol matrices, once per
//! memory state:
//! - `subj_sim[s, i] = sim(slot_s, subj (*) c_i)`: how strongly slot s binds
//!   symbol i in the subject role (about 1 if it does, about 0 otherwise);
//! - `obj_t[s, j]`: one-hot cleanup of `slot_s (/) obj` to the codebook
//!   (the nearest symbol with cosine >= cleanup_min, else a zero row);
//! - `pos[s]`, `neg[s]`: active slots of positive and negative polarity.
//!
//! **C6 `relation_reach`** answers "is y reachable from x over relation r?"
//! as a fixed number of recurrent steps instead of a graph search. With
//! `E[s, i] = [subj_sim[s, i] + sim(slot_s, rel (*) r) >= 2 - margin] * pol[s]`
//! (slot s is an r-edge leaving symbol i), frontier f, reached set R, and
//! premise matrix P (P[j, s] = 1 if slot s is on the proof of j):
//! ```text
//! g    = E f                       slots leaving the frontier
//! cand = (obj_t * g) * (1 - R)     new targets per slot
//! Q    = top1 per target over cand lowest slot index wins
//! P   += Q (E*f) P + Q             parent's proof plus the edge slot
//! R   += rowmax(Q);  f = rowmax(Q)
//! ```
//! A depth gate (1 for the first step, the operator's `enable` constant
//! afterwards) makes "operator disabled" mean "direct statements only", which
//! is the Phase 6 ablation semantics.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn::*, Instr, IsaError, NormKind, ReduceFn};

/// Operators whose `enable` constant (one block each) the reach circuit reads.
pub const OPERATORS: [&str; 8] =
    ["class_inclusion", "transitivity", "attribute_inheritance", "comparison", "arithmetic", "temporal_ordering", "negation", "conditionals"];

/// Role rows of `core.reasoner.roles`.
pub const ROLE_SUBJ: usize = 0;
pub const ROLE_REL: usize = 1;
pub const ROLE_OBJ: usize = 2;
pub const N_ROLES: usize = 5;
/// Metadata columns of `memory.relations.meta`.
pub const META_ACTIVE: usize = 0;
pub const META_POLARITY: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    /// Complex dimension d (rows are 2d floats).
    pub dim: usize,
    pub meta_cols: usize,
    pub max_depth: usize,
    pub match_margin: f32,
    pub cleanup_min: f32,
    pub symbol_same: f32,
}

fn check(id: &str, p: &Params) -> Result<(), IsaError> {
    if p.dim == 0 || p.max_depth == 0 || p.meta_cols <= META_POLARITY {
        return Err(IsaError(format!("{id}: invalid relational parameters")));
    }
    Ok(())
}

fn onehot(n: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0.0; n];
    v[i] = 1.0;
    v
}

fn support(b: &mut Builder, p: &Params) {
    b.block("support");
    b.scalar("one", 1.0);
    b.scalar("zero", 0.0);
    b.scalar("half", 0.5);
    b.scalar("inv_d", 1.0 / p.dim as f32);
    b.scalar("inv_sqrt_d", 1.0 / (p.dim as f32).sqrt());
    b.scalar("eps", 0.0);
}

/// Row `role` of the roles table as a 1 x 2d vector.
fn role(b: &mut Builder, role: usize, out: &str) {
    let sel = format!("{out}_sel");
    b.constant(&sel, 1, N_ROLES, onehot(N_ROLES, role));
    b.matvec("roles", &sel, out, true);
}

/// C5: decode the relation memory.
pub fn compile_graph(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let d2 = 2 * p.dim;
    let mut b = Builder::new(id);
    b.external_dyn("slots", None, Some(d2));
    b.external_dyn("meta", None, Some(p.meta_cols));
    b.external_dyn("codebook", None, Some(d2));
    b.external("roles", N_ROLES, d2);
    support(&mut b, p);

    b.block("decode");
    b.scalar("cleanup_min", p.cleanup_min);
    role(&mut b, ROLE_SUBJ, "role_subj");
    role(&mut b, ROLE_OBJ, "role_obj");
    // Subject similarity: slots against every codebook symbol bound to subj.
    b.op(Instr::Bind { a: "codebook".into(), b: "role_subj".into(), out: "subj_codes".into(), conj: false });
    b.matmul("slots", "subj_codes", "subj_dot");
    b.gate(Mul, "subj_dot", "inv_d", "subj_sim");
    // Object cleanup: unbind obj, cosine against unit-scaled codebook, best >= min.
    b.op(Instr::Bind { a: "slots".into(), b: "role_obj".into(), out: "obj_noisy".into(), conj: true });
    b.op(Instr::Norm { kind: NormKind::L2, x: "obj_noisy".into(), eps: "eps".into(), out: "obj_unit".into() });
    b.gate(Mul, "codebook", "inv_sqrt_d", "codes_unit");
    b.matmul("obj_unit", "codes_unit", "obj_cos");
    b.op(Instr::Topk { x: "obj_cos".into(), k: 1, idx: None, val: Some("obj_best".into()), mask: Some("obj_mask".into()) });
    b.gate(Ge, "obj_best", "cleanup_min", "obj_ok");
    b.gate(Mul, "obj_mask", "obj_ok", "obj_t");

    b.block("polarity");
    b.constant("active_sel", 1, p.meta_cols, onehot(p.meta_cols, META_ACTIVE));
    b.constant("polarity_sel", 1, p.meta_cols, onehot(p.meta_cols, META_POLARITY));
    b.matvec("meta", "active_sel", "active", false);
    b.matvec("meta", "polarity_sel", "polarity", false);
    b.gate(Ge, "active", "half", "is_active");
    b.gate(Ge, "polarity", "half", "is_pos");
    b.unary(Neg, "polarity", "neg_polarity");
    b.gate(Ge, "neg_polarity", "half", "is_neg");
    b.gate(Mul, "is_pos", "is_active", "pos");
    b.gate(Mul, "is_neg", "is_active", "neg");
    Ok(b.finish(&["subj_sim", "obj_t", "pos", "neg"]))
}

/// Cleanup of a query vector to a one-hot codebook row (cosine >= symbol_same).
fn cleanup_query(b: &mut Builder, v: &str) {
    b.op(Instr::Norm { kind: NormKind::L2, x: v.into(), eps: "eps".into(), out: format!("{v}_unit") });
    b.matvec("codes_unit", &format!("{v}_unit"), &format!("{v}_cos"), false);
    b.op(Instr::Topk { x: format!("{v}_cos"), k: 1, idx: None, val: Some(format!("{v}_best")), mask: Some(format!("{v}_mask")) });
    b.gate(Ge, &format!("{v}_best"), "symbol_same", &format!("{v}_known"));
    b.gate(Mul, &format!("{v}_mask"), &format!("{v}_known"), &format!("{v}_onehot"));
}

/// C6: reachability with provenance.
pub fn compile_reach(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let (d2, depth) = (2 * p.dim, p.max_depth);
    let mut b = Builder::new(id);
    b.input("x", Some(1), d2);
    b.input("y", Some(1), d2);
    b.input("rel", Some(1), d2);
    // Which operator is running (one-hot over OPERATORS): structure, not content.
    b.input("operator", Some(1), OPERATORS.len());
    b.external_dyn("slots", None, Some(d2));
    b.external_dyn("codebook", None, Some(d2));
    b.external("roles", N_ROLES, d2);
    b.external_dyn("subj_sim", None, None);
    b.external_dyn("obj_t", None, None);
    b.external_dyn("pol", Some(1), None);
    support(&mut b, p);

    // One enable constant per operator: zeroing a block disables that operator.
    for op in OPERATORS {
        b.block(op);
        b.scalar(&format!("enable_{op}"), 1.0);
    }
    b.block("depth");
    b.constant("first_step", 1, depth, onehot(depth, 0));
    b.constant("later_steps", 1, depth, (0..depth).map(|t| (t > 0) as u8 as f32).collect());
    let mut shift = vec![0.0; depth * depth];
    for t in 0..depth.saturating_sub(1) {
        shift[t * depth + t + 1] = 1.0;
    }
    b.constant("shift", depth, depth, shift);
    b.block("probe");
    b.scalar("need", 2.0 - p.match_margin);
    b.scalar("symbol_same", p.symbol_same);

    // enable = sum_k operator_k * enable_k
    b.block("probe");
    let mut acc = "enable_acc_0".to_string();
    b.gate(Mul, "operator", "zero", &acc);
    b.reduce(ReduceFn::Sum, &acc.clone(), Axis::All, &acc);
    for (k, op) in OPERATORS.iter().enumerate() {
        let sel = format!("op_sel_{k}");
        b.constant(&sel, 1, OPERATORS.len(), onehot(OPERATORS.len(), k));
        b.gate(Mul, "operator", &sel, &format!("{sel}_on"));
        b.reduce(ReduceFn::Sum, &format!("{sel}_on"), Axis::All, &format!("{sel}_flag"));
        b.gate(Mul, &format!("{sel}_flag"), &format!("enable_{op}"), &format!("{sel}_en"));
        let next = format!("enable_acc_{}", k + 1);
        b.gate(Add, &acc, &format!("{sel}_en"), &next);
        acc = next;
    }

    // Edges E[s, i]: slot s has subject i and relation r, with the bound polarity.
    role(&mut b, ROLE_REL, "role_rel");
    b.op(Instr::Bind { a: "rel".into(), b: "role_rel".into(), out: "rel_bound".into(), conj: false });
    b.matvec("slots", "rel_bound", "rel_dot", false);
    b.gate(Mul, "rel_dot", "inv_d", "rel_sim_row");
    b.op(Instr::Transpose { x: "rel_sim_row".into(), out: "rel_sim".into() });
    b.gate(Add, "subj_sim", "rel_sim", "probe_sim");
    b.gate(Ge, "probe_sim", "need", "probe_hit");
    b.op(Instr::Transpose { x: "pol".into(), out: "pol_col".into() });
    b.gate(Mul, "probe_hit", "pol_col", "edges");

    // Query symbols.
    b.gate(Mul, "codebook", "inv_sqrt_d", "codes_unit");
    cleanup_query(&mut b, "x");
    cleanup_query(&mut b, "y");

    // Recurrence state.
    b.block("depth");
    b.unary(Copy, "x_onehot", "frontier");
    b.unary(Copy, "x_onehot", "reached_all");
    // Target-major layouts (N x S), built once outside the loop.
    b.op(Instr::Transpose { x: "edges".into(), out: "edges_t".into() });
    b.op(Instr::Transpose { x: "obj_t".into(), out: "obj_tt".into() });
    b.gate(Mul, "edges_t", "zero", "proofs");
    b.gate(Mul, "later_steps", &acc, "later_enabled");
    b.gate(Add, "first_step", "later_enabled", "depth_gate");
    let g = |f, a: &str, b: &str, out: &str| Instr::Gate { f, a: a.into(), b: Some(b.into()), out: out.into() };
    b.op(Instr::Step {
        n: depth,
        body: vec![
            // Depth gate for this step, then shift the schedule.
            g(Mul, "depth_gate", "first_step", "gate_now_row"),
            Instr::Reduce { f: ReduceFn::Sum, x: "gate_now_row".into(), axis: Axis::All, out: "gate_now".into() },
            Instr::Matvec { w: "shift".into(), x: "depth_gate".into(), out: "depth_gate".into(), transpose: false },
            // leaving[s] = sum_i E[s, i] f[i]: slots leaving the frontier.
            Instr::Matvec { w: "edges_t".into(), x: "frontier".into(), out: "leaving".into(), transpose: true },
            // cand[j, s] = obj[s, j] * leaving[s] * unreached[j] * gate.
            g(Sub, "one", "reached_all", "unreached_row"),
            Instr::Transpose { x: "unreached_row".into(), out: "unreached".into() },
            g(Mul, "unreached", "gate_now", "open"),
            g(Mul, "obj_tt", "leaving", "cand_any"),
            g(Mul, "cand_any", "open", "cand"),
            // One slot per new target (lowest index wins); pick_val is 1 for a new target.
            Instr::Topk { x: "cand".into(), k: 1, idx: None, val: Some("pick_val".into()), mask: Some("pick_mask".into()) },
            g(Mul, "pick_mask", "pick_val", "pick"),
            // Proof of j = proof of the slot's subject + the slot: pick E P + pick.
            Instr::Matmul { a: "pick".into(), b: "edges".into(), out: "parent".into(), ab: true },
            Instr::Matmul { a: "parent".into(), b: "proofs".into(), out: "inherited".into(), ab: true },
            g(Add, "proofs", "inherited", "proofs"),
            g(Add, "proofs", "pick", "proofs"),
            Instr::Transpose { x: "pick_val".into(), out: "new".into() },
            g(Add, "reached_all", "new", "reached_all"),
            Instr::Gate { f: Copy, a: "new".into(), b: None, out: "frontier".into() },
        ],
    });
    // Answer: is y reached (excluding x itself), and its proof.
    b.gate(Sub, "reached_all", "x_onehot", "reached");
    b.gate(Mul, "reached", "y_onehot", "hit_row");
    b.reduce(ReduceFn::Sum, "hit_row", Axis::All, "hit");
    b.matvec("proofs", "y_onehot", "proof", true);
    Ok(b.finish(&["hit", "proof", "reached", "proofs", "x_onehot", "y_onehot", "x_known", "y_known", "enable_acc_8"]))
}
