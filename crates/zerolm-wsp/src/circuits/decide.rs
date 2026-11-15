//! The reasoning operators' decisions as circuits, on top of C5
//! (`relation_graph`: `subj_sim`, `obj_t`, `pos`, `neg`) and C6
//! (`relation_reach`). Each circuit is a fixed number of recurrent steps; the
//! host only chooses which circuits a question's parsed form runs, passes
//! one circuit's outputs to the next unchanged, and labels the outputs.
//!
//! Shared notation: S slots, N codebook symbols, `E_r[s, i] = [subj_sim[s, i]
//! + sim(slot_s, rel (*) r) >= 2 - margin] * pol[s]` (slot s is an r-edge
//! leaving i with polarity pol), proofs as slot masks (1 x S) or premise
//! matrices (N x S, row j = slots on the proof of j). Among candidates, the
//! shortest proof wins (score `BIG - length`), ties to the lowest slot.
//!
//! - **`relation_join`** (negation): a negative r-edge from a source set to a
//!   target set. Sources are x (cleaned up from its vector) plus, when class
//!   inclusion is enabled, `src_more` with premises `src_proofs` (for "is x
//!   a y": x and its ancestors); targets likewise from y. Outputs `found`, `proof`, `len`.
//! - **`relation_levels`** ("can"): breadth-first levels over positive `isa`
//!   edges from x; at the first level where some node has a positive or (with
//!   negation) negative `can` edge to v, both are latched. Outputs
//!   `pos_found`, `pos_proof`, `neg_found`, `neg_proof`, `level`.
//! - **`relation_rules`** (property, conditionals): backward levels from
//!   property p over rules "if <scope> is q then it is p" (scope "something"
//!   or x itself); the first level at which x has a positive property edge to
//!   a frontier property is latched. Outputs `found`, `proof`, `level`.
//! - **`relation_counts`** (arithmetic): breadth-first over `more` edges from
//!   x, carrying the summed difference as a number phasor (binding is
//!   addition); every reached node's `has` count is cleaned up against the
//!   number codebook, bound to its carried difference, and decoded. One
//!   distinct value answers it (shortest proof), several are a contradiction
//!   (all proofs). Outputs `found`, `contradiction`, `value`, `premises`,
//!   `arith`.
//! - **`relation_verdict`**: positive and negative findings to a one-hot
//!   verdict (yes, no, contradiction, unknown) and the union of the proofs
//!   that support it.

use super::relational::{check, cleanup_query, onehot, role, support, Params, N_ROLES, ROLE_OBJ, ROLE_REL, ROLE_SUBJ};
use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn::*, Instr, IsaError, NormKind, ReduceFn};

pub const ROLE_QUANT: usize = 3;
pub const ROLE_SCOPE: usize = 4;
/// Proof-length scores: `BIG - length` (exact in f32).
const BIG: f32 = 1.0e6;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountParams {
    pub relational: Params,
    /// The number codebook covers [-range, range].
    pub range: u32,
}

pub const JOIN_OUTPUTS: [&str; 3] = ["found", "proof", "len"];
pub const LEVELS_OUTPUTS: [&str; 5] = ["pos_found", "pos_proof", "neg_found", "neg_proof", "level"];
pub const RULES_OUTPUTS: [&str; 3] = ["found", "proof", "level"];
pub const COUNTS_OUTPUTS: [&str; 5] = ["found", "contradiction", "value", "premises", "arith"];
pub const VERDICT_OUTPUTS: [&str; 2] = ["verdict", "premises"];

fn g(f: zerolm_engine::isa::GateFn, a: &str, b: &str, out: &str) -> Instr {
    Instr::Gate { f, a: a.into(), b: Some(b.into()), out: out.into() }
}

/// Graph tensors (C5 outputs) and memory, shared by the search circuits.
fn graph_io(b: &mut Builder, d2: usize) {
    b.external_dyn("slots", None, Some(d2));
    b.external_dyn("codebook", None, Some(d2));
    b.external("roles", N_ROLES, d2);
    b.external_dyn("subj_sim", None, None);
    b.external_dyn("obj_t", None, None);
    b.external_dyn("pos", Some(1), None);
    b.external_dyn("neg", Some(1), None);
}

fn probe_block(b: &mut Builder, p: &Params) {
    b.block("probe");
    b.scalar("need", 2.0 - p.match_margin);
    b.scalar("symbol_same", p.symbol_same);
    b.scalar("cleanup_min", p.cleanup_min);
    role(b, ROLE_REL, "role_rel");
    b.gate(Mul, "codebook", "inv_sqrt_d", "codes_unit");
}

/// `out` (S x N): r-edges with polarity row `pol` (1 x S).
fn edges(b: &mut Builder, rel: &str, pol: &str, out: &str) {
    b.op(Instr::Bind { a: rel.into(), b: "role_rel".into(), out: format!("{out}_rb"), conj: false });
    b.matvec("slots", &format!("{out}_rb"), &format!("{out}_dot"), false);
    b.gate(Mul, &format!("{out}_dot"), "inv_d", &format!("{out}_row"));
    b.op(Instr::Transpose { x: format!("{out}_row"), out: format!("{out}_col") });
    b.gate(Add, "subj_sim", &format!("{out}_col"), &format!("{out}_sim"));
    b.gate(Ge, &format!("{out}_sim"), "need", &format!("{out}_hit"));
    b.op(Instr::Transpose { x: pol.into(), out: format!("{out}_pc") });
    b.gate(Mul, &format!("{out}_hit"), &format!("{out}_pc"), out);
}

/// Zero premise matrix (N x S) and a zero slot row, from `obj_t`.
fn zeros(b: &mut Builder, px: &str) {
    b.op(Instr::Transpose { x: "obj_t".into(), out: format!("{px}_zt") });
    b.gate(Mul, &format!("{px}_zt"), "zero", &format!("{px}_proofs"));
}

/// One breadth-first expansion (the C6 step): from `frontier` over edge
/// matrix `e` (S x N, with `e_t` = N x S) to targets `to_t` (N x S), gated
/// by the 1 x 1 `gate`. Updates `{px}_frontier`, `{px}_reached`,
/// `{px}_proofs`; with `carry`, also the phasor rows `{px}_acc` (N x 2d),
/// bound with the picked slot's row of `carry` (S x 2d).
fn expand(px: &str, e: &str, e_t: &str, to_t: &str, gate: &str, carry: Option<&str>) -> Vec<Instr> {
    let n = |s: &str| format!("{px}_{s}");
    let mut v = vec![
        Instr::Matvec { w: e_t.into(), x: n("frontier"), out: n("leaving"), transpose: true },
        g(Sub, "one", &n("reached"), &n("unreached_row")),
        Instr::Transpose { x: n("unreached_row"), out: n("unreached") },
        g(Mul, &n("unreached"), gate, &n("open")),
        g(Mul, to_t, &n("leaving"), &n("cand_any")),
        g(Mul, &n("cand_any"), &n("open"), &n("cand")),
        Instr::Topk { x: n("cand"), k: 1, idx: None, val: Some(n("pick_val")), mask: Some(n("pick_mask")) },
        g(Mul, &n("pick_mask"), &n("pick_val"), &n("pick")),
        Instr::Matmul { a: n("pick"), b: e.into(), out: n("parent"), ab: true },
        Instr::Matmul { a: n("parent"), b: n("proofs"), out: n("inherited"), ab: true },
    ];
    if let Some(c) = carry {
        v.extend([
            Instr::Matmul { a: n("parent"), b: n("acc"), out: n("acc_parent"), ab: true },
            Instr::Matmul { a: n("pick"), b: c.into(), out: n("acc_step"), ab: true },
            Instr::Bind { a: n("acc_parent"), b: n("acc_step"), out: n("acc_new"), conj: false },
            g(Add, &n("acc"), &n("acc_new"), &n("acc")),
        ]);
    }
    v.extend([
        g(Add, &n("proofs"), &n("inherited"), &n("proofs")),
        g(Add, &n("proofs"), &n("pick"), &n("proofs")),
        Instr::Transpose { x: n("pick_val"), out: n("new") },
        g(Add, &n("reached"), &n("new"), &n("reached")),
        Instr::Gate { f: Copy, a: n("new"), b: None, out: n("frontier") },
    ]);
    v
}

/// Latch the best (shortest) candidate slot of `cand` (1 x S) whose subject
/// path is in `paths` (N x S) via `e` (S x N): writes `{px}_val` (1 or 0),
/// `{px}_proof` (1 x S), `{px}_len` (1 x 1).
fn best_proof(px: &str, cand: &str, e: &str, paths: &str) -> Vec<Instr> {
    let n = |s: &str| format!("{px}_{s}");
    vec![
        // Path length of each slot's subject.
        Instr::Reduce { f: ReduceFn::Sum, x: paths.into(), axis: Axis::Rows, out: n("plen_col") },
        Instr::Matmul { a: e.into(), b: n("plen_col"), out: n("slot_plen_col"), ab: true },
        Instr::Transpose { x: n("slot_plen_col"), out: n("slot_plen") },
        g(Sub, "big_minus_one", &n("slot_plen"), &n("score_raw")),
        g(Mul, &n("score_raw"), cand, &n("score")),
        Instr::Topk { x: n("score"), k: 1, idx: None, val: Some(n("best")), mask: Some(n("mask")) },
        g(Ge, &n("best"), "one", &n("val")),
        g(Mul, &n("mask"), &n("val"), &n("slot")),
        Instr::Matmul { a: n("slot"), b: e.into(), out: n("node"), ab: true },
        Instr::Matmul { a: n("node"), b: paths.into(), out: n("path"), ab: true },
        g(Add, &n("slot"), &n("path"), &n("proof")),
        Instr::Reduce { f: ReduceFn::Sum, x: n("proof"), axis: Axis::All, out: n("len") },
    ]
}

fn length_constants(b: &mut Builder) {
    b.block("support");
    b.scalar("big_minus_one", BIG - 1.0);
}

/// Negative r-edges from a source set to a target set.
pub fn compile_join(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let d2 = 2 * p.dim;
    let mut b = Builder::new(id);
    b.input("x", Some(1), d2);
    b.input("y", Some(1), d2);
    b.input("rel", Some(1), d2);
    for side in ["src", "tgt"] {
        b.input_dyn(&format!("{side}_more"), Some(1), None);
        b.input_dyn(&format!("{side}_proofs"), None, None);
    }
    graph_io(&mut b, d2);
    support(&mut b, p);
    length_constants(&mut b);
    probe_block(&mut b, p);
    b.block("negation");
    b.scalar("enable_negation", 1.0);
    b.block("class_inclusion");
    b.scalar("enable_class_inclusion", 1.0);

    b.block("probe");
    edges(&mut b, "rel", "neg", "e_neg");
    cleanup_query(&mut b, "x");
    cleanup_query(&mut b, "y");
    // Effective sets: base plus (when enabled) the reached nodes and their proofs.
    for (side, base) in [("src", "x_onehot"), ("tgt", "y_onehot")] {
        b.gate(Mul, &format!("{side}_more"), "enable_class_inclusion", &format!("{side}_extra"));
        b.gate(Add, base, &format!("{side}_extra"), &format!("{side}_set"));
        b.gate(Mul, &format!("{side}_proofs"), "enable_class_inclusion", &format!("{side}_paths"));
    }
    // Slots leaving a source (masked edge matrix) and arriving at a target.
    b.gate(Mul, "e_neg", "src_set", "e_src");
    b.gate(Mul, "obj_t", "tgt_set", "e_tgt");
    b.reduce(ReduceFn::Sum, "e_src", Axis::Rows, "from_col");
    b.reduce(ReduceFn::Sum, "e_tgt", Axis::Rows, "to_col");
    b.gate(Mul, "from_col", "to_col", "cand_col");
    b.op(Instr::Transpose { x: "cand_col".into(), out: "cand_raw".into() });
    b.gate(Mul, "cand_raw", "enable_negation", "cand");
    // Length = source path + 1 + target path; shortest wins.
    b.reduce(ReduceFn::Sum, "src_paths", Axis::Rows, "src_len_col");
    b.reduce(ReduceFn::Sum, "tgt_paths", Axis::Rows, "tgt_len_col");
    b.matmul_ab("e_src", "src_len_col", "slot_src_len");
    b.matmul_ab("e_tgt", "tgt_len_col", "slot_tgt_len");
    b.gate(Add, "slot_src_len", "slot_tgt_len", "slot_len_col");
    b.op(Instr::Transpose { x: "slot_len_col".into(), out: "slot_len".into() });
    b.gate(Sub, "big_minus_one", "slot_len", "score_raw");
    b.gate(Mul, "score_raw", "cand", "score");
    b.op(Instr::Topk { x: "score".into(), k: 1, idx: None, val: Some("best".into()), mask: Some("mask".into()) });
    b.gate(Ge, "best", "one", "found");
    b.gate(Mul, "mask", "found", "slot");
    b.matmul_ab("slot", "e_src", "src_node");
    b.matmul_ab("src_node", "src_paths", "src_path");
    b.matmul_ab("slot", "e_tgt", "tgt_node");
    b.matmul_ab("tgt_node", "tgt_paths", "tgt_path");
    b.gate(Add, "slot", "src_path", "proof_a");
    b.gate(Add, "proof_a", "tgt_path", "proof");
    b.reduce(ReduceFn::Sum, "proof", Axis::All, "len");
    Ok(b.finish(&JOIN_OUTPUTS))
}

/// "can": breadth-first levels over isa with a latched first decision.
pub fn compile_levels(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let d2 = 2 * p.dim;
    let mut b = Builder::new(id);
    b.input("x", Some(1), d2);
    b.input("v", Some(1), d2);
    b.input("rel_can", Some(1), d2);
    b.input("rel_isa", Some(1), d2);
    graph_io(&mut b, d2);
    support(&mut b, p);
    length_constants(&mut b);
    probe_block(&mut b, p);
    b.block("negation");
    b.scalar("enable_negation", 1.0);
    b.block("attribute_inheritance");
    b.scalar("enable_attribute_inheritance", 1.0);

    b.block("probe");
    edges(&mut b, "rel_can", "pos", "e_can_pos");
    edges(&mut b, "rel_can", "neg", "e_can_neg");
    edges(&mut b, "rel_isa", "pos", "e_isa");
    b.op(Instr::Transpose { x: "e_isa".into(), out: "e_isa_t".into() });
    b.op(Instr::Transpose { x: "obj_t".into(), out: "obj_tt".into() });
    cleanup_query(&mut b, "x");
    cleanup_query(&mut b, "v");
    // Slots whose object is v.
    b.matvec("obj_t", "v_onehot", "to_v", false);

    b.block("depth");
    b.unary(Copy, "x_onehot", "s_frontier");
    b.unary(Copy, "x_onehot", "s_reached");
    zeros(&mut b, "s");
    for r in ["done", "pos_found", "neg_found", "level", "lvl"] {
        b.gate(Mul, "zero", "one", r);
    }
    b.gate(Mul, "x_onehot", "zero", "zero_row_n");
    b.matvec("obj_t", "zero_row_n", "zero_row_s", false);
    b.unary(Copy, "zero_row_s", "pos_proof");
    b.unary(Copy, "zero_row_s", "neg_proof");
    let mut body = Vec::new();
    body.push(g(Sub, "one", "done", "open"));
    for (side, e) in [("p", "e_can_pos"), ("n", "e_can_neg")] {
        body.push(Instr::Matmul { a: e.into(), b: "s_frontier_col".into(), out: format!("{side}_from_col"), ab: true });
        body.push(Instr::Transpose { x: format!("{side}_from_col"), out: format!("{side}_from") });
        body.push(g(Mul, &format!("{side}_from"), "to_v", &format!("{side}_cand")));
        // Restrict paths to the frontier so the subject's proof is used.
        body.extend(best_proof(side, &format!("{side}_cand"), e, "s_proofs"));
    }
    body.extend([
        g(Mul, "n_val", "enable_negation", "n_val_on"),
        g(Mul, "p_val", "open", "p_new"),
        g(Mul, "n_val_on", "open", "n_new"),
        g(Add, "pos_found", "p_new", "pos_found"),
        g(Add, "neg_found", "n_new", "neg_found"),
        g(Mul, "p_proof", "p_new", "p_add"),
        g(Mul, "n_proof", "n_new", "n_add"),
        g(Add, "pos_proof", "p_add", "pos_proof"),
        g(Add, "neg_proof", "n_add", "neg_proof"),
        g(Max, "p_new", "n_new", "found_now"),
        g(Mul, "found_now", "lvl", "lvl_hit"),
        g(Add, "level", "lvl_hit", "level"),
        g(Max, "done", "found_now", "done"),
    ]);
    body.extend(expand("s", "e_isa", "e_isa_t", "obj_tt", "enable_attribute_inheritance", None));
    body.extend([Instr::Transpose { x: "s_frontier".into(), out: "s_frontier_col".into() }, g(Add, "lvl", "one", "lvl")]);
    b.op(Instr::Transpose { x: "s_frontier".into(), out: "s_frontier_col".into() });
    b.op(Instr::Step { n: p.max_depth + 1, body });
    Ok(b.finish(&LEVELS_OUTPUTS))
}

/// Properties: backward levels over in-scope rules, latched at the first
/// level where x has the property directly.
pub fn compile_rules(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let d2 = 2 * p.dim;
    let mut b = Builder::new(id);
    b.input("x", Some(1), d2);
    b.input("p", Some(1), d2);
    b.input("rel_prop", Some(1), d2);
    b.input("rel_implies", Some(1), d2);
    b.input("something", Some(1), d2);
    graph_io(&mut b, d2);
    support(&mut b, p);
    length_constants(&mut b);
    probe_block(&mut b, p);
    b.block("conditionals");
    b.scalar("enable_conditionals", 1.0);

    b.block("probe");
    role(&mut b, ROLE_SUBJ, "role_subj");
    role(&mut b, ROLE_OBJ, "role_obj");
    role(&mut b, ROLE_SCOPE, "role_scope");
    cleanup_query(&mut b, "x");
    cleanup_query(&mut b, "p");
    // x's positive property slots (as an S x N edge matrix restricted to x).
    edges(&mut b, "rel_prop", "pos", "e_prop");
    b.gate(Mul, "e_prop", "x_onehot", "e_prop_x");
    // Rules concluding j: rel = implies and object similarity to j.
    b.op(Instr::Bind { a: "rel_implies".into(), b: "role_rel".into(), out: "imp_rb".into(), conj: false });
    b.matvec("slots", "imp_rb", "imp_dot", false);
    b.gate(Mul, "imp_dot", "inv_d", "imp_row");
    b.op(Instr::Transpose { x: "imp_row".into(), out: "imp_col".into() });
    b.op(Instr::Bind { a: "codebook".into(), b: "role_obj".into(), out: "obj_codes".into(), conj: false });
    b.matmul("slots", "obj_codes", "obj_dot");
    b.gate(Mul, "obj_dot", "inv_d", "obj_sim");
    b.gate(Add, "obj_sim", "imp_col", "rule_sim");
    b.gate(Ge, "rule_sim", "need", "rule_hit");
    // Scope: "something" or x itself (cosine >= cleanup_min).
    b.op(Instr::Bind { a: "slots".into(), b: "role_scope".into(), out: "scope_noisy".into(), conj: true });
    b.op(Instr::Norm { kind: NormKind::L2, x: "scope_noisy".into(), eps: "eps".into(), out: "scope_unit".into() });
    b.matvec("scope_unit", "something", "sc_dot", false);
    b.gate(Mul, "sc_dot", "inv_sqrt_d", "sc_cos");
    b.matmul_ab("x_onehot", "codebook", "x_code");
    b.matvec("scope_unit", "x_code", "sx_dot", false);
    b.gate(Mul, "sx_dot", "inv_sqrt_d", "sx_cos");
    b.gate(Ge, "sc_cos", "cleanup_min", "sc_ok");
    b.gate(Ge, "sx_cos", "cleanup_min", "sx_ok");
    b.gate(Max, "sc_ok", "sx_ok", "scope_ok");
    b.gate(Mul, "scope_ok", "pos", "rule_ok");
    b.op(Instr::Transpose { x: "rule_ok".into(), out: "rule_ok_col".into() });
    b.gate(Mul, "rule_hit", "rule_ok_col", "e_rule");
    b.op(Instr::Transpose { x: "e_rule".into(), out: "e_rule_t".into() });
    // A rule's premise property: cleanup of slot (/) subj.
    b.op(Instr::Bind { a: "slots".into(), b: "role_subj".into(), out: "prem_noisy".into(), conj: true });
    b.op(Instr::Norm { kind: NormKind::L2, x: "prem_noisy".into(), eps: "eps".into(), out: "prem_unit".into() });
    b.matmul("prem_unit", "codes_unit", "prem_cos");
    b.op(Instr::Topk { x: "prem_cos".into(), k: 1, idx: None, val: Some("prem_best".into()), mask: Some("prem_mask".into()) });
    b.gate(Ge, "prem_best", "cleanup_min", "prem_ok");
    b.gate(Mul, "prem_mask", "prem_ok", "prem_t");
    b.op(Instr::Transpose { x: "prem_t".into(), out: "prem_tt".into() });

    b.block("depth");
    b.unary(Copy, "p_onehot", "r_frontier");
    b.unary(Copy, "p_onehot", "r_reached");
    zeros(&mut b, "r");
    for r in ["done", "found", "level", "lvl"] {
        b.gate(Mul, "zero", "one", r);
    }
    b.gate(Mul, "x_onehot", "zero", "zero_row_n");
    b.matvec("obj_t", "zero_row_n", "zero_row_s", false);
    b.unary(Copy, "zero_row_s", "proof");
    // x's property slot whose object is in the frontier; the path is the rule
    // chain of that object (premise matrix over properties).
    let mut body = vec![
        g(Sub, "one", "done", "open"),
        Instr::Matvec { w: "obj_t".into(), x: "r_frontier".into(), out: "to_frontier".into(), transpose: false },
        Instr::Reduce { f: ReduceFn::Sum, x: "e_prop_x".into(), axis: Axis::Rows, out: "x_prop_col".into() },
        Instr::Transpose { x: "x_prop_col".into(), out: "x_prop".into() },
        g(Mul, "x_prop", "to_frontier", "d_cand"),
    ];
    body.extend(best_proof("d", "d_cand", "obj_t", "r_proofs"));
    body.extend([
        g(Mul, "d_val", "open", "d_new"),
        g(Add, "found", "d_new", "found"),
        g(Mul, "d_proof", "d_new", "d_add"),
        g(Add, "proof", "d_add", "proof"),
        g(Mul, "d_new", "lvl", "lvl_hit"),
        g(Add, "level", "lvl_hit", "level"),
        g(Max, "done", "d_new", "done"),
    ]);
    body.extend(expand("r", "e_rule_obj", "e_rule_t", "prem_tt", "enable_conditionals", None));
    body.push(g(Add, "lvl", "one", "lvl"));
    // e_rule_obj: the rule matrix as "edges leaving a conclusion" (S x N).
    b.unary(Copy, "e_rule", "e_rule_obj");
    b.op(Instr::Step { n: p.max_depth + 1, body });
    Ok(b.finish(&RULES_OUTPUTS))
}

/// Counts along `more` chains with phasor arithmetic.
pub fn compile_counts(id: &str, c: &CountParams) -> Result<Compiled, IsaError> {
    let p = &c.relational;
    check(id, p)?;
    if c.range == 0 || c.range > 1 << 23 {
        return Err(IsaError(format!("{id}: invalid number range")));
    }
    let d2 = 2 * p.dim;
    let r = c.range as i64;
    let m = (2 * r + 1) as usize;
    let mut b = Builder::new(id);
    b.input("x", Some(1), d2);
    b.input("rel_has", Some(1), d2);
    b.input("rel_more", Some(1), d2);
    graph_io(&mut b, d2);
    b.external("numbers", m, d2);
    support(&mut b, p);
    length_constants(&mut b);
    probe_block(&mut b, p);
    b.block("arithmetic");
    b.scalar("enable_arithmetic", 1.0);
    b.block("readout");
    b.constant("values", m, 1, (-r..=r).map(|n| n as f32).collect());
    b.constant("identity", 1, d2, (0..d2).map(|k| if k < p.dim { 1.0 } else { 0.0 }).collect());

    b.block("probe");
    role(&mut b, ROLE_QUANT, "role_quant");
    cleanup_query(&mut b, "x");
    b.gate(Add, "pos", "neg", "active");
    edges(&mut b, "rel_has", "pos", "e_has");
    edges(&mut b, "rel_more", "active", "e_more_raw");
    b.gate(Mul, "e_more_raw", "enable_arithmetic", "e_more");
    b.op(Instr::Transpose { x: "e_more".into(), out: "e_more_t".into() });
    b.op(Instr::Transpose { x: "obj_t".into(), out: "obj_tt".into() });

    // Breadth-first over more edges, carrying number(sum of differences).
    // Differences are cleaned up first (only rows of more slots are nonzero).
    b.op(Instr::Bind { a: "slots".into(), b: "role_quant".into(), out: "quant".into(), conj: true });
    b.reduce(ReduceFn::Sum, "e_more", Axis::Rows, "more_rows");
    b.gate(Mul, "quant", "more_rows", "quant_more");
    b.matmul("quant_more", "numbers", "delta_scores");
    b.op(Instr::Topk { x: "delta_scores".into(), k: 1, idx: None, val: None, mask: Some("delta_mask".into()) });
    b.gate(Mul, "delta_mask", "more_rows", "delta_sel");
    b.matmul_ab("delta_sel", "numbers", "delta_clean");

    b.block("depth");
    b.unary(Copy, "x_onehot", "c_frontier");
    b.unary(Copy, "x_onehot", "c_reached");
    zeros(&mut b, "c");
    b.op(Instr::Transpose { x: "x_onehot".into(), out: "x_col".into() });
    b.matmul_ab("x_col", "identity", "c_acc");
    b.op(Instr::Step { n: p.max_depth, body: expand("c", "e_more", "e_more_t", "obj_tt", "one", Some("delta_clean")) });

    // Every reached node's positive has-count slots.
    b.block("readout");
    b.gate(Mul, "e_has", "c_reached", "e_cand");
    b.reduce(ReduceFn::Sum, "e_cand", Axis::Rows, "cand_col");
    b.op(Instr::Transpose { x: "cand_col".into(), out: "cand".into() });
    b.gate(Mul, "quant", "cand_col", "quant_cand");
    b.matmul("quant_cand", "numbers", "count_scores");
    b.op(Instr::Topk { x: "count_scores".into(), k: 1, idx: None, val: None, mask: Some("count_mask".into()) });
    b.gate(Mul, "count_mask", "cand_col", "count_sel");
    b.matmul_ab("count_sel", "numbers", "count_clean");
    b.matmul_ab("e_cand", "c_acc", "slot_acc");
    b.op(Instr::Bind { a: "count_clean".into(), b: "slot_acc".into(), out: "total".into(), conj: false });
    b.matmul("total", "numbers", "total_scores");
    b.op(Instr::Topk { x: "total_scores".into(), k: 1, idx: None, val: None, mask: Some("total_mask".into()) });
    b.gate(Mul, "total_mask", "cand_col", "value_sel");
    b.reduce(ReduceFn::Max, "value_sel", Axis::Cols, "present");
    b.reduce(ReduceFn::Sum, "present", Axis::All, "distinct");
    b.gate(Ge, "distinct", "one", "found");
    b.scalar("two", 2.0);
    b.gate(Ge, "distinct", "two", "contradiction");
    b.gate(Sub, "one", "contradiction", "single");
    b.gate(Mul, "found", "single", "one_value");
    b.matmul_ab("present", "values", "value_raw");
    b.gate(Mul, "value_raw", "one_value", "value");
    // Proofs: the shortest candidate, or the union of all of them.
    for i in best_proof("o", "cand", "e_cand", "c_proofs") {
        b.op(i);
    }
    b.matmul_ab("cand", "e_cand", "cand_nodes");
    b.matmul_ab("cand_nodes", "c_proofs", "all_paths");
    b.gate(Add, "all_paths", "cand", "all_raw");
    b.gate(Min, "all_raw", "one", "all_proof");
    b.gate(Mul, "o_proof", "one_value", "one_part");
    b.gate(Mul, "all_proof", "contradiction", "all_part");
    b.gate(Add, "one_part", "all_part", "premises");
    // Arithmetic was used: a path behind the chosen (or any) candidate.
    b.reduce(ReduceFn::Sum, "o_path", Axis::All, "o_path_len");
    b.reduce(ReduceFn::Sum, "all_paths", Axis::All, "all_path_len");
    b.gate(Ge, "o_path_len", "one", "o_arith");
    b.gate(Ge, "all_path_len", "one", "all_arith");
    b.gate(Mul, "o_arith", "one_value", "arith_one");
    b.gate(Mul, "all_arith", "contradiction", "arith_all");
    b.gate(Add, "arith_one", "arith_all", "arith");
    Ok(b.finish(&COUNTS_OUTPUTS))
}

/// Positive and negative findings to a verdict and its premises.
pub fn compile_verdict(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let mut b = Builder::new(id);
    b.input("pos_found", Some(1), 1);
    b.input("neg_found", Some(1), 1);
    b.input_dyn("pos_proof", Some(1), None);
    b.input_dyn("neg_proof", Some(1), None);
    b.block("combine");
    b.scalar("one", 1.0);
    b.constant("yes_sel", 1, 4, onehot(4, 0));
    b.constant("no_sel", 1, 4, onehot(4, 1));
    b.constant("contra_sel", 1, 4, onehot(4, 2));
    b.constant("unknown_sel", 1, 4, onehot(4, 3));
    b.gate(Min, "pos_found", "one", "pf");
    b.gate(Min, "neg_found", "one", "nf");
    b.gate(Sub, "one", "pf", "not_p");
    b.gate(Sub, "one", "nf", "not_n");
    b.gate(Mul, "pf", "not_n", "yes");
    b.gate(Mul, "nf", "not_p", "no");
    b.gate(Mul, "pf", "nf", "contra");
    b.gate(Mul, "not_p", "not_n", "unknown");
    b.gate(Mul, "yes", "yes_sel", "v_yes");
    b.gate(Mul, "no", "no_sel", "v_no");
    b.gate(Mul, "contra", "contra_sel", "v_contra");
    b.gate(Mul, "unknown", "unknown_sel", "v_unknown");
    b.gate(Add, "v_yes", "v_no", "v_a");
    b.gate(Add, "v_a", "v_contra", "v_b");
    b.gate(Add, "v_b", "v_unknown", "verdict");
    b.gate(Mul, "pos_proof", "pf", "pp");
    b.gate(Mul, "neg_proof", "nf", "np");
    b.gate(Add, "pp", "np", "both");
    b.gate(Min, "both", "one", "premises");
    Ok(b.finish(&VERDICT_OUTPUTS))
}
