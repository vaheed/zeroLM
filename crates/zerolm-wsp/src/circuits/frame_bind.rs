//! Typed frame binding as a circuit: the numeric frame from `template-match`
//! and the atom vectors of the clause's tokens in, the role-filler structure
//! of the utterance out.
//!
//! The frame's field positions are the interface contract with the seeded
//! template table (`FIELDS` in `zerolm-reasoner`): 0 form, 1 predicate,
//! 2 subject index, 3 subject singular, 4 pronoun, 5 object index, 6 object
//! singular, 7 dimension (1-based), 8 number, 9 negated, 11 scope, 12 premise,
//! 13 conclusion, 14 higher, 15 thing index. The circuit knows these as
//! positions only; no words, forms, or relations are named in it.
//!
//! Inputs
//! - `frame` (1 x fields): the selected template's frame.
//! - `words` (2 tokens x 2d): row i is the atom of token i, row tokens + i the
//!   atom of its singular form. The host runs `atom-encode` over the bytes.
//! - `dimensions` (n x 2d): atoms of the seeded dimension names.
//! - `relations` (m x 2d): the relation atoms, in the fixed order of
//!   `zerolm-reasoner::RELATIONS`.
//! - `quant` (1 x 2d): `number-encode` of the frame's number.
//!
//! Outputs
//! - `parts` (5 x 2d) and `mask` (1 x 5): the filler of each role (subject,
//!   relation, object, quantity, scope) and which roles the utterance fills.
//! - `interns` (3 x 2d) and `intern_mask` (1 x 3): the vectors to enter into
//!   the symbol codebook, in order, and which rows are used.
//! - `control` (1 x 8): form, effective predicate (a count question counts as
//!   predicate 7, a which-question as 5), metadata kind, polarity (+1 or
//!   -1), pronoun, higher, number, number of filled roles.
//! - `subject`, `name_a`, `object`, `rel`, `rel_more` (1 x 2d): the subject
//!   atom, the first name of a which-question, the object atom, the
//!   predicate's relation vector (bound with its dimension or counted thing
//!   where the predicate has one), and the relation of "more" bound with the
//!   counted thing.
//!
//! Blocks (units of ablation): `index` (one-hot selectors of token and
//! dimension), `route` (predicate and form selectors), `assemble` (the output
//! stacks).

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{GateFn, Instr, IsaError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub tokens: usize,
    pub fields: usize,
    pub dim: usize,
    pub dimensions: usize,
    pub relations: usize,
}

pub const OUTPUTS: [&str; 10] = ["parts", "mask", "interns", "intern_mask", "control", "subject", "name_a", "object", "rel", "rel_more"];

fn eq(b: &mut Builder, x: &str, value: f32, out: &str) {
    let c = format!("{out}_const");
    b.scalar(&c, value);
    b.gate(GateFn::Ge, x, &c, &format!("{out}_lo"));
    b.gate(GateFn::Ge, &c, x, &format!("{out}_hi"));
    b.gate(GateFn::Mul, &format!("{out}_lo"), &format!("{out}_hi"), out);
}

/// One-hot (1 x n) of scalar `idx` over positions `first..first + n`.
fn onehot(b: &mut Builder, idx: &str, first: usize, n: usize, out: &str) {
    let iota: Vec<f32> = (0..n).map(|i| (first + i) as f32).collect();
    b.constant(&format!("{out}_iota"), 1, n, iota);
    b.gate(GateFn::Ge, &format!("{out}_iota"), idx, &format!("{out}_lo"));
    b.gate(GateFn::Ge, idx, &format!("{out}_iota"), &format!("{out}_hi"));
    b.gate(GateFn::Mul, &format!("{out}_lo"), &format!("{out}_hi"), out);
}

/// Sum of scalar-weighted rows.
fn mix(b: &mut Builder, terms: &[(&str, &str)], out: &str) {
    let mut acc = String::new();
    for (i, (w, row)) in terms.iter().enumerate() {
        let t = format!("{out}_t{i}");
        b.gate(GateFn::Mul, w, row, &t);
        if i == 0 {
            acc = t;
        } else {
            let next = format!("{out}_s{i}");
            b.gate(GateFn::Add, &acc, &t, &next);
            acc = next;
        }
    }
    b.unary(GateFn::Copy, &acc, out);
}

fn add(b: &mut Builder, x: &str, y: &str, out: &str) {
    b.gate(GateFn::Add, x, y, out);
}

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    if p.tokens == 0 || p.tokens > 32 || p.fields < 16 || p.fields > 64 || p.dim == 0 || p.dimensions == 0 || p.relations != 10 {
        return Err(IsaError(format!("{id}: invalid frame_bind parameters")));
    }
    let (t, k, d2) = (p.tokens, p.fields, 2 * p.dim);
    let mut b = Builder::new(id);
    b.input("frame", Some(1), k);
    b.input("words", Some(2 * t), d2);
    b.input("dimensions", Some(p.dimensions), d2);
    b.input("relations", Some(p.relations), d2);
    b.input("quant", Some(1), d2);

    b.block("index");
    b.scalar("one", 1.0);
    b.scalar("tokens", t as f32);
    for (i, name) in [(0, "form"), (1, "pred"), (2, "subj_i"), (3, "subj_s"), (4, "pronoun"), (5, "obj_i"), (6, "obj_s"), (7, "dim_i"), (8, "number"), (9, "negated"), (11, "scope"), (12, "prem_i"), (13, "conc_i"), (14, "higher"), (15, "thing_i")] {
        let mut pick = vec![0.0; k];
        pick[i] = 1.0;
        b.constant(&format!("fpick{i}"), k, 1, pick);
        b.matmul_ab("frame", &format!("fpick{i}"), name);
    }
    // Token atoms: the singular flag moves the index into the second half.
    let word = |b: &mut Builder, idx: &str, singular: Option<&str>, out: &str| {
        let at = match singular {
            Some(s) => {
                b.gate(GateFn::Min, s, "one", &format!("{out}_flag"));
                b.gate(GateFn::Mul, &format!("{out}_flag"), "tokens", &format!("{out}_shift"));
                b.gate(GateFn::Add, idx, &format!("{out}_shift"), &format!("{out}_at"));
                format!("{out}_at")
            }
            None => idx.to_string(),
        };
        onehot(b, &at, 0, 2 * t, &format!("{out}_sel"));
        b.matmul_ab(&format!("{out}_sel"), "words", out);
    };
    word(&mut b, "subj_i", Some("subj_s"), "subj_w");
    word(&mut b, "subj_i", None, "name_a");
    word(&mut b, "obj_i", Some("obj_s"), "obj_w");
    word(&mut b, "prem_i", None, "prem_w");
    word(&mut b, "conc_i", None, "conc_w");
    b.gate(GateFn::Add, "thing_i", "tokens", "thing_at");
    onehot(&mut b, "thing_at", 0, 2 * t, "thing_sel");
    b.matmul_ab("thing_sel", "words", "thing_w");
    onehot(&mut b, "dim_i", 1, p.dimensions, "dim_sel");
    b.matmul_ab("dim_sel", "dimensions", "dim_a");

    b.block("route");
    eq(&mut b, "form", 2.0, "is_rule");
    eq(&mut b, "form", 4.0, "is_count");
    eq(&mut b, "form", 5.0, "is_which");
    b.gate(GateFn::Sub, "one", "is_rule", "not_rule");
    b.scalar("seven", 7.0);
    b.scalar("five", 5.0);
    b.gate(GateFn::Mul, "is_count", "seven", "count_shift");
    b.gate(GateFn::Mul, "is_which", "five", "which_shift");
    b.gate(GateFn::Add, "pred", "count_shift", "pred_a");
    b.gate(GateFn::Add, "pred_a", "which_shift", "pred_eff");
    for n in 1..=8 {
        eq(&mut b, "pred_eff", n as f32, &format!("e{n}"));
    }
    for (j, name) in ["rel_isa", "rel_can", "rel_in", "rel_prop", "rel_gt", "rel_before", "rel_has", "rel_more_base", "rel_implies", "rel_something"].iter().enumerate() {
        let mut pick = vec![0.0; p.relations];
        pick[j] = 1.0;
        b.constant(&format!("rpick{j}"), 1, p.relations, pick);
        b.matmul_ab(&format!("rpick{j}"), "relations", name);
    }
    b.op(Instr::Bind { a: "rel_gt".into(), b: "dim_a".into(), out: "rel_dim".into(), conj: false });
    b.op(Instr::Bind { a: "rel_has".into(), b: "thing_w".into(), out: "rel_count".into(), conj: false });
    b.op(Instr::Bind { a: "rel_more_base".into(), b: "thing_w".into(), out: "rel_more".into(), conj: false });
    mix(&mut b, &[("e1", "rel_isa"), ("e2", "rel_can"), ("e3", "rel_in"), ("e4", "rel_prop"), ("e5", "rel_dim"), ("e6", "rel_before"), ("e7", "rel_count"), ("e8", "rel_more")], "rel");

    b.block("assemble");
    b.gate(GateFn::Sub, "one", "e7", "not_count");
    b.gate(GateFn::Add, "e7", "e8", "has_quant");
    b.gate(GateFn::Sub, "one", "pronoun", "named");
    // Role fillers.
    mix(&mut b, &[("is_rule", "prem_w"), ("not_rule", "subj_w")], "f_subj");
    mix(&mut b, &[("is_rule", "rel_implies"), ("not_rule", "rel")], "f_rel");
    mix(&mut b, &[("is_rule", "conc_w"), ("not_rule", "obj_w")], "f_obj");
    b.gate(GateFn::Sub, "one", "scope", "no_scope");
    mix(&mut b, &[("scope", "name_a"), ("no_scope", "rel_something")], "f_scope");
    let fillers = ["f_subj", "f_rel", "f_obj", "quant", "f_scope"];
    let mut acc = String::new();
    for (j, name) in fillers.iter().enumerate() {
        let mut col = vec![0.0; 5];
        col[j] = 1.0;
        b.constant(&format!("role_col{j}"), 5, 1, col);
        b.matmul_ab(&format!("role_col{j}"), name, &format!("placed{j}"));
        if j == 0 {
            acc = "placed0".into();
        } else {
            let next = format!("parts_acc{j}");
            add(&mut b, &acc, &format!("placed{j}"), &next);
            acc = next;
        }
    }
    b.unary(GateFn::Copy, &acc, "parts");

    // Which roles are filled: subject and relation always, object unless counting,
    // quantity for count and more, scope for rules.
    fn unit(b: &mut Builder, n: usize, at: usize, name: &str) {
        let mut u = vec![0.0; n];
        u[at] = 1.0;
        b.constant(name, 1, n, u);
    }
    for j in 0..5 {
        unit(&mut b, 5, j, &format!("mask_unit{j}"));
    }
    let mut ones = vec![0.0; 5];
    ones[0] = 1.0;
    ones[1] = 1.0;
    b.constant("mask_base", 1, 5, ones);
    mix(&mut b, &[("one", "mask_base"), ("not_count", "mask_unit2"), ("has_quant", "mask_unit3"), ("is_rule", "mask_unit4")], "mask");

    // Codebook entries, in write order.
    b.gate(GateFn::Add, "e1", "e2", "set_a");
    b.gate(GateFn::Add, "set_a", "e3", "set_b");
    b.gate(GateFn::Add, "set_b", "e4", "set_c");
    b.gate(GateFn::Add, "set_c", "e6", "obj_first");
    b.gate(GateFn::Add, "e7", "e8", "thing_first");
    b.gate(GateFn::Add, "e5", "e8", "obj_second");
    mix(&mut b, &[("is_rule", "prem_w"), ("not_rule", "subj_w")], "intern0");
    mix(&mut b, &[("obj_first", "obj_w"), ("e5", "dim_a"), ("thing_first", "thing_w")], "intern1_fact");
    mix(&mut b, &[("is_rule", "conc_w"), ("not_rule", "intern1_fact")], "intern1");
    mix(&mut b, &[("obj_second", "obj_w")], "intern2_fact");
    mix(&mut b, &[("is_rule", "name_a"), ("not_rule", "intern2_fact")], "intern2");
    for j in 0..3 {
        let mut col = vec![0.0; 3];
        col[j] = 1.0;
        b.constant(&format!("intern_col{j}"), 3, 1, col);
        b.matmul_ab(&format!("intern_col{j}"), &format!("intern{j}"), &format!("iplaced{j}"));
    }
    add(&mut b, "iplaced0", "iplaced1", "iacc");
    add(&mut b, "iacc", "iplaced2", "interns");
    for j in 0..3 {
        unit(&mut b, 3, j, &format!("imask_unit{j}"));
    }
    b.gate(GateFn::Mul, "not_rule", "named", "fact_named");
    b.gate(GateFn::Add, "is_rule", "fact_named", "m0");
    b.gate(GateFn::Mul, "is_rule", "scope", "rule_scope");
    b.gate(GateFn::Mul, "not_rule", "obj_second", "fact_second");
    b.gate(GateFn::Add, "rule_scope", "fact_second", "m2");
    mix(&mut b, &[("m0", "imask_unit0"), ("one", "imask_unit1"), ("m2", "imask_unit2")], "intern_mask");

    // Scalars.
    b.scalar("nine", 9.0);
    b.scalar("two", 2.0);
    b.gate(GateFn::Mul, "is_rule", "nine", "kind_rule");
    b.gate(GateFn::Mul, "not_rule", "pred", "kind_fact");
    b.gate(GateFn::Add, "kind_rule", "kind_fact", "kind");
    b.gate(GateFn::Mul, "negated", "two", "neg2");
    b.gate(GateFn::Sub, "one", "neg2", "polarity");
    b.reduce(zerolm_engine::isa::ReduceFn::Sum, "mask", zerolm_engine::isa::Axis::All, "n_roles");
    for (j, name) in ["form", "pred_eff", "kind", "polarity", "pronoun", "higher", "number", "n_roles"].iter().enumerate() {
        unit(&mut b, 8, j, &format!("ctl_unit{j}"));
        b.gate(GateFn::Mul, name, &format!("ctl_unit{j}"), &format!("ctl{j}"));
    }
    let mut acc = "ctl0".to_string();
    for j in 1..8 {
        let next = format!("ctl_acc{j}");
        add(&mut b, &acc, &format!("ctl{j}"), &next);
        acc = next;
    }
    b.unary(GateFn::Copy, &acc, "control");
    b.unary(GateFn::Copy, "subj_w", "subject");
    b.unary(GateFn::Copy, "obj_w", "object");
    Ok(b.finish(&OUTPUTS))
}
