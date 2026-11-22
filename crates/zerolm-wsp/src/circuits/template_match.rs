//! Ordered template selection over external literal, feature, equality, and
//! readout tables. The compiler knows tensor capacities, never grammar rules.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub tokens: usize,
    pub features: usize,
    pub fields: usize,
}

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    if p.tokens == 0 || p.tokens > 32 || p.features < 14 || p.features > 64 || p.fields == 0 || p.fields > 64 {
        return Err(IsaError(format!("{id}: invalid template capacities")));
    }
    let (t, f, k) = (p.tokens, p.features, p.fields);
    let mut b = Builder::new(id);
    b.input("features", Some(t), f);
    b.input("same", Some(t), t);
    b.input("length", Some(1), 1);
    b.input("raw_question", Some(1), 1);
    for name in ["lengths", "questions", "priorities"] { b.external_dyn(name, None, Some(1)); }
    for name in ["literal_ids", "literal_mask"] { b.external_dyn(name, None, Some(t)); }
    for i in 0..t { b.external_dyn(&format!("required{i}"), None, Some(f)); }
    b.external_dyn("equalities", None, Some(t * t));
    b.external_dyn("payload", None, Some(k));
    b.external_dyn("sources", None, Some(k * t * f));

    b.block("shape");
    b.scalar("one", 1.0);
    let mut first = vec![0.0; t]; first[0] = 1.0;
    b.constant("first", 1, t, first);
    let mut lead = vec![0.0; f]; lead[13] = 1.0;
    b.constant("lead_pick", f, 1, lead);
    let mut id_pick = vec![0.0; f]; id_pick[0] = 1.0;
    b.constant("id_pick", f, 1, id_pick);
    b.matmul_ab("first", "features", "first_features");
    b.matmul_ab("first_features", "lead_pick", "leading_question");
    b.gate(GateFn::Max, "raw_question", "leading_question", "question");
    b.gate(GateFn::Ge, "length", "lengths", "len_lo");
    b.gate(GateFn::Ge, "lengths", "length", "len_hi");
    b.gate(GateFn::Mul, "len_lo", "len_hi", "length_ok");
    b.gate(GateFn::Ge, "question", "questions", "q_lo");
    b.gate(GateFn::Ge, "questions", "question", "q_hi");
    b.gate(GateFn::Mul, "q_lo", "q_hi", "question_ok");
    b.gate(GateFn::Mul, "length_ok", "question_ok", "shape_ok");

    b.block("match");
    b.scalar("match_enabled", 1.0);
    b.gate(GateFn::Mul, "shape_ok", "match_enabled", "initial");
    let mut acc = "initial".to_string();
    for i in 0..t {
        let mut pick = vec![0.0; t]; pick[i] = 1.0;
        b.constant(&format!("pick{i}"), 1, t, pick.clone());
        b.constant(&format!("col_pick{i}"), t, 1, pick);
        b.matmul_ab(&format!("pick{i}"), "features", &format!("feat{i}"));
        b.matmul_ab(&format!("feat{i}"), "id_pick", &format!("id{i}"));
        b.matmul_ab("literal_ids", &format!("col_pick{i}"), &format!("expected{i}"));
        b.matmul_ab("literal_mask", &format!("col_pick{i}"), &format!("mask{i}"));
        b.gate(GateFn::Ge, &format!("id{i}"), &format!("expected{i}"), &format!("lo{i}"));
        b.gate(GateFn::Ge, &format!("expected{i}"), &format!("id{i}"), &format!("hi{i}"));
        b.gate(GateFn::Mul, &format!("lo{i}"), &format!("hi{i}"), &format!("eq{i}"));
        b.gate(GateFn::Mul, &format!("eq{i}"), &format!("mask{i}"), &format!("masked{i}"));
        b.gate(GateFn::Sub, "one", &format!("mask{i}"), &format!("ignored{i}"));
        b.gate(GateFn::Add, &format!("masked{i}"), &format!("ignored{i}"), &format!("literal_ok{i}"));
        b.matmul(&format!("required{i}"), &format!("feat{i}"), &format!("feature_score{i}"));
        b.reduce(ReduceFn::Sum, &format!("required{i}"), Axis::Rows, &format!("feature_count{i}"));
        b.gate(GateFn::Ge, &format!("feature_score{i}"), &format!("feature_count{i}"), &format!("features_ok{i}"));
        b.gate(GateFn::Mul, &format!("literal_ok{i}"), &format!("features_ok{i}"), &format!("token_ok{i}"));
        let next = format!("accepted{i}");
        b.gate(GateFn::Mul, &acc, &format!("token_ok{i}"), &next);
        acc = next;
    }
    b.block("equal");
    b.reshape("same", 1, t * t, "same_flat");
    b.matmul("equalities", "same_flat", "equality_score");
    b.reduce(ReduceFn::Sum, "equalities", Axis::Rows, "equality_count");
    b.gate(GateFn::Ge, "equality_score", "equality_count", "equalities_ok");
    b.gate(GateFn::Mul, &acc, "equalities_ok", "accepted");

    b.block("select");
    b.scalar("select_enabled", 1.0);
    b.gate(GateFn::Mul, "accepted", "priorities", "ranked");
    b.op(Instr::Transpose { x: "ranked".into(), out: "scores".into() });
    b.op(Instr::Topk { x: "scores".into(), k: 1, idx: None, val: Some("best".into()), mask: Some("selection".into()) });
    b.gate(GateFn::Ge, "best", "one", "found");
    b.gate(GateFn::Mul, "found", "select_enabled", "valid");
    b.matmul_ab("selection", "payload", "base");
    b.matmul_ab("selection", "sources", "source_flat");
    b.reshape("source_flat", k, t * f, "source_matrix");
    b.reshape("features", t * f, 1, "feature_col");
    b.matmul_ab("source_matrix", "feature_col", "derived_col");
    b.op(Instr::Transpose { x: "derived_col".into(), out: "derived".into() });
    b.gate(GateFn::Add, "base", "derived", "selected_frame");
    b.gate(GateFn::Mul, "selected_frame", "valid", "frame");
    Ok(b.finish(&["frame", "valid", "question"]))
}
