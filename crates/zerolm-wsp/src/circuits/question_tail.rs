//! Last non-covered byte selection. Coverage lengths and the marker byte are
//! external prior data; the compiler contains no whitespace or punctuation.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {}

pub fn compile(id: &str, _: &Params) -> Result<Compiled, IsaError> {
    let mut b = Builder::new(id);
    b.input("bytes", None, 1);
    b.input("spans", None, 3);
    b.input_dyn("positions", Some(1), None);
    b.external("marker", 1, 1);
    b.block("coverage");
    b.scalar("one", 1.0);
    for i in 0..3 {
        let mut pick = vec![0.0; 3]; pick[i] = 1.0;
        b.constant(&format!("pick{i}"), 3, 1, pick);
        b.scalar(&format!("offset{i}"), (i + 1) as f32);
        b.matmul_ab("spans", &format!("pick{i}"), &format!("span{i}"));
        b.gate(GateFn::Ge, &format!("span{i}"), &format!("offset{i}"), &format!("covered{i}"));
    }
    b.gate(GateFn::Max, "covered0", "covered1", "covered01");
    b.gate(GateFn::Max, "covered01", "covered2", "covered");
    b.gate(GateFn::Sub, "one", "covered", "significant");
    b.op(Instr::Transpose { x: "significant".into(), out: "significant_row".into() });
    b.block("select");
    b.gate(GateFn::Mul, "significant_row", "positions", "scores");
    b.op(Instr::Topk { x: "scores".into(), k: 1, idx: None, val: Some("score".into()), mask: Some("mask".into()) });
    b.op(Instr::Transpose { x: "bytes".into(), out: "byte_row".into() });
    b.gate(GateFn::Mul, "byte_row", "mask", "selected");
    b.reduce(ReduceFn::Sum, "selected", Axis::All, "last");
    b.gate(GateFn::Ge, "score", "one", "has_byte");
    b.gate(GateFn::Ge, "last", "marker", "eq_lo");
    b.gate(GateFn::Ge, "marker", "last", "eq_hi");
    b.gate(GateFn::Mul, "eq_lo", "eq_hi", "equal");
    b.gate(GateFn::Mul, "equal", "has_byte", "question");
    Ok(b.finish(&["question"]))
}
