//! Decimal and suffix attributes from externally classified byte traits.
//! No character codes or language words are compiler constants. Position
//! weights and first/last rows are structural inputs copied by the host.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {}

pub fn compile(id: &str, _: &Params) -> Result<Compiled, IsaError> {
    let mut b = Builder::new(id);
    b.input("traits", None, 4);
    b.input_dyn("places", Some(1), None);
    b.input("first", Some(1), 4);
    b.input("last", Some(1), 4);
    b.input("length", Some(1), 1);
    b.external("limit", 1, 1);
    b.block("select");
    for (i, name) in ["digit", "value", "minus", "suffix"].iter().enumerate() {
        let mut pick = vec![0.0; 4];
        pick[i] = 1.0;
        b.constant(&format!("pick_{name}"), 4, 1, pick);
    }
    b.matmul_ab("traits", "pick_digit", "digit_col");
    b.matmul_ab("traits", "pick_value", "value_col");
    b.matmul_ab("first", "pick_minus", "negative");
    b.matmul_ab("last", "pick_suffix", "ends_s");
    b.reduce(ReduceFn::Sum, "digit_col", Axis::All, "digit_count");
    b.op(Instr::Transpose { x: "value_col".into(), out: "digit_values".into() });

    b.block("decimal");
    b.scalar("one", 1.0);
    b.scalar("zero", 0.0);
    b.gate(GateFn::Mul, "digit_values", "places", "contributions");
    b.reduce(ReduceFn::Sum, "contributions", Axis::All, "raw_value");
    b.gate(GateFn::Add, "limit", "one", "overflow");
    b.gate(GateFn::Min, "raw_value", "overflow", "number");
    b.gate(GateFn::Add, "digit_count", "negative", "recognized_count");
    b.gate(GateFn::Ge, "recognized_count", "length", "all_lo");
    b.gate(GateFn::Ge, "length", "recognized_count", "all_hi");
    b.gate(GateFn::Mul, "all_lo", "all_hi", "all_recognized");
    b.gate(GateFn::Ge, "digit_count", "one", "has_digit");
    b.gate(GateFn::Ge, "limit", "number", "in_range");
    b.gate(GateFn::Ge, "zero", "negative", "positive");
    b.gate(GateFn::Ge, "zero", "number", "is_zero");
    b.gate(GateFn::Max, "positive", "is_zero", "nonnegative");
    b.gate(GateFn::Mul, "all_recognized", "has_digit", "syntax_valid");
    b.gate(GateFn::Mul, "syntax_valid", "in_range", "bounded");
    b.gate(GateFn::Mul, "bounded", "nonnegative", "number_valid");

    b.block("suffix");
    b.scalar("three", 3.0);
    b.gate(GateFn::Ge, "length", "three", "long_enough");
    b.gate(GateFn::Mul, "long_enough", "ends_s", "strip");
    b.gate(GateFn::Sub, "length", "strip", "singular_length");
    Ok(b.finish(&["number", "number_valid", "ends_s", "singular_length"]))
}
