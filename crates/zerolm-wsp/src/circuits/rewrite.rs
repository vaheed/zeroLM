//! One step of a shrinking, left-to-right sequence rewrite. The pattern,
//! replacement, and their masks are external weights, never W_PROG data.
//! A fixed window bounds the program, not the input sequence: the host copies
//! the window at the returned cursor and invokes a fixed input-length budget.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn, IsaError, ReduceFn};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub window: usize,
}

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    let w = p.window;
    if !(1..=32).contains(&w) {
        return Err(IsaError(format!("{id}: rewrite window must be in 1..=32")));
    }
    let mut b = Builder::new(id);
    for name in ["ids", "origins"] {
        b.input(name, Some(1), w);
    }
    for name in ["cursor", "length", "base"] {
        b.input(name, Some(1), 1);
    }
    for name in ["from_ids", "from_mask", "to_ids", "to_mask"] {
        b.external(name, 1, w);
    }
    b.external("at_end", 1, 1);

    b.block("match");
    b.scalar("one", 1.0);
    b.scalar("zero", 0.0);
    b.gate(GateFn::Ge, "ids", "from_ids", "eq_lo");
    b.gate(GateFn::Ge, "from_ids", "ids", "eq_hi");
    b.gate(GateFn::Mul, "eq_lo", "eq_hi", "equal");
    b.gate(GateFn::Sub, "one", "equal", "different");
    b.gate(GateFn::Mul, "different", "from_mask", "masked_different");
    b.reduce(ReduceFn::Sum, "masked_different", Axis::All, "errors");
    b.gate(GateFn::Ge, "zero", "errors", "pattern_match");
    b.gate(GateFn::Add, "cursor", "one", "next_position");
    b.gate(GateFn::Ge, "length", "next_position", "active");
    b.reduce(ReduceFn::Sum, "from_mask", Axis::All, "from_len");
    b.gate(GateFn::Sub, "length", "cursor", "remaining");
    b.gate(GateFn::Ge, "remaining", "from_len", "fits");
    b.gate(GateFn::Mul, "pattern_match", "fits", "fitting_match");

    b.block("end");
    b.scalar("end_enabled", 1.0);
    b.gate(GateFn::Ge, "from_len", "remaining", "end_hi");
    b.gate(GateFn::Mul, "end_hi", "fits", "ends_here");
    b.gate(GateFn::Sub, "one", "ends_here", "not_at_end");
    b.gate(GateFn::Mul, "at_end", "not_at_end", "wrong_end");
    b.gate(GateFn::Sub, "one", "wrong_end", "end_allowed");
    b.gate(GateFn::Mul, "end_allowed", "end_enabled", "end_gate");
    b.gate(GateFn::Mul, "fitting_match", "end_gate", "match_at_end");
    b.gate(GateFn::Mul, "match_at_end", "active", "matched");
    b.gate(GateFn::Sub, "active", "matched", "copy");

    b.block("emit");
    let mut first = vec![0.0; w];
    first[0] = 1.0;
    b.constant("first", 1, w, first);
    b.gate(GateFn::Mul, "copy", "first", "copy_mask");
    b.gate(GateFn::Mul, "copy_mask", "ids", "copied_ids");
    b.gate(GateFn::Mul, "copy_mask", "origins", "copied_origins");
    b.gate(GateFn::Mul, "matched", "to_mask", "replacement_mask");
    b.gate(GateFn::Mul, "replacement_mask", "to_ids", "replacement_ids");
    b.gate(GateFn::Add, "base", "to_ids", "literal_origins");
    b.gate(GateFn::Mul, "replacement_mask", "literal_origins", "replacement_origins");
    b.gate(GateFn::Add, "copied_ids", "replacement_ids", "output_ids");
    b.gate(GateFn::Add, "copied_origins", "replacement_origins", "output_origins");
    b.gate(GateFn::Add, "copy_mask", "replacement_mask", "output_mask");
    b.reduce(ReduceFn::Sum, "output_mask", Axis::All, "count");

    b.block("advance");
    b.scalar("advance_enabled", 1.0);
    b.gate(GateFn::Mul, "matched", "from_len", "matched_len");
    b.gate(GateFn::Add, "copy", "matched_len", "consumed");
    b.gate(GateFn::Mul, "consumed", "advance_enabled", "advance");
    b.gate(GateFn::Add, "cursor", "advance", "next_cursor");
    Ok(b.finish(&["output_ids", "output_origins", "count", "next_cursor", "matched"]))
}
