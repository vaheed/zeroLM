//! Exact row lookup against external model weights. Keys and returned values
//! are never compiled constants. Duplicate matching rows fail closed.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub key_cols: usize,
    pub value_cols: usize,
}

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    if !(1..=256).contains(&p.key_cols) || !(1..=4096).contains(&p.value_cols) {
        return Err(IsaError(format!("{id}: invalid row lookup dimensions")));
    }
    let mut b = Builder::new(id);
    b.input("keys", None, p.key_cols);
    b.external_dyn("table_keys", None, Some(p.key_cols));
    b.external_dyn("table_values", None, Some(p.value_cols));
    b.block("match");
    b.scalar("enabled", 1.0);
    let mut acc = "enabled".to_string();
    for i in 0..p.key_cols {
        let mut pick = vec![0.0; p.key_cols];
        pick[i] = 1.0;
        b.constant(&format!("pick{i}"), p.key_cols, 1, pick);
        b.matmul_ab("keys", &format!("pick{i}"), &format!("key{i}"));
        b.matmul_ab("table_keys", &format!("pick{i}"), &format!("table_col{i}"));
        b.op(Instr::Transpose { x: format!("table_col{i}"), out: format!("table_row{i}") });
        b.gate(GateFn::Ge, &format!("key{i}"), &format!("table_row{i}"), &format!("lo{i}"));
        b.gate(GateFn::Ge, &format!("table_row{i}"), &format!("key{i}"), &format!("hi{i}"));
        b.gate(GateFn::Mul, &format!("lo{i}"), &format!("hi{i}"), &format!("eq{i}"));
        let next = format!("match{i}");
        b.gate(GateFn::Mul, &acc, &format!("eq{i}"), &next);
        acc = next;
    }
    b.reduce(ReduceFn::Sum, &acc, Axis::Rows, "matches");
    b.block("unique");
    b.scalar("one", 1.0);
    b.gate(GateFn::Ge, "matches", "one", "at_least_one");
    b.gate(GateFn::Ge, "one", "matches", "at_most_one");
    b.gate(GateFn::Mul, "at_least_one", "at_most_one", "unique");
    b.block("read");
    b.scalar("read_enabled", 1.0);
    b.matmul_ab(&acc, "table_values", "selected");
    b.gate(GateFn::Mul, "selected", "unique", "checked");
    b.gate(GateFn::Mul, "checked", "read_enabled", "values");
    Ok(b.finish(&["values", "matches", "unique"]))
}
