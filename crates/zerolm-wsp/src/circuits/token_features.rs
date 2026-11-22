//! Combine external lexical attributes and numeric circuit outputs into the
//! boolean features consumed by an external template table.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{GateFn, IsaError};

pub const WIDTH: usize = 20;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {}

pub fn compile(id: &str, _: &Params) -> Result<Compiled, IsaError> {
    let mut b = Builder::new(id);
    b.input("lexical", Some(1), WIDTH);
    b.input("singular", Some(1), WIDTH);
    b.input("numeric", Some(1), 4);
    b.block("select");
    for (i, name) in [(0, "id"), (1, "reserved"), (2, "pronoun"), (3, "word_number"), (4, "word_value"), (5, "comparative"), (6, "dimension"), (7, "greater"), (8, "article"), (9, "quantifier"), (10, "leading_question")] {
        let mut pick = vec![0.0; WIDTH]; pick[i] = 1.0;
        b.constant(&format!("lex_pick{i}"), WIDTH, 1, pick);
        b.matmul_ab("lexical", &format!("lex_pick{i}"), name);
    }
    b.matmul_ab("singular", "lex_pick1", "singular_reserved");
    for (i, name) in ["decimal_valid", "decimal_value", "ends_s", "singular_length"].iter().enumerate() {
        let mut pick = vec![0.0; 4]; pick[i] = 1.0;
        b.constant(&format!("num_pick{i}"), 4, 1, pick);
        b.matmul_ab("numeric", &format!("num_pick{i}"), name);
    }
    b.block("features");
    b.scalar("one", 1.0);
    b.gate(GateFn::Sub, "one", "reserved", "name");
    b.gate(GateFn::Sub, "one", "singular_reserved", "singular_name");
    b.gate(GateFn::Max, "name", "pronoun", "term");
    b.gate(GateFn::Max, "singular_name", "pronoun", "plural_subject");
    b.gate(GateFn::Max, "decimal_valid", "word_number", "number_valid");
    b.gate(GateFn::Sub, "one", "decimal_valid", "not_decimal");
    b.gate(GateFn::Mul, "word_number", "not_decimal", "use_word");
    b.gate(GateFn::Mul, "word_value", "use_word", "selected_word_value");
    b.gate(GateFn::Mul, "decimal_value", "decimal_valid", "selected_decimal_value");
    b.gate(GateFn::Add, "selected_word_value", "selected_decimal_value", "number");
    let names = ["id", "name", "term", "pronoun", "singular_name", "ends_s", "number_valid", "number", "comparative", "dimension", "greater", "article", "quantifier", "leading_question", "singular_length", "one", "plural_subject"];
    let mut acc = String::new();
    for (i, name) in names.iter().enumerate() {
        let mut place = vec![0.0; WIDTH]; place[i] = 1.0;
        b.constant(&format!("place{i}"), 1, WIDTH, place);
        b.gate(GateFn::Mul, name, &format!("place{i}"), &format!("field{i}"));
        let next = format!("sum{i}");
        if i == 0 { b.unary(GateFn::Copy, &format!("field{i}"), &next); }
        else { b.gate(GateFn::Add, &acc, &format!("field{i}"), &next); }
        acc = next;
    }
    b.unary(GateFn::Copy, &acc, "features");
    Ok(b.finish(&["features"]))
}
