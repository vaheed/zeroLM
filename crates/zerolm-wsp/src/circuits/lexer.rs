//! ASCII word segmentation and case folding. The circuit returns ordered
//! byte spans, so the host only copies bytes; it never classifies them.
//! Chunks carry a single tail-word bit, allowing arbitrarily long inputs and
//! words without truncation. No vocabulary or text appears in the constants.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub chunk_bytes: usize,
}

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    let l = p.chunk_bytes;
    if !(1..=512).contains(&l) {
        return Err(IsaError(format!("{id}: chunk_bytes must be in 1..=512")));
    }
    let mut b = Builder::new(id);
    b.input("text", Some(1), l);
    b.input("tail", Some(1), 1);

    b.block("classify");
    for (n, v) in [("b45", 45.0), ("b46", 46.0), ("b48", 48.0), ("b58", 58.0), ("b65", 65.0), ("b91", 91.0), ("b97", 97.0), ("b123", 123.0)] {
        b.scalar(n, v);
        b.gate(GateFn::Ge, "text", n, &format!("ge_{n}"));
    }
    for (lo, hi, out) in [("b45", "b46", "hyphen"), ("b48", "b58", "digit"), ("b65", "b91", "upper"), ("b97", "b123", "lower")] {
        b.gate(GateFn::Sub, &format!("ge_{lo}"), &format!("ge_{hi}"), out);
    }
    b.gate(GateFn::Add, "upper", "lower", "alpha");
    b.gate(GateFn::Add, "alpha", "digit", "alnum");
    b.gate(GateFn::Add, "alnum", "hyphen", "word");

    b.block("fold");
    b.scalar("case_offset", 32.0);
    b.gate(GateFn::Mul, "upper", "case_offset", "offset");
    b.gate(GateFn::Add, "text", "offset", "folded");

    b.block("segment");
    b.scalar("one", 1.0);
    let mut shift = vec![0.0; l * l];
    for i in 1..l {
        shift[(i - 1) * l + i] = 1.0;
    }
    b.constant("shift", l, l, shift);
    b.matmul_ab("word", "shift", "prev");
    b.matvec("shift", "word", "next", false);
    b.gate(GateFn::Sub, "one", "prev", "not_prev");
    b.gate(GateFn::Sub, "one", "next", "not_next");
    b.gate(GateFn::Mul, "word", "not_prev", "starts");
    b.gate(GateFn::Mul, "word", "not_next", "ends");
    b.reduce(ReduceFn::Sum, "starts", Axis::All, "count");

    // Positive descending scores order all valid spans before the unused
    // top-k entries. Only the first `count` entries are read by the host.
    b.block("order");
    b.constant("scores", 1, l, (0..l).map(|i| (l - i) as f32).collect());
    for (mask, out) in [("starts", "start"), ("ends", "end")] {
        let scores = format!("{out}_scores");
        b.gate(GateFn::Mul, mask, "scores", &scores);
        b.op(Instr::Topk { x: scores, k: l, idx: Some(out.into()), val: None, mask: None });
    }

    b.block("carry");
    let mut first = vec![0.0; l];
    let mut last = vec![0.0; l];
    first[0] = 1.0;
    last[l - 1] = 1.0;
    b.constant("first", l, 1, first);
    b.constant("last", l, 1, last);
    b.matmul_ab("word", "first", "head_word");
    b.matmul_ab("word", "last", "tail_word");
    b.gate(GateFn::Mul, "tail", "head_word", "join");
    Ok(b.finish(&["folded", "start", "end", "count", "join", "tail_word"]))
}
