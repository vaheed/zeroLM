//! Memory retrieval as a circuit: active-slot selection, attention read,
//! late-interaction coverage verification, nearest-code value cleanup, and
//! the accept gate. Every decision is an ISA operation on tensors; the host
//! only binds the memory tensors and the query vectors.
//!
//! Blocks (units of ablation):
//! - `select`: status-column selector and the active-status window.
//! - `attend`: inverse temperature and the minimum read weight.
//! - `verify`: read and coverage thresholds and the key-word position index.
//! - `decode`: the cleanup score scale and the top-2 margin selector.
//! - `support`: shared constants (1, 2, a large penalty).

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn::*, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub slots: usize,
    pub key_dim: usize,
    pub max_key_words: usize,
    /// Value positions (value_bytes + 1, for the NUL terminator).
    pub positions: usize,
    /// Rows and width of the byte code table (`core.embed`).
    pub code_rows: usize,
    pub code_dim: usize,
    pub meta_cols: usize,
    /// Metadata column holding the slot status, and the status value of an active slot.
    pub status_col: usize,
    pub active_status: f32,
    /// Metadata column holding the number of stored key words.
    pub key_words_col: usize,
    pub beta: f32,
    pub read_threshold: f32,
    pub coverage_threshold: f32,
    /// Read weights below this are dropped from the value read.
    pub min_weight: f32,
}

/// Penalty added to excluded scores. Large but finite, so `beta * -BIG` stays finite.
pub const BIG: f32 = 1e30;

/// Outputs, in order.
pub const OUTPUTS: [&str; 11] = ["value", "decoded", "best", "have", "score", "runner_up", "weight", "coverage", "margin", "accept", "best_meta"];

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    if p.slots < 2 || p.positions == 0 || p.code_rows < 2 || p.status_col >= p.meta_cols || p.key_words_col >= p.meta_cols {
        return Err(IsaError(format!("{id}: invalid attention_read parameters")));
    }
    let (s, d, w, pos, e) = (p.slots, p.key_dim, p.max_key_words, p.positions, p.code_dim);
    let mut b = Builder::new(id);
    b.input("q", Some(1), d);
    b.input("qw", None, d);
    b.external("keys", s, d);
    b.external("key_words", s, w * d);
    b.external("values", s, pos * e);
    b.external("meta", s, p.meta_cols);
    b.external("codes", p.code_rows, e);

    b.block("support");
    b.scalar("one", 1.0);
    b.scalar("two", 2.0);
    b.scalar("big", BIG);

    // Active slots: status in [active - 0.5, active + 0.5).
    b.block("select");
    let mut sel = vec![0.0; p.meta_cols];
    sel[p.status_col] = 1.0;
    b.constant("status_sel", 1, p.meta_cols, sel);
    b.scalar("active_lo", p.active_status - 0.5);
    b.scalar("active_hi", p.active_status + 0.5);
    b.matvec("meta", "status_sel", "status", false);
    b.gate(Ge, "status", "active_lo", "ge_lo");
    b.gate(Ge, "status", "active_hi", "ge_hi");
    b.gate(Sub, "ge_lo", "ge_hi", "active");
    b.reduce(ReduceFn::Sum, "active", Axis::All, "n_active");
    b.gate(Ge, "n_active", "one", "have");
    b.gate(Ge, "n_active", "two", "have2");

    // Attention over active slots.
    b.block("attend");
    b.scalar("beta", p.beta);
    b.scalar("min_weight", p.min_weight);
    b.matvec("keys", "q", "raw", false);
    b.gate(Sub, "one", "active", "inactive");
    b.gate(Mul, "inactive", "big", "penalty");
    b.gate(Mul, "raw", "active", "raw_active");
    b.gate(Sub, "raw_active", "penalty", "scores");
    b.op(Instr::Topk { x: "scores".into(), k: 1, idx: Some("best".into()), val: Some("best_val".into()), mask: Some("best_mask".into()) });
    b.op(Instr::Topk { x: "scores".into(), k: 2, idx: None, val: Some("top2".into()), mask: None });
    b.gate(Mul, "scores", "beta", "logits");
    b.op(Instr::Softmax { x: "logits".into(), out: "attn".into() });
    b.gate(Ge, "attn", "min_weight", "attn_keep");
    b.gate(Mul, "attn", "attn_keep", "attn_read");
    b.matvec("values", "attn_read", "read", true);
    b.gate(Mul, "attn", "best_mask", "attn_best");
    b.reduce(ReduceFn::Sum, "attn_best", Axis::All, "weight_raw");
    b.gate(Mul, "weight_raw", "have", "weight");
    // score = best (or -1 with no active slot); runner_up likewise needs two.
    b.gate(Sub, "have", "one", "have_m1");
    b.gate(Mul, "best_val", "have", "score_h");
    b.gate(Add, "score_h", "have_m1", "score");
    b.reduce(ReduceFn::Min, "top2", Axis::All, "second");
    b.gate(Sub, "have2", "one", "have2_m1");
    b.gate(Mul, "second", "have2", "second_h");
    b.gate(Add, "second_h", "have2_m1", "runner_up");
    b.matvec("meta", "best_mask", "best_meta_raw", true);
    b.gate(Mul, "best_meta_raw", "have", "best_meta");

    // Coverage: min over the best slot's stored key words of the best cosine
    // with any query word.
    b.block("verify");
    b.scalar("read_threshold", p.read_threshold);
    b.scalar("coverage_threshold", p.coverage_threshold);
    let mut kw_sel = vec![0.0; p.meta_cols];
    kw_sel[p.key_words_col] = 1.0;
    b.constant("key_words_sel", 1, p.meta_cols, kw_sel);
    b.constant("word_index", w, 1, (0..w).map(|i| i as f32).collect());
    b.matvec("key_words", "best_mask", "kw_row", true);
    b.reshape("kw_row", w, d, "kw");
    b.matmul("kw", "qw", "kw_sims");
    b.reduce(ReduceFn::Max, "kw_sims", Axis::Rows, "kw_best");
    b.gate(Mul, "best_meta", "key_words_sel", "n_words_row");
    b.reduce(ReduceFn::Sum, "n_words_row", Axis::All, "n_words");
    b.gate(Ge, "word_index", "n_words", "unused_word");
    b.gate(Mul, "unused_word", "big", "word_penalty");
    b.gate(Add, "kw_best", "word_penalty", "kw_masked");
    b.reduce(ReduceFn::Min, "kw_masked", Axis::All, "kw_min");
    b.gate(Ge, "n_words", "one", "has_words");
    b.gate(Mul, "kw_min", "has_words", "coverage_raw");
    b.gate(Mul, "coverage_raw", "have", "coverage");

    // Cleanup: per position, the code maximizing 2 x.c - |c|^2 (nearest code).
    b.block("decode");
    b.scalar("score_scale", 2.0);
    b.constant("margin_sel", 1, 2, vec![1.0, -1.0]);
    b.reshape("read", pos, e, "positions");
    b.matmul("positions", "codes", "code_dots");
    b.gate(Mul, "codes", "codes", "code_sq");
    b.reduce(ReduceFn::Sum, "code_sq", Axis::Rows, "code_norm_col");
    b.reshape("code_norm_col", 1, p.code_rows, "code_norm");
    b.gate(Mul, "code_dots", "score_scale", "code_dots2");
    b.gate(Sub, "code_dots2", "code_norm", "code_scores");
    b.op(Instr::Topk { x: "code_scores".into(), k: 1, idx: Some("byte_col".into()), val: None, mask: None });
    b.op(Instr::Topk { x: "code_scores".into(), k: 2, idx: None, val: Some("code_top2".into()), mask: None });
    b.matmul("code_top2", "margin_sel", "pos_margin_col");
    b.gate(Ge, "byte_col", "one", "nonnull_col");
    b.reshape("nonnull_col", 1, pos, "nonnull");
    b.op(Instr::Scan { x: "nonnull".into(), out: "keep".into(), exclusive: false });
    b.op(Instr::Scan { x: "nonnull".into(), out: "visited".into(), exclusive: true });
    b.reshape("byte_col", 1, pos, "bytes_all");
    b.gate(Mul, "bytes_all", "keep", "bytes_kept");
    b.gate(Mul, "bytes_kept", "have", "decoded");
    b.reshape("pos_margin_col", 1, pos, "pos_margin");
    b.gate(Sub, "one", "visited", "unvisited");
    b.gate(Mul, "unvisited", "big", "margin_penalty");
    b.gate(Add, "pos_margin", "margin_penalty", "margin_masked");
    b.reduce(ReduceFn::Min, "margin_masked", Axis::All, "margin_raw");
    b.gate(Mul, "margin_raw", "have", "margin");
    b.reduce(ReduceFn::Max, "keep", Axis::All, "nonempty");

    // Accept: score and coverage thresholds, a non-empty value, an active slot.
    b.block("verify");
    b.gate(Ge, "score", "read_threshold", "ok_score");
    b.gate(Ge, "coverage", "coverage_threshold", "ok_coverage");
    b.gate(Mul, "ok_score", "ok_coverage", "ok_both");
    b.gate(Mul, "ok_both", "nonempty", "ok_value");
    b.gate(Mul, "ok_value", "have", "accept");
    b.gate(Mul, "decoded", "accept", "value");

    let mut c = b.finish(&OUTPUTS);
    // Keep constants grouped by block in a stable order for hashing and ablation.
    c.tensors.sort_by(|a, b| a.block.cmp(&b.block));
    Ok(c)
}
