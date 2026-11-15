//! Memory value codec and the short-lived associative matrix `W_FAST` as
//! circuits. A value is a row of byte codes: position `p` holds row `b_p` of
//! the byte code table (`core.embed`), NUL-padded to `positions`. Every
//! decision below (which code row, which byte a read is nearest to, which
//! key features win, whether two values are the same) is an ISA operation;
//! the host copies bytes into input rows and binds model tensors.
//!
//! Programs:
//! - `value-encode`: value bytes (1 x positions) to the code row
//!   (1 x positions * code_dim): a byte one-hot (compare with 0..code_rows)
//!   times the code table.
//! - `fast-recall`: a pooled key (1 x key_dim) to the bytes recalled from
//!   `W_FAST`: pattern separation s = kWTA(P k) (the `active` largest entries
//!   set to 1/sqrt(active), ties to the lower index), the read M s, and
//!   nearest-code decoding.
//! - `fast-write`: a pooled key and value bytes; the same separation, the
//!   value encoded in-circuit, then two `delta_write`s into the state `fast`:
//!   decay (learning rate 0), then the normalized delta rule on the decayed
//!   matrix: M <- d M; M <- M + lr (v - M s) s^T / (s.s + eps).
//!
//! Nearest-code decoding (shared): per position, the code maximizing
//! 2 x.c - |c|^2 (the nearest code); bytes after the first NUL are zero; the
//! margin is the smallest top-1 minus top-2 score over the positions up to
//! and including the first NUL.
//!
//! Blocks (units of ablation): `onehot` (byte indices), `decode` (score
//! scale, margin selector), `separate` (the
//! kWTA amplitude), `plasticity` (learning rate, decay, epsilon), `support`
//! (shared constants 1 and a large penalty).

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn::*, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecParams {
    /// Value positions (value bytes + 1 for the NUL terminator).
    pub positions: usize,
    /// Rows and width of the byte code table (`core.embed`).
    pub code_rows: usize,
    pub code_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastParams {
    pub codec: CodecParams,
    pub key_dim: usize,
    /// Rows of the pattern separator (`core.fast_separator`).
    pub expansion: usize,
    /// Winners of the kWTA code.
    pub active: usize,
    pub lr: f32,
    pub decay: f32,
    pub eps: f32,
}

pub const ENCODE_OUTPUTS: [&str; 1] = ["row"];
pub const RECALL_OUTPUTS: [&str; 2] = ["bytes", "margin"];
pub const WRITE_OUTPUTS: [&str; 1] = ["sparse"];

/// Penalty for excluded margins (finite).
const BIG: f32 = 1e30;

pub(crate) fn check(id: &str, p: &CodecParams) -> Result<(), IsaError> {
    if p.positions == 0 || p.code_rows < 2 || p.code_rows > 256 || p.code_dim == 0 {
        return Err(IsaError(format!("{id}: invalid value codec parameters")));
    }
    Ok(())
}

fn check_fast(id: &str, p: &FastParams) -> Result<(), IsaError> {
    check(id, &p.codec)?;
    if p.key_dim == 0 || p.active == 0 || p.active > p.expansion || !(p.lr.is_finite() && p.decay.is_finite() && p.eps.is_finite()) {
        return Err(IsaError(format!("{id}: invalid fast-weight parameters")));
    }
    Ok(())
}

pub(crate) fn support(b: &mut Builder) {
    b.block("support");
    b.scalar("one", 1.0);
    b.scalar("big", BIG);
}

/// `bytes_reg` (1 x positions byte values) to the code row `out` (1 x positions*code_dim).
pub(crate) fn emit_encode(b: &mut Builder, p: &CodecParams, bytes_reg: &str, out: &str) {
    b.block("onehot");
    b.constant("byte_index", 1, p.code_rows, (0..p.code_rows).map(|i| i as f32).collect());
    b.op(Instr::Transpose { x: bytes_reg.into(), out: "enc_col".into() });
    b.gate(Ge, "enc_col", "byte_index", "enc_lo");
    b.gate(Ge, "byte_index", "enc_col", "enc_hi");
    b.gate(Mul, "enc_lo", "enc_hi", "enc_onehot");
    b.matmul_ab("enc_onehot", "codes", "enc_rows");
    b.reshape("enc_rows", 1, p.positions * p.code_dim, out);
}

/// The code row `read` (1 x positions*code_dim) to `bytes` (1 x positions)
/// and `margin` (1 x 1). Needs `one` and `big`.
pub(crate) fn emit_decode(b: &mut Builder, p: &CodecParams, read: &str) {
    let pos = p.positions;
    b.block("decode");
    b.scalar("score_scale", 2.0);
    b.constant("margin_sel", 1, 2, vec![1.0, -1.0]);
    b.reshape(read, pos, p.code_dim, "positions");
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
    b.gate(Mul, "bytes_all", "keep", "bytes");
    b.reshape("pos_margin_col", 1, pos, "pos_margin");
    b.gate(Sub, "one", "visited", "unvisited");
    b.gate(Mul, "unvisited", "big", "margin_penalty");
    b.gate(Add, "pos_margin", "margin_penalty", "margin_masked");
    b.reduce(ReduceFn::Min, "margin_masked", Axis::All, "margin");
}

/// The kWTA code `sparse` (1 x expansion) of the input `key`.
fn emit_separate(b: &mut Builder, p: &FastParams) {
    b.block("separate");
    b.scalar("amplitude", 1.0 / (p.active as f32).sqrt());
    b.matvec("separator", "key", "features", false);
    b.op(Instr::Topk { x: "features".into(), k: p.active, idx: None, val: None, mask: Some("winners".into()) });
    b.gate(Mul, "winners", "amplitude", "sparse");
}

pub(crate) fn codec_io(b: &mut Builder, p: &CodecParams) {
    b.external("codes", p.code_rows, p.code_dim);
}

pub fn compile_encode(id: &str, p: &CodecParams) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let mut b = Builder::new(id);
    b.input("value", Some(1), p.positions);
    codec_io(&mut b, p);
    emit_encode(&mut b, p, "value", "row");
    Ok(b.finish(&ENCODE_OUTPUTS))
}

pub fn compile_fast_recall(id: &str, p: &FastParams) -> Result<Compiled, IsaError> {
    check_fast(id, p)?;
    let c = &p.codec;
    let mut b = Builder::new(id);
    b.input("key", Some(1), p.key_dim);
    b.external("separator", p.expansion, p.key_dim);
    b.external("fast", c.positions * c.code_dim, p.expansion);
    codec_io(&mut b, c);
    support(&mut b);
    emit_separate(&mut b, p);
    b.matvec("fast", "sparse", "read", false);
    emit_decode(&mut b, c, "read");
    Ok(b.finish(&RECALL_OUTPUTS))
}

pub fn compile_fast_write(id: &str, p: &FastParams) -> Result<Compiled, IsaError> {
    check_fast(id, p)?;
    let c = &p.codec;
    let mut b = Builder::new(id);
    b.input("key", Some(1), p.key_dim);
    b.input("value", Some(1), c.positions);
    b.external("separator", p.expansion, p.key_dim);
    codec_io(&mut b, c);
    b.state("fast", c.positions * c.code_dim, p.expansion);
    emit_separate(&mut b, p);
    emit_encode(&mut b, c, "value", "target");
    b.block("plasticity");
    b.scalar("lr", p.lr);
    b.scalar("decay", p.decay);
    b.scalar("eps", p.eps);
    b.scalar("no_lr", 0.0);
    b.scalar("no_decay", 1.0);
    b.op(Instr::DeltaWrite { m: "fast".into(), k: "sparse".into(), v: "target".into(), lr: "no_lr".into(), decay: "decay".into(), eps: "eps".into() });
    b.op(Instr::DeltaWrite { m: "fast".into(), k: "sparse".into(), v: "target".into(), lr: "lr".into(), decay: "no_decay".into(), eps: "eps".into() });
    Ok(b.finish(&WRITE_OUTPUTS))
}
