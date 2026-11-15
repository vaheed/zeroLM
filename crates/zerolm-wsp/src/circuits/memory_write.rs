//! The memory store's write decisions as circuits. The host prepares a stamp
//! (sequence number, source, provenance, key word count: bookkeeping it
//! owns), runs a program, and moves the rows the outputs specify; every
//! decision about what the memory becomes is made here.
//!
//! Metadata layout (`memory.semantic.meta`, `META_COLS` columns) and
//! episodic layout (`memory.episodic.meta`, `EP_COLS`) are fixed below and
//! checked against `zerolm_memory` by its tests.
//!
//! - **`memory_write`** (teach, correct, confirm; `op` one-hot). The same-key
//!   slot is the best active key (`keys . key`) with score >= the same-key
//!   threshold; its value row is decoded and compared with the new value.
//!   The free slot is the lowest empty one. Outcome (one-hot, `OUTCOMES`):
//!   teach: unchanged (same value), conflict (different), stored (no key);
//!   correct: unchanged, superseded, stored; confirm: confirmed, unknown.
//!   Outputs the new slot's key, key-word, value, and metadata rows (zero
//!   unless a new slot is written), the old slot's updated metadata row, the
//!   episode metadata row, and the slots involved.
//! - **`memory_forget`**: the same-key slot and every slot it supersedes
//!   (one link per step, `slots` steps, so any chain is followed); outputs
//!   the chain mask, the episodic rows that refer to it (their keys are
//!   erased), the metadata with the chain marked deleted, and the episode row.
//! - **`memory_compact`**: active slots move to the lowest slots in order
//!   (`order`, from top-k with ties to the lowest index); supersession links
//!   and episodic slot references are remapped (-1 when the target is not
//!   active); outputs the new metadata and episodic slot column.
//! - **`memory_history`**: valid episodes whose key matches the query key
//!   (dot product >= the same-key threshold).

use super::value_codec::{codec_io, emit_decode, emit_encode, support, CodecParams};
use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn::*, Instr, IsaError, ReduceFn};

/// Semantic metadata columns.
pub mod field {
    pub const STATUS: usize = 0;
    pub const CONFIDENCE: usize = 1;
    pub const UPDATED_LO: usize = 4;
    pub const UPDATED_HI: usize = 5;
    pub const SUPERSEDES: usize = 10;
    pub const SUPERSEDED_BY: usize = 11;
    pub const CONTRADICTIONS: usize = 12;
}
pub const META_COLS: usize = 14;
/// Episodic metadata columns.
pub mod ep_field {
    pub const OP: usize = 0;
    pub const SLOT: usize = 3;
    pub const VALID: usize = 8;
}
pub const EP_COLS: usize = 9;
/// Status codes.
pub const EMPTY: f32 = 0.0;
pub const ACTIVE: f32 = 1.0;
pub const SUPERSEDED: f32 = 2.0;
pub const DELETED: f32 = 3.0;
/// Episode op codes (index into the memory store's op names).
pub const OP_ADD: f32 = 1.0;
pub const OP_UNCHANGED: f32 = 2.0;
pub const OP_CONTRADICTION: f32 = 3.0;
pub const OP_SUPERSEDE: f32 = 4.0;
pub const OP_CONFIRM: f32 = 5.0;
pub const OP_FORGET: f32 = 6.0;
/// `memory_write` outcome columns.
pub const OUTCOMES: [&str; 6] = ["stored", "unchanged", "conflict", "superseded", "confirmed", "unknown"];

const BIG: f32 = 1.0e6;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WriteParams {
    pub codec: CodecParams,
    pub slots: usize,
    pub key_dim: usize,
    /// Width of a key-word row (max_key_words * key_dim).
    pub key_words: usize,
    pub episodes: usize,
    pub same_key_threshold: f32,
    pub unconfirmed: f32,
    pub confirmed: f32,
}

pub const WRITE_OUTPUTS: [&str; 11] = ["outcome", "best", "slot", "free", "full", "new_key", "new_words", "new_value", "new_meta", "old_meta", "episode"];
pub const FORGET_OUTPUTS: [&str; 6] = ["found", "slot", "chain", "erase", "meta_out", "episode"];
pub const COMPACT_OUTPUTS: [&str; 5] = ["order", "n_active", "freed", "meta_out", "ep_slot"];
pub const HISTORY_OUTPUTS: [&str; 1] = ["hits"];

fn check(id: &str, p: &WriteParams) -> Result<(), IsaError> {
    if p.slots < 2 || p.key_dim == 0 || p.key_words == 0 || p.episodes == 0 {
        return Err(IsaError(format!("{id}: invalid memory write parameters")));
    }
    Ok(())
}

fn onehot(n: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0.0; n];
    v[i] = 1.0;
    v
}

fn eq(b: &mut Builder, a: &str, x: &str, out: &str) {
    b.gate(Ge, a, x, &format!("{out}_lo"));
    b.gate(Ge, x, a, &format!("{out}_hi"));
    b.gate(Mul, &format!("{out}_lo"), &format!("{out}_hi"), out);
}

/// Column `col` of `meta` as a 1 x S row, into `out`.
fn column(b: &mut Builder, m: &str, cols: usize, col: usize, out: &str) {
    let sel = format!("{out}_sel");
    b.constant(&sel, 1, cols, onehot(cols, col));
    b.matvec(m, &sel, out, false);
}

/// The same-key slot: `best_mask` (1 x S), `best` (index), `same` (1 x 1),
/// and `active` (1 x S). Needs `one`, `big`, and `status`.
fn same_key(b: &mut Builder, p: &WriteParams) {
    b.block("same_key");
    b.scalar("threshold", p.same_key_threshold);
    b.scalar("active_code", ACTIVE);
    eq(b, "status", "active_code", "active");
    b.matvec("keys", "key", "raw", false);
    b.gate(Sub, "one", "active", "inactive");
    b.gate(Mul, "inactive", "big", "penalty");
    b.gate(Mul, "raw", "active", "raw_active");
    b.gate(Sub, "raw_active", "penalty", "scores");
    b.op(Instr::Topk { x: "scores".into(), k: 1, idx: Some("best".into()), val: Some("best_val".into()), mask: Some("best_mask".into()) });
    b.reduce(ReduceFn::Sum, "active", Axis::All, "n_active");
    b.gate(Ge, "n_active", "one", "have");
    b.gate(Ge, "best_val", "threshold", "close");
    b.gate(Mul, "close", "have", "same");
}

fn status_row(b: &mut Builder) {
    b.block("layout");
    column(b, "meta", META_COLS, field::STATUS, "status");
}

pub fn compile_write(id: &str, p: &WriteParams) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let c = &p.codec;
    let mut b = Builder::new(id);
    b.input("key", Some(1), p.key_dim);
    b.input("words", Some(1), p.key_words);
    b.input("value", Some(1), c.positions);
    b.input("op", Some(1), 3);
    b.input("stamp", Some(1), META_COLS);
    b.input("ep_stamp", Some(1), EP_COLS);
    b.external("keys", p.slots, p.key_dim);
    b.external("values", p.slots, c.positions * c.code_dim);
    b.external("meta", p.slots, META_COLS);
    codec_io(&mut b, c);
    support(&mut b);
    status_row(&mut b);
    same_key(&mut b, p);

    // The stored value of the same-key slot, decoded and compared.
    b.matmul_ab("best_mask", "values", "best_row");
    emit_decode(&mut b, c, "best_row");
    b.block("compare");
    b.scalar("tolerance", 0.5);
    b.gate(Sub, "bytes", "value", "diff_pos");
    b.gate(Sub, "value", "bytes", "diff_neg");
    b.gate(Max, "diff_pos", "diff_neg", "diff_abs");
    b.reduce(ReduceFn::Max, "diff_abs", Axis::All, "diff_max");
    b.gate(Ge, "diff_max", "tolerance", "differs");
    b.gate(Sub, "one", "differs", "same_value");

    // The lowest empty slot.
    b.block("free");
    b.scalar("empty_code", EMPTY);
    eq(&mut b, "status", "empty_code", "empty");
    b.op(Instr::Topk { x: "empty".into(), k: 1, idx: Some("free".into()), val: Some("has_free".into()), mask: None });
    b.gate(Sub, "one", "has_free", "full");

    // Outcomes.
    b.block("decide");
    for (k, name) in ["teach", "correct", "confirm"].iter().enumerate() {
        let sel = format!("op_{name}_sel");
        b.constant(&sel, 3, 1, onehot(3, k));
        b.matmul_ab("op", &sel, &format!("is_{name}"));
    }
    b.gate(Add, "is_teach", "is_correct", "is_write");
    b.gate(Sub, "one", "same", "absent");
    b.gate(Sub, "one", "same_value", "changed");
    b.gate(Mul, "same", "same_value", "same_both");
    b.gate(Mul, "same", "changed", "same_changed");
    b.gate(Mul, "is_write", "same_both", "o_unchanged");
    b.gate(Mul, "is_teach", "same_changed", "o_conflict");
    b.gate(Mul, "is_correct", "same_changed", "o_superseded");
    b.gate(Mul, "is_write", "absent", "o_stored");
    b.gate(Mul, "is_confirm", "same", "o_confirmed");
    b.gate(Mul, "is_confirm", "absent", "o_unknown");
    let mut acc = String::new();
    for (k, name) in OUTCOMES.iter().enumerate() {
        let sel = format!("out_{name}_sel");
        b.constant(&sel, 1, OUTCOMES.len(), onehot(OUTCOMES.len(), k));
        let term = format!("out_{name}");
        b.gate(Mul, &format!("o_{name}"), &sel, &term);
        if acc.is_empty() {
            acc = term;
        } else {
            let next = format!("outcome_{k}");
            b.gate(Add, &acc, &term, &next);
            acc = next;
        }
    }
    b.unary(Copy, &acc, "outcome");
    b.gate(Add, "o_stored", "o_superseded", "writes_new");

    // The new slot's rows: a fresh record plus the stamp; a superseding
    // record points at the old slot.
    b.block("record");
    let mut fresh = vec![0.0f32; META_COLS];
    fresh[field::STATUS] = ACTIVE;
    fresh[field::CONFIDENCE] = p.unconfirmed;
    fresh[field::SUPERSEDES] = -1.0;
    fresh[field::SUPERSEDED_BY] = -1.0;
    b.constant("fresh", 1, META_COLS, fresh);
    b.constant("e_supersedes", 1, META_COLS, onehot(META_COLS, field::SUPERSEDES));
    b.gate(Add, "fresh", "stamp", "record_base");
    b.gate(Add, "best", "one", "best_plus");
    b.gate(Mul, "best_plus", "o_superseded", "link_shift");
    b.gate(Mul, "e_supersedes", "link_shift", "link_row");
    b.gate(Add, "record_base", "link_row", "record");
    b.gate(Mul, "record", "writes_new", "new_meta");
    b.gate(Mul, "key", "writes_new", "new_key");
    b.gate(Mul, "words", "writes_new", "new_words");
    emit_encode(&mut b, c, "value", "value_row");
    b.block("record");
    b.gate(Mul, "value_row", "writes_new", "new_value");

    // The same-key slot's metadata after the operation.
    b.matmul_ab("best_mask", "meta", "old");
    let sel = |b: &mut Builder, col: usize, name: &str| {
        b.constant(&format!("e_{name}"), 1, META_COLS, onehot(META_COLS, col));
        b.gate(Mul, "old", &format!("e_{name}"), &format!("old_{name}_row"));
        b.reduce(ReduceFn::Sum, &format!("old_{name}_row"), Axis::All, &format!("old_{name}"));
    };
    sel(&mut b, field::STATUS, "status");
    sel(&mut b, field::CONFIDENCE, "confidence");
    sel(&mut b, field::SUPERSEDED_BY, "superseded_by");
    b.constant("e_contradictions", 1, META_COLS, onehot(META_COLS, field::CONTRADICTIONS));
    let mut upd = vec![0.0f32; META_COLS];
    upd[field::UPDATED_LO] = 1.0;
    upd[field::UPDATED_HI] = 1.0;
    b.constant("e_updated", 1, META_COLS, upd);
    b.scalar("superseded_code", SUPERSEDED);
    b.scalar("confirmed_level", p.confirmed);
    // conflict: contradictions + 1
    b.gate(Mul, "e_contradictions", "o_conflict", "d_conflict");
    // superseded: status -> SUPERSEDED, superseded_by -> free
    b.gate(Sub, "superseded_code", "old_status", "status_gap");
    b.gate(Mul, "status_gap", "o_superseded", "status_shift");
    b.gate(Mul, "e_status", "status_shift", "d_status");
    b.gate(Sub, "free", "old_superseded_by", "link_gap");
    b.gate(Mul, "link_gap", "o_superseded", "link_move");
    b.gate(Mul, "e_superseded_by", "link_move", "d_link");
    // confirmed: confidence -> confirmed level
    b.gate(Sub, "confirmed_level", "old_confidence", "conf_gap");
    b.gate(Mul, "conf_gap", "o_confirmed", "conf_shift");
    b.gate(Mul, "e_confidence", "conf_shift", "d_conf");
    // superseded or confirmed: updated time -> stamp
    b.gate(Add, "o_superseded", "o_confirmed", "touched");
    b.gate(Mul, "stamp", "e_updated", "stamp_upd");
    b.gate(Mul, "old", "e_updated", "old_upd");
    b.gate(Sub, "stamp_upd", "old_upd", "upd_gap");
    b.gate(Mul, "upd_gap", "touched", "d_upd");
    b.gate(Add, "old", "d_conflict", "old_a");
    b.gate(Add, "old_a", "d_status", "old_b");
    b.gate(Add, "old_b", "d_link", "old_c");
    b.gate(Add, "old_c", "d_conf", "old_d");
    b.gate(Add, "old_d", "d_upd", "old_meta");

    // The episode row: op code and the slot it concerns.
    b.block("episode");
    b.constant("op_codes", OUTCOMES.len(), 1, vec![OP_ADD, OP_UNCHANGED, OP_CONTRADICTION, OP_SUPERSEDE, OP_CONFIRM, 0.0]);
    b.matmul_ab("outcome", "op_codes", "op_code");
    b.constant("e_op", 1, EP_COLS, onehot(EP_COLS, ep_field::OP));
    b.constant("e_slot", 1, EP_COLS, onehot(EP_COLS, ep_field::SLOT));
    b.gate(Sub, "free", "best", "slot_gap");
    b.gate(Mul, "slot_gap", "writes_new", "slot_shift");
    b.gate(Add, "best", "slot_shift", "slot");
    b.gate(Mul, "e_op", "op_code", "ep_op");
    b.gate(Mul, "e_slot", "slot", "ep_slot");
    b.gate(Add, "ep_stamp", "ep_op", "ep_a");
    b.gate(Add, "ep_a", "ep_slot", "episode");
    Ok(b.finish(&WRITE_OUTPUTS))
}

pub fn compile_forget(id: &str, p: &WriteParams) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let s = p.slots;
    let mut b = Builder::new(id);
    b.input("key", Some(1), p.key_dim);
    b.input("stamp", Some(1), META_COLS);
    b.input("ep_stamp", Some(1), EP_COLS);
    b.external("keys", s, p.key_dim);
    b.external("meta", s, META_COLS);
    b.external("ep_meta", p.episodes, EP_COLS);
    support(&mut b);
    status_row(&mut b);
    same_key(&mut b, p);

    // Follow supersedes links from the same-key slot.
    b.block("chain");
    b.scalar("zero", 0.0);
    b.constant("index", 1, s, (0..s).map(|i| i as f32).collect());
    column(&mut b, "meta", META_COLS, field::SUPERSEDES, "links");
    b.gate(Mul, "best_mask", "same", "chain");
    b.unary(Copy, "chain", "cur");
    let g = |f, a: &str, x: &str, out: &str| Instr::Gate { f, a: a.into(), b: Some(x.into()), out: out.into() };
    b.op(Instr::Step {
        n: s,
        body: vec![
            g(Mul, "cur", "links", "cur_link_row"),
            Instr::Reduce { f: ReduceFn::Sum, x: "cur_link_row".into(), axis: Axis::All, out: "target".into() },
            Instr::Reduce { f: ReduceFn::Sum, x: "cur".into(), axis: Axis::All, out: "moving".into() },
            g(Ge, "target", "zero", "linked"),
            g(Ge, "index", "target", "t_lo"),
            g(Ge, "target", "index", "t_hi"),
            g(Mul, "t_lo", "t_hi", "t_hit"),
            g(Mul, "t_hit", "linked", "t_linked"),
            g(Mul, "t_linked", "moving", "t_live"),
            g(Sub, "one", "chain", "unvisited"),
            g(Mul, "t_live", "unvisited", "cur"),
            g(Max, "chain", "cur", "chain"),
        ],
    });

    // Episodes that refer to a chain slot (valid ones) lose their key.
    b.block("erase");
    column(&mut b, "ep_meta", EP_COLS, ep_field::SLOT, "ep_slot_row");
    column(&mut b, "ep_meta", EP_COLS, ep_field::VALID, "ep_valid");
    b.op(Instr::Transpose { x: "ep_slot_row".into(), out: "ep_slot_col".into() });
    eq(&mut b, "ep_slot_col", "index", "ep_onehot");
    b.matvec("ep_onehot", "chain", "ep_in_chain", false);
    b.gate(Mul, "ep_in_chain", "ep_valid", "erase");

    // Chain slots become deleted, updated at the stamp.
    b.block("mark");
    let mut cols = vec![0.0f32; META_COLS];
    cols[field::STATUS] = 1.0;
    cols[field::UPDATED_LO] = 1.0;
    cols[field::UPDATED_HI] = 1.0;
    b.constant("marked_cols", 1, META_COLS, cols);
    b.constant("deleted_row", 1, META_COLS, onehot(META_COLS, field::STATUS).iter().map(|x| x * DELETED).collect());
    b.gate(Mul, "stamp", "marked_cols", "stamp_cols");
    b.gate(Add, "deleted_row", "stamp_cols", "marked_vals");
    b.op(Instr::Transpose { x: "chain".into(), out: "chain_col".into() });
    b.matmul_ab("chain_col", "marked_cols", "replace");
    b.gate(Sub, "one", "replace", "keep");
    b.gate(Mul, "meta", "keep", "kept");
    b.matmul_ab("chain_col", "marked_vals", "written");
    b.gate(Add, "kept", "written", "meta_out");

    b.block("episode");
    b.scalar("forget_code", OP_FORGET);
    b.constant("e_op", 1, EP_COLS, onehot(EP_COLS, ep_field::OP));
    b.constant("e_slot", 1, EP_COLS, onehot(EP_COLS, ep_field::SLOT));
    b.gate(Mul, "e_op", "forget_code", "ep_op");
    b.gate(Mul, "e_slot", "best", "ep_slot");
    b.gate(Add, "ep_stamp", "ep_op", "ep_a");
    b.gate(Add, "ep_a", "ep_slot", "episode");
    b.unary(Copy, "same", "found");
    b.unary(Copy, "best", "slot");
    Ok(b.finish(&FORGET_OUTPUTS))
}

pub fn compile_compact(id: &str, p: &WriteParams) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let s = p.slots;
    let mut b = Builder::new(id);
    b.external("meta", s, META_COLS);
    b.external("ep_meta", p.episodes, EP_COLS);
    support(&mut b);
    status_row(&mut b);

    b.block("order");
    b.scalar("active_code", ACTIVE);
    b.scalar("empty_code", EMPTY);
    b.scalar("big_score", BIG * 10.0);
    b.constant("index", 1, s, (0..s).map(|i| i as f32).collect());
    b.op(Instr::Transpose { x: "index".into(), out: "index_col".into() });
    eq(&mut b, "status", "active_code", "active");
    eq(&mut b, "status", "empty_code", "empty");
    b.reduce(ReduceFn::Sum, "active", Axis::All, "n_active");
    b.reduce(ReduceFn::Sum, "empty", Axis::All, "n_empty");
    b.scalar("n_slots", s as f32);
    b.gate(Sub, "n_slots", "n_active", "not_active");
    b.gate(Sub, "not_active", "n_empty", "freed");
    // Active slots first, in index order (score big - index), then the rest.
    b.gate(Sub, "big_score", "index", "rank_score");
    b.gate(Mul, "rank_score", "active", "score");
    b.op(Instr::Topk { x: "score".into(), k: s, idx: Some("order".into()), val: None, mask: None });
    // P[j, s] = 1 when new row j holds old slot s (j < n_active).
    b.op(Instr::Transpose { x: "order".into(), out: "order_col".into() });
    eq(&mut b, "order_col", "index", "perm_all");
    b.gate(Add, "index_col", "one", "position");
    b.gate(Ge, "n_active", "position", "keep_col");
    b.gate(Mul, "perm_all", "keep_col", "perm");
    b.matmul_ab("perm", "meta", "moved");
    // map[s]: the new index of an active slot, else -1.
    b.matmul_ab("index", "perm", "rank");
    b.gate(Mul, "rank", "active", "rank_active");
    b.gate(Sub, "active", "one", "minus_inactive");
    b.gate(Add, "rank_active", "minus_inactive", "map");
    b.op(Instr::Transpose { x: "map".into(), out: "map_col".into() });

    // Remap a column of slot references: map[v] for v >= 0, else -1.
    b.block("remap");
    let remap = |b: &mut Builder, col_reg: &str, out: &str| {
        eq(b, col_reg, "index", &format!("{out}_onehot"));
        b.matmul_ab(&format!("{out}_onehot"), "map_col", &format!("{out}_lookup"));
        b.reduce(ReduceFn::Sum, &format!("{out}_onehot"), Axis::Rows, &format!("{out}_any"));
        b.gate(Sub, &format!("{out}_any"), "one", &format!("{out}_none"));
        b.gate(Add, &format!("{out}_lookup"), &format!("{out}_none"), out);
    };
    for (col, name) in [(field::SUPERSEDES, "sup"), (field::SUPERSEDED_BY, "sup_by")] {
        let sel = format!("{name}_sel");
        b.constant(&sel, META_COLS, 1, onehot(META_COLS, col));
        b.matmul_ab("moved", &sel, &format!("{name}_col"));
        remap(&mut b, &format!("{name}_col"), &format!("{name}_new"));
    }
    let mut links = vec![0.0f32; META_COLS];
    links[field::SUPERSEDES] = 1.0;
    links[field::SUPERSEDED_BY] = 1.0;
    b.constant("link_cols", 1, META_COLS, links);
    b.gate(Sub, "one", "link_cols", "other_cols");
    b.gate(Mul, "moved", "other_cols", "moved_rest");
    b.constant("e_sup", 1, META_COLS, onehot(META_COLS, field::SUPERSEDES));
    b.constant("e_sup_by", 1, META_COLS, onehot(META_COLS, field::SUPERSEDED_BY));
    b.matmul_ab("sup_new", "e_sup", "sup_part");
    b.matmul_ab("sup_by_new", "e_sup_by", "sup_by_part");
    b.gate(Add, "moved_rest", "sup_part", "remapped_a");
    b.gate(Add, "remapped_a", "sup_by_part", "remapped");
    // Rows past the active ones are empty records (links -1).
    let mut empty_row = vec![0.0f32; META_COLS];
    empty_row[field::SUPERSEDES] = -1.0;
    empty_row[field::SUPERSEDED_BY] = -1.0;
    b.constant("empty_row", 1, META_COLS, empty_row);
    b.gate(Mul, "remapped", "keep_col", "kept_rows");
    b.gate(Sub, "one", "keep_col", "drop_col");
    b.matmul_ab("drop_col", "empty_row", "empty_rows");
    b.gate(Add, "kept_rows", "empty_rows", "meta_out");
    // Episodic slot references.
    column(&mut b, "ep_meta", EP_COLS, ep_field::SLOT, "ep_slot_row");
    b.op(Instr::Transpose { x: "ep_slot_row".into(), out: "ep_slot_col".into() });
    remap(&mut b, "ep_slot_col", "ep_slot");
    Ok(b.finish(&COMPACT_OUTPUTS))
}

pub fn compile_history(id: &str, p: &WriteParams) -> Result<Compiled, IsaError> {
    check(id, p)?;
    let mut b = Builder::new(id);
    b.input("key", Some(1), p.key_dim);
    b.external("ep_keys", p.episodes, p.key_dim);
    b.external("ep_meta", p.episodes, EP_COLS);
    b.block("match");
    b.scalar("threshold", p.same_key_threshold);
    b.scalar("one", 1.0);
    column(&mut b, "ep_meta", EP_COLS, ep_field::VALID, "valid");
    eq(&mut b, "valid", "one", "is_valid");
    b.matvec("ep_keys", "key", "dot", false);
    b.gate(Ge, "dot", "threshold", "close");
    b.gate(Mul, "close", "is_valid", "hits");
    Ok(b.finish(&HISTORY_OUTPUTS))
}
