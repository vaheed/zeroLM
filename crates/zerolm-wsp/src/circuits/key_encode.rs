//! The memory key encoder as a circuit: raw text bytes in, a pooled unit key
//! and per-word unit vectors out. It computes exactly the bag of hashed byte
//! trigrams of `#word#` that `docs/memory-format.md` specifies, with every
//! step an ISA operation on tensors.
//!
//! Input `text` is 1 x L byte values (exact integers): the text occupies
//! positions 1..=len and every other position is 0, so that position 0 and
//! position len + 1 act as separators. The caller must keep len <= L - 2.
//!
//! The derivation that makes word splitting a fixed-shape computation: map
//! every separator byte to `#` and fold ASCII upper case to lower case. Then
//! the trigrams of `#word#` for every word are exactly the length-3 windows
//! whose middle byte is a word byte. Window i is (m[i-1], m[i], m[i+1]).
//! Word ids come from a prefix sum over word starts; a one-hot word-assignment
//! matrix sums each word's window rows; L2 norms and a column sum give the
//! word vectors and the pooled key.
//!
//! The only constants are byte-class bounds of the ASCII control protocol
//! (digits, letters, the upper-case offset, the `#` pad byte) and structural
//! index matrices; no vocabulary or corpus statistic.
//!
//! Blocks (units of ablation):
//! - `classify`: byte-class bounds (word bytes: ASCII letters and digits, and
//!   every byte >= 0x80).
//! - `fold`: the upper-case offset and the separator pad byte.
//! - `window`: the window gather matrix and the previous-byte selector.
//! - `segment`: the prefix-sum matrix and the word ranks.
//! - `support`: the constant 1 and the norm epsilon (0: a zero row stays zero).

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn::*, Instr, IsaError, NormKind, ReduceFn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    /// Input width L in bytes (the text plus two separator positions).
    pub text_bytes: usize,
    /// Rows of the hashed trigram table (`core.memory_encoder`).
    pub buckets: usize,
    pub key_dim: usize,
}

impl Params {
    /// Word rows in the output: at most one word per two input positions.
    pub fn max_words(&self) -> usize {
        self.text_bytes / 2
    }
}

/// Outputs, in order. `words` is max_words x key_dim with words in text order
/// and zero rows after the last word; `word_mask` (1 x max_words) is 1 for a
/// present word; `n_words` is 1 x 1.
pub const OUTPUTS: [&str; 4] = ["key", "words", "word_mask", "n_words"];

/// The `#` byte that pads every word (`docs/memory-format.md`).
pub const PAD: u8 = b'#';

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    if p.text_bytes < 4 || p.buckets == 0 || p.key_dim == 0 {
        return Err(IsaError(format!("{id}: invalid key_encode parameters")));
    }
    let (l, d, w) = (p.text_bytes, p.key_dim, p.max_words());
    let mut b = Builder::new(id);
    b.input("text", Some(1), l);
    b.external("table", p.buckets, d);

    b.block("support");
    b.scalar("one", 1.0);
    b.scalar("eps", 0.0);

    // Word bytes: [48, 58) digits, [65, 91) upper, [97, 123) lower, >= 128.
    b.block("classify");
    for (n, v) in [("b48", 48.0), ("b58", 58.0), ("b65", 65.0), ("b91", 91.0), ("b97", 97.0), ("b123", 123.0), ("b128", 128.0)] {
        b.scalar(n, v);
        b.gate(Ge, "text", n, &format!("ge_{n}"));
    }
    b.gate(Sub, "ge_b48", "ge_b58", "digit");
    b.gate(Sub, "ge_b65", "ge_b91", "upper");
    b.gate(Sub, "ge_b97", "ge_b123", "lower");
    b.gate(Add, "digit", "upper", "alnum_du");
    b.gate(Add, "alnum_du", "lower", "alnum");
    b.gate(Add, "alnum", "ge_b128", "word");

    // m = word ? fold(byte) : '#'.
    b.block("fold");
    b.scalar("case_offset", 32.0);
    b.scalar("pad", PAD as f32);
    b.gate(Mul, "upper", "case_offset", "shift");
    b.gate(Add, "text", "shift", "folded");
    b.gate(Sub, "folded", "pad", "rel");
    b.gate(Mul, "rel", "word", "rel_word");
    b.gate(Add, "rel_word", "pad", "mapped");

    // Windows: gather[j][3i + k] = 1 where j = i - 1 + k (in range).
    b.block("window");
    let mut gather = vec![0.0f32; l * 3 * l];
    for i in 0..l {
        for k in 0..3 {
            if let Some(j) = (i + k).checked_sub(1).filter(|&j| j < l) {
                gather[j * 3 * l + 3 * i + k] = 1.0;
            }
        }
    }
    b.constant("gather", l, 3 * l, gather);
    b.constant("prev_sel", 1, 3, vec![1.0, 0.0, 0.0]);
    b.matvec("gather", "mapped", "win_flat", true);
    b.reshape("win_flat", l, 3, "windows");
    b.op(Instr::HashRows { bytes: "windows".into(), table: "table".into(), n: 3, out: "rows".into() });
    // Previous position's word flag (0 before position 0).
    b.matvec("gather", "word", "word_win_flat", true);
    b.reshape("word_win_flat", l, 3, "word_win");
    b.matmul("word_win", "prev_sel", "prev_col");
    b.reshape("prev_col", 1, l, "prev");

    // Word ids: prefix sum of word starts; one-hot assignment by rank.
    b.block("segment");
    let mut prefix = vec![0.0f32; l * l];
    for r in 0..l {
        for c in r..l {
            prefix[r * l + c] = 1.0;
        }
    }
    b.constant("prefix", l, l, prefix);
    b.constant("rank", w, 1, (1..=w).map(|k| k as f32).collect());
    b.constant("rank_next", w, 1, (2..=w + 1).map(|k| k as f32).collect());
    b.gate(Sub, "one", "prev", "not_prev");
    b.gate(Mul, "word", "not_prev", "start");
    b.matvec("prefix", "start", "wid", true);
    b.gate(Ge, "wid", "rank", "at_least");
    b.gate(Ge, "wid", "rank_next", "beyond");
    b.gate(Sub, "at_least", "beyond", "is_rank");
    b.gate(Mul, "is_rank", "word", "assign");
    b.matmul_ab("assign", "rows", "word_sums");
    b.reduce(ReduceFn::Sum, "start", Axis::All, "n_words");
    b.reshape("rank", 1, w, "rank_row");
    b.gate(Ge, "n_words", "rank_row", "word_mask");

    // Unit word vectors and the pooled unit key.
    b.op(Instr::Norm { kind: NormKind::L2, x: "word_sums".into(), eps: "eps".into(), out: "words".into() });
    b.reduce(ReduceFn::Sum, "words", Axis::Cols, "pooled");
    b.op(Instr::Norm { kind: NormKind::L2, x: "pooled".into(), eps: "eps".into(), out: "key".into() });

    let mut c = b.finish(&OUTPUTS);
    c.tensors.sort_by(|a, b| a.block.cmp(&b.block));
    Ok(c)
}
