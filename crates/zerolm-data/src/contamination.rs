//! Word n-gram overlap between exposure text and evaluation text.

use std::collections::HashSet;
use zerolm_tensor::fnv1a;

/// Hashes of every `n`-word window of `text` (words split on ASCII
/// whitespace, compared case-sensitively). Two independent 64-bit hashes are
/// kept so a chance collision needs both to agree.
pub fn ngram_hashes(text: &[u8], n: usize) -> HashSet<(u64, u64)> {
    let words: Vec<&[u8]> = text.split(|b| b.is_ascii_whitespace()).filter(|w| !w.is_empty()).collect();
    let mut set = HashSet::new();
    if n == 0 || words.len() < n {
        return set;
    }
    for w in words.windows(n) {
        set.insert(window_hash(w));
    }
    set
}

fn window_hash(words: &[&[u8]]) -> (u64, u64) {
    let mut joined = Vec::new();
    for w in words {
        joined.extend_from_slice(w);
        joined.push(b' ');
    }
    let a = fnv1a(&joined);
    joined.insert(0, 0xA5);
    (a, fnv1a(&joined))
}

/// Result of checking one evaluation text against an exposure index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overlap {
    /// Windows in the evaluation text.
    pub windows: usize,
    /// Of those, windows also present in exposure.
    pub shared: usize,
}

/// Count evaluation windows that also occur in `exposure`.
pub fn overlap(exposure: &HashSet<(u64, u64)>, eval: &[u8], n: usize) -> Overlap {
    let eval_set = ngram_hashes(eval, n);
    Overlap { windows: eval_set.len(), shared: eval_set.iter().filter(|h| exposure.contains(h)).count() }
}
