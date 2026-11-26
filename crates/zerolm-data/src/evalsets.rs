//! Parsers for the evaluation-only sets (BLiMP and LAMBADA) and their
//! contamination checks. These files are never part of exposure.

use crate::contamination::ngram_hashes;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

/// One BLiMP minimal pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlimpPair {
    pub good: String,
    pub bad: String,
    /// Paradigm id (67 of them in BLiMP), for example `adjunct_island`.
    pub uid: String,
    /// Phenomenon area (`syntax`, `morphology`, ...).
    pub field: String,
}

/// Parse BLiMP JSON Lines (the 67 files concatenated, or any one of them).
pub fn parse_blimp(bytes: &[u8]) -> Result<Vec<BlimpPair>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("blimp: {e}"))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("blimp line {}: {e}", i + 1))?;
        let get = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string).ok_or(format!("blimp line {}: missing '{k}'", i + 1));
        let (good, bad) = (get("sentence_good")?, get("sentence_bad")?);
        // Seven pairs in the published data have identical sentences; they are kept (a
        // model cannot win them) so the denominator matches the published one.
        if good.is_empty() || bad.is_empty() {
            return Err(format!("blimp line {}: empty sentence", i + 1));
        }
        out.push(BlimpPair { good, bad, uid: get("UID")?, field: get("field")? });
    }
    if out.is_empty() {
        return Err("blimp: no pairs".into());
    }
    Ok(out)
}

/// One LAMBADA item: the passage up to the final word, and that word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LambadaItem {
    pub context: Vec<u8>,
    pub target: Vec<u8>,
}

/// Undo LAMBADA's word tokenization with fixed string rules (no model): `` and
/// '' become quotes, space before punctuation and before contraction
/// endings is removed. The file is also lower-cased; that cannot be undone.
pub fn detokenize(line: &str) -> String {
    let mut s = line.replace("`` ", "\"").replace(" ''", "\"");
    for (from, to) in [(" n't", "n't"), (" 's", "'s"), (" 're", "'re"), (" 've", "'ve"), (" 'll", "'ll"), (" 'd", "'d"), (" 'm", "'m"), (" ,", ","), (" .", "."), (" ?", "?"), (" !", "!"), (" ;", ";"), (" :", ":")] {
        s = s.replace(from, to);
    }
    s
}

/// Parse the LAMBADA plain-text test file (one passage per line, detokenized). The target
/// is the last whitespace-delimited token with surrounding punctuation
/// removed; the context is everything before it, so a model must produce the
/// word itself, not its punctuation. Lines with no alphanumeric final token
/// are returned in the second value (they are skipped, and counted).
pub fn parse_lambada(bytes: &[u8]) -> Result<(Vec<LambadaItem>, usize), String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("lambada: {e}"))?;
    let (mut items, mut skipped) = (Vec::new(), 0);
    for line in text.lines().map(str::trim_end).filter(|l| !l.is_empty()).map(detokenize) {
        let line = line.as_str();
        let tok_start = line.rfind(char::is_whitespace).map(|i| i + line[i..].chars().next().unwrap().len_utf8()).unwrap_or(0);
        let tok = &line[tok_start..];
        let lead = tok.len() - tok.trim_start_matches(|c: char| !c.is_alphanumeric()).len();
        let word = tok.trim_matches(|c: char| !c.is_alphanumeric());
        if word.is_empty() || tok_start + lead == 0 {
            skipped += 1;
            continue;
        }
        items.push(LambadaItem { context: line[..tok_start + lead].as_bytes().to_vec(), target: word.as_bytes().to_vec() });
    }
    if items.is_empty() {
        return Err("lambada: no items".into());
    }
    Ok((items, skipped))
}

/// Number of `sentences` that occur verbatim (as a whole run of words) in
/// `exposure`. Short items such as BLiMP sentences never reach 13 words, so
/// the long n-gram check alone would be vacuous for them.
pub fn verbatim_overlap(exposure: &[u8], sentences: &[&[u8]]) -> usize {
    let mut by_len: BTreeMap<usize, Vec<&[u8]>> = BTreeMap::new();
    for s in sentences {
        let n = s.split(|b| b.is_ascii_whitespace()).filter(|w| !w.is_empty()).count();
        if n > 0 {
            by_len.entry(n).or_default().push(s);
        }
    }
    let mut shared = 0;
    for (n, group) in by_len {
        let index: HashSet<(u64, u64)> = ngram_hashes(exposure, n);
        shared += group.iter().filter(|s| ngram_hashes(s, n).iter().all(|h| index.contains(h))).count();
    }
    shared
}
