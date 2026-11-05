//! ZeroLM runtime: safe-Rust port of `main.c`.
//!
//! Weight file format and observable behaviour are kept identical to the C
//! reference so `scripts/evaluate.py` and `tests/*.tsv` work unchanged.

pub mod context;
pub mod http;
pub mod jsonl;
pub mod live;
pub mod lock;
pub mod memory;
pub mod serve;

use std::collections::HashMap;
use std::fmt;
use std::fs;

pub const MAX_INTENTS: usize = 32;
pub const MAX_FEATURES: usize = 512;
pub const MAX_STATES: usize = 2048;
pub const MAX_TEXT: usize = 4096;
pub const MAX_TOKEN: usize = 128;
pub const MAX_STEPS: usize = 256;
pub const MAX_VOCAB: usize = 1024;
pub const ABSTAIN_THRESHOLD: f64 = 0.75;
/// `sizeof(Model)` of the C reference build, reported by `--info` for parity.
pub const C_MODEL_STRUCT_BYTES: usize = 690_072;

#[derive(Debug)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

fn err<T>(msg: String) -> Result<T, ModelError> {
    Err(ModelError(msg))
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub intent: usize,
    pub weight: f64,
    /// Normalized phrase bytes (as written; used for display).
    pub phrase: Vec<u8>,
    /// The phrase after `stem_text`; this is what is matched against the prepared input.
    pub key: Vec<u8>,
}

impl Feature {
    pub fn new(intent: usize, weight: f64, phrase: Vec<u8>) -> Feature {
        let key = stem_text(&phrase);
        Feature { intent, weight, phrase, key }
    }
}

/// Longest word kept in the spell-snap word list, and the list's capacity (same in the C runtime).
pub const MAX_SNAP_WORD: usize = 31;
pub const MAX_SNAP_WORDS: usize = 2048;
/// Words shorter than this are never spell-snapped (short words are too ambiguous).
pub const MIN_SNAP_LEN: usize = 6;

fn is_lower_alpha(w: &[u8]) -> bool {
    !w.is_empty() && w.iter().all(|c| c.is_ascii_lowercase())
}

fn undouble(s: &mut Vec<u8>) {
    let n = s.len();
    if n >= 2 && s[n - 1] == s[n - 2] && !matches!(s[n - 1], b'l' | b's' | b'z') {
        s.pop();
    }
}

/// Light suffix stemmer for one lowercase ASCII word (anything else is returned unchanged).
/// Plurals (`ies`, `es`, `s`), `ing`, `ed` and a final `e` are removed, so that
/// "tests"/"test", "recalls"/"recall" and "licensed"/"license" share a key. The same function
/// is applied to phrases and to input, so exact phrases keep matching.
pub fn stem_word(w: &[u8]) -> Vec<u8> {
    let mut s = w.to_vec();
    if !is_lower_alpha(w) {
        return s;
    }
    let ends = |s: &Vec<u8>, suf: &[u8]| s.ends_with(suf);
    if s.len() > 4 && ends(&s, b"ies") {
        s.truncate(s.len() - 3);
        s.push(b'y');
    } else if s.len() > 3
        && ends(&s, b"es")
        && (matches!(s[s.len() - 3], b's' | b'x' | b'z') || ends(&s, b"ches") || ends(&s, b"shes"))
    {
        s.truncate(s.len() - 2);
    } else if s.len() > 3 && ends(&s, b"s") && !ends(&s, b"ss") && !ends(&s, b"us") && !ends(&s, b"is") {
        s.pop();
    }
    if s.len() > 5 && ends(&s, b"ing") {
        s.truncate(s.len() - 3);
        undouble(&mut s);
    } else if s.len() > 4 && ends(&s, b"ed") {
        s.truncate(s.len() - 2);
        undouble(&mut s);
    }
    if s.len() > 4 && ends(&s, b"e") {
        s.pop();
    }
    s
}

/// Apply `stem_word` to every space-separated word of an already normalized text.
pub fn stem_text(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for (i, w) in text.split(|&c| c == b' ').enumerate() {
        if i > 0 {
            out.push(b' ');
        }
        out.extend_from_slice(&stem_word(w));
    }
    out
}

/// True if `a` and `b` differ by at most one substitution, insertion, deletion or swap of two
/// adjacent letters (identical words count).
pub fn within_one_edit(a: &[u8], b: &[u8]) -> bool {
    let (la, lb) = (a.len(), b.len());
    if la == lb {
        let i = a.iter().zip(b).take_while(|(x, y)| x == y).count();
        if i == la {
            return true;
        }
        return a[i + 1..] == b[i + 1..] || (i + 1 < la && a[i] == b[i + 1] && a[i + 1] == b[i] && a[i + 2..] == b[i + 2..]);
    }
    let (long, short) = if la > lb { (a, b) } else { (b, a) };
    if long.len() != short.len() + 1 {
        return false;
    }
    let i = long.iter().zip(short).take_while(|(x, y)| x == y).count();
    long[i + 1..] == short[i..]
}

#[derive(Debug, Clone)]
pub struct State {
    /// Successor state, or `None` for the C file's `-1`.
    pub next: Option<usize>,
    pub token: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub intents: usize,
    pub features: Vec<Feature>,
    pub states: Vec<State>,
    pub start: Vec<usize>,
    pub intent_name: Vec<String>,
    pub vocab: Vec<String>,
    pub state_vocab: Vec<usize>,
    /// Stemmed words of the base phrases (at most `MAX_SNAP_WORDS`), used for spell-snapping.
    pub snap_words: Vec<Vec<u8>>,
}

fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

fn parse_int(s: Option<&str>) -> Option<i64> {
    s?.parse().ok()
}

impl Model {
    pub fn load(path: &str) -> Result<Model, ModelError> {
        let text = fs::read(path)
            .map_err(|_| ModelError(format!("cannot open model file: {path}")))?;
        Model::parse(&String::from_utf8_lossy(&text))
    }

    pub fn parse(text: &str) -> Result<Model, ModelError> {
        let mut header_seen = false;
        let mut intents = 0usize;
        let mut state_count = 0usize;
        let mut features: Vec<Feature> = Vec::new();
        let mut start: Vec<Option<usize>> = vec![None; MAX_INTENTS];
        let mut names: Vec<Option<String>> = vec![None; MAX_INTENTS];
        let mut states: Vec<Option<State>> = vec![None; MAX_STATES];

        for (idx, raw) in text.lines().enumerate() {
            let n = idx + 1;
            let line = raw.trim_end_matches(['\r', '\n']);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if !header_seen {
                let mut it = line.split_whitespace();
                if it.next() != Some("ZEROLM") || parse_int(it.next()) != Some(1) {
                    return err(format!("invalid model header at line {n}"));
                }
                header_seen = true;
                continue;
            }
            if let Some(rest) = line.strip_prefix("META ") {
                let mut it = rest.split_whitespace();
                match (parse_int(it.next()), parse_int(it.next())) {
                    (Some(i), Some(s))
                        if (1..=MAX_INTENTS as i64).contains(&i)
                            && (1..=MAX_STATES as i64).contains(&s) =>
                    {
                        intents = i as usize;
                        state_count = s as usize;
                    }
                    _ => return err(format!("invalid META at line {n}")),
                }
            } else if let Some(rest) = line.strip_prefix("INTENT ") {
                let mut it = rest.split_whitespace();
                match (parse_int(it.next()), it.next()) {
                    (Some(id), Some(name)) if (0..MAX_INTENTS as i64).contains(&id) => {
                        names[id as usize] = Some(truncate_bytes(name, 47));
                    }
                    _ => return err(format!("invalid INTENT at line {n}")),
                }
            } else if let Some(rest) = line.strip_prefix("FEATURE ") {
                let rest = rest.trim_start();
                let (a, rest) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                let rest = rest.trim_start();
                let (b, phrase) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                let phrase = phrase.trim_start();
                let ok = features.len() < MAX_FEATURES && !phrase.is_empty();
                match (a.parse::<i64>(), b.parse::<f64>()) {
                    (Ok(id), Ok(w)) if ok && (0..MAX_INTENTS as i64).contains(&id) => {
                        features.push(Feature::new(
                            id as usize,
                            w,
                            truncate_bytes(phrase, 191).into_bytes(),
                        ));
                    }
                    _ => return err(format!("invalid FEATURE at line {n}")),
                }
            } else if let Some(rest) = line.strip_prefix("START ") {
                let mut it = rest.split_whitespace();
                match (parse_int(it.next()), parse_int(it.next())) {
                    (Some(i), Some(s))
                        if (0..MAX_INTENTS as i64).contains(&i)
                            && (0..MAX_STATES as i64).contains(&s) =>
                    {
                        start[i as usize] = Some(s as usize);
                    }
                    _ => return err(format!("invalid START at line {n}")),
                }
            } else if let Some(rest) = line.strip_prefix("NODE ") {
                let mut it = rest.split_whitespace();
                match (parse_int(it.next()), parse_int(it.next()), it.next()) {
                    (Some(id), Some(next), Some(tok))
                        if (0..MAX_STATES as i64).contains(&id)
                            && (-1..MAX_STATES as i64).contains(&next) =>
                    {
                        states[id as usize] = Some(State {
                            next: if next < 0 { None } else { Some(next as usize) },
                            token: truncate_bytes(tok, MAX_TOKEN - 1),
                        });
                    }
                    _ => return err(format!("invalid NODE at line {n}")),
                }
            } else {
                return err(format!("unknown record at line {n}"));
            }
        }

        if !header_seen || intents < 1 || state_count < 1 || features.is_empty() {
            return err("model is incomplete (missing header, META, or features)".into());
        }
        let mut start_v = Vec::with_capacity(intents);
        let mut name_v = Vec::with_capacity(intents);
        for i in 0..intents {
            match (start[i], names[i].take()) {
                (Some(s), Some(nm)) => {
                    start_v.push(s);
                    name_v.push(nm);
                }
                _ => return err(format!("model missing START or INTENT for id {i}")),
            }
        }
        let mut state_v = Vec::with_capacity(state_count);
        for (i, slot) in states.into_iter().take(state_count).enumerate() {
            match slot {
                Some(s) => state_v.push(s),
                None => return err(format!("model missing NODE {i}")),
            }
        }
        // Dangling references would be silent bugs; the C code would read garbage.
        if start_v.iter().any(|&s| s >= state_count)
            || state_v.iter().any(|s| s.next.is_some_and(|n| n >= state_count))
            || features.iter().any(|f| f.intent >= intents)
        {
            return err("model references an out-of-range state or intent".into());
        }

        // Vocabulary: first-seen order, O(S) via HashMap (C version was O(S*V)).
        let mut vocab: Vec<String> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut state_vocab = Vec::with_capacity(state_v.len());
        for s in &state_v {
            let id = match index.get(&s.token) {
                Some(&id) => id,
                None => {
                    if vocab.len() >= MAX_VOCAB {
                        return err(format!("vocabulary exceeds maximum size {MAX_VOCAB}"));
                    }
                    vocab.push(s.token.clone());
                    index.insert(s.token.clone(), vocab.len() - 1);
                    vocab.len() - 1
                }
            };
            state_vocab.push(id);
        }

        // Phrases of one intent that share a stemmed key are the same evidence: only the heaviest
        // keeps a key (ties: the first); the others stay in the file but never match.
        for i in 0..features.len() {
            for j in 0..i {
                if features[j].intent == features[i].intent && !features[j].key.is_empty() && features[j].key == features[i].key {
                    if features[i].weight > features[j].weight {
                        features[j].key.clear();
                    } else {
                        features[i].key.clear();
                    }
                    break;
                }
            }
        }

        let mut snap_words: Vec<Vec<u8>> = Vec::new();
        for f in &features {
            for w in f.key.split(|&c| c == b' ') {
                if is_lower_alpha(w)
                    && w.len() <= MAX_SNAP_WORD
                    && snap_words.len() < MAX_SNAP_WORDS
                    && !snap_words.iter().any(|x| x == w)
                {
                    snap_words.push(w.to_vec());
                }
            }
        }

        Ok(Model {
            snap_words,
            intents,
            features,
            states: state_v,
            start: start_v,
            intent_name: name_v,
            vocab,
            state_vocab,
        })
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Normalized input made ready for matching: every word is stemmed, and a word of at least
    /// `MIN_SNAP_LEN` letters that no phrase uses is replaced by the one phrase word that is a
    /// single edit away and starts with the same letter (if there is exactly one).
    pub fn prepare(&self, normalized: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(normalized.len() + 8);
        for (i, w) in normalized.split(|&c| c == b' ').enumerate() {
            if i > 0 {
                out.push(b' ');
            }
            let s = stem_word(w);
            if s.len() >= MIN_SNAP_LEN && is_lower_alpha(&s) && !self.snap_words.contains(&s) {
                let mut hit: Option<&Vec<u8>> = None;
                let mut count = 0;
                for cand in &self.snap_words {
                    if cand[0] == s[0] && within_one_edit(&s, cand) {
                        count += 1;
                        hit = Some(cand);
                    }
                }
                if let (1, Some(c)) = (count, hit) {
                    out.extend_from_slice(c);
                    continue;
                }
            }
            out.extend_from_slice(&s);
        }
        out
    }

    /// Result of `classify`: (selected intent, raw logit score). Abstention
    /// (score below threshold) is applied by the caller via `decide`.
    pub fn classify(&self, normalized: &[u8]) -> (usize, f64) {
        self.classify_with(normalized, &[])
    }

    /// Like `classify`, with extra (learned) features added on top of the base
    /// weights. Extra features must reference valid intent ids.
    pub fn classify_with(&self, normalized: &[u8], extra: &[Feature]) -> (usize, f64) {
        let text = self.prepare(normalized);
        let mut logits = vec![0.0f64; self.intents];
        for f in self.features.iter().chain(extra.iter()) {
            if contains_phrase(&text, &f.key) {
                logits[f.intent] += f.weight;
            }
        }
        let best = argmax(&logits);
        (best, logits[best])
    }

    /// Distinct phrases (base + extra) that fired for `intent` on this input.
    pub fn matched_phrases(&self, normalized: &[u8], intent: usize, extra: &[Feature]) -> Vec<String> {
        let text = self.prepare(normalized);
        let mut out: Vec<String> = Vec::new();
        for f in self.features.iter().chain(extra.iter()) {
            if f.intent == intent && contains_phrase(&text, &f.key) {
                let p = String::from_utf8_lossy(&f.phrase).into_owned();
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out
    }

    pub fn intent_index(&self, name: &str) -> Option<usize> {
        self.intent_name.iter().position(|n| n == name)
    }

    /// Classify and apply the abstain rule (last intent is `unknown`).
    pub fn decide(&self, raw: &str) -> (usize, f64) {
        self.decide_with(raw, &[])
    }

    pub fn decide_with(&self, raw: &str, extra: &[Feature]) -> (usize, f64) {
        let norm = normalize_text(raw.as_bytes(), MAX_TEXT);
        let (intent, score) = self.classify_with(&norm, extra);
        if score < ABSTAIN_THRESHOLD {
            (self.intents - 1, score)
        } else {
            (intent, score)
        }
    }

    /// Full classification for routing: the decision (after abstention), the raw best
    /// intent, and the `top` highest-scoring intents with the phrases that fired.
    pub fn rank(&self, raw: &str, extra: &[Feature], top: usize) -> Ranking {
        let norm = normalize_text(raw.as_bytes(), MAX_TEXT);
        let text = self.prepare(&norm);
        let mut logits = vec![0.0f64; self.intents];
        for f in self.features.iter().chain(extra.iter()) {
            if contains_phrase(&text, &f.key) {
                logits[f.intent] += f.weight;
            }
        }
        let best = argmax(&logits);
        let abstained = logits[best] < ABSTAIN_THRESHOLD;
        let intent = if abstained { self.intents - 1 } else { best };
        let mut order: Vec<usize> = (0..self.intents - 1).collect(); // `unknown` is never a candidate
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let candidates = order
            .into_iter()
            .take(top)
            .filter(|&i| logits[i] > 0.0)
            .map(|i| Candidate { intent: i, score: logits[i], matched: self.matched_phrases(&norm, i, extra) })
            .collect();
        Ranking { intent, score: logits[best], abstained, candidates }
    }

    /// Greedy decode by following successor pointers. The C softmax/recurrent
    /// path on a one-hot state reduces to exactly this walk.
    pub fn generate(&self, intent: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut used = 0usize;
        let mut cur = Some(self.start[intent]);
        for _ in 0..MAX_STEPS {
            let Some(s) = cur else { break };
            let tok = &self.states[s].token;
            if tok == "<eos>" {
                break;
            }
            let needed = tok.len() + usize::from(!out.is_empty());
            if used + needed + 1 >= MAX_TEXT {
                break;
            }
            used += needed;
            out.push(tok.clone());
            cur = self.states[s].next;
        }
        out
    }

    /// Faithful port of the C softmax decoder, for parity testing (`--faithful`).
    pub fn generate_faithful(&self, intent: usize) -> Vec<String> {
        let n = self.states.len();
        let mut h = vec![-8.0f64; n];
        h[self.start[intent]] = 8.0;
        softmax(&mut h);
        let mut out = Vec::new();
        let mut used = 0usize;
        for _ in 0..MAX_STEPS {
            let mut probs = vec![0.0f64; self.vocab.len()];
            for (s, &v) in self.state_vocab.iter().enumerate() {
                probs[v] += h[s];
            }
            probs.iter_mut().for_each(|p| *p *= 16.0);
            softmax(&mut probs);
            let tok = &self.vocab[argmax(&probs)];
            if tok == "<eos>" {
                break;
            }
            let needed = tok.len() + usize::from(!out.is_empty());
            if used + needed + 1 >= MAX_TEXT {
                break;
            }
            used += needed;
            out.push(tok.clone());
            let mut next_h = vec![0.0f64; n];
            for (src, st) in self.states.iter().enumerate() {
                if let Some(t) = st.next {
                    next_h[t] += 16.0 * h[src];
                }
            }
            softmax(&mut next_h);
            h = next_h;
        }
        out
    }
}

/// One scored intent in a `Ranking`.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub intent: usize,
    pub score: f64,
    pub matched: Vec<String>,
}

/// Result of `Model::rank`. `intent` is the decision (the `unknown` intent when the
/// best score is below `ABSTAIN_THRESHOLD`); `score` is the best raw score.
#[derive(Debug, Clone)]
pub struct Ranking {
    pub intent: usize,
    pub score: f64,
    pub abstained: bool,
    pub candidates: Vec<Candidate>,
}

impl Ranking {
    /// One JSON object: decision, score, abstained flag and ranked candidates.
    pub fn to_json(&self, model: &Model) -> String {
        use crate::jsonl::escape;
        let cands: Vec<String> = self
            .candidates
            .iter()
            .map(|c| {
                let m: Vec<String> = c.matched.iter().map(|p| format!("\"{}\"", escape(p))).collect();
                format!(
                    "{{\"intent\":\"{}\",\"score\":{:.2},\"matched\":[{}]}}",
                    escape(&model.intent_name[c.intent]),
                    c.score,
                    m.join(",")
                )
            })
            .collect();
        format!(
            "{{\"intent\":\"{}\",\"score\":{:.2},\"abstained\":{},\"candidates\":[{}]}}",
            escape(&model.intent_name[self.intent]),
            self.score,
            self.abstained,
            cands.join(",")
        )
    }
}

pub fn softmax(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum <= 0.0 {
        let u = 1.0 / v.len() as f64;
        v.iter_mut().for_each(|x| *x = u);
        return;
    }
    v.iter_mut().for_each(|x| *x /= sum);
}

/// First index of the maximum (matches the C tie-break).
pub fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn is_word_byte(c: u8) -> bool {
    c >= 0x80 || c.is_ascii_alphanumeric() || c == b'_' || c == b'-'
}

/// ASCII case folding, punctuation/whitespace to single spaces. Non-ASCII
/// bytes are preserved. Output is capped like the C buffer (`cap - 2` bytes).
pub fn normalize_text(src: &[u8], cap: usize) -> Vec<u8> {
    let mut dst: Vec<u8> = Vec::with_capacity(src.len());
    let mut prev_space = true;
    for &c in src {
        if dst.len() + 2 >= cap {
            break;
        }
        if c < 0x80 {
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' {
                dst.push(c.to_ascii_lowercase());
                prev_space = false;
            } else if c.is_ascii_whitespace() || c == 0x0b || c.is_ascii_punctuation() {
                if !prev_space {
                    dst.push(b' ');
                }
                prev_space = true;
            } else {
                dst.push(c);
                prev_space = false;
            }
        } else {
            dst.push(c);
            prev_space = false;
        }
    }
    while dst.last() == Some(&b' ') {
        dst.pop();
    }
    dst
}

/// Whole-word substring match.
pub fn contains_phrase(text: &[u8], phrase: &[u8]) -> bool {
    let n = phrase.len();
    if n == 0 || n > text.len() {
        return false;
    }
    for i in 0..=(text.len() - n) {
        if &text[i..i + n] != phrase {
            continue;
        }
        let left_ok = i == 0 || !is_word_byte(text[i - 1]);
        let right_ok = i + n >= text.len() || !is_word_byte(text[i + n]);
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "ZEROLM 1\nMETA 2 4\nINTENT 0 greeting\nINTENT 1 unknown\n\
FEATURE 0 8.0 hello\nSTART 0 0\nSTART 1 2\n\
NODE 0 1 hi\nNODE 1 -1 <eos>\nNODE 2 3 sorry\nNODE 3 -1 <eos>\n";

    #[test]
    fn stemming_unifies_word_forms_and_leaves_other_text_alone() {
        let st = |w: &str| String::from_utf8(stem_word(w.as_bytes())).unwrap();
        for (a, b) in [("tests", "test"), ("queries", "query"), ("boxes", "box"), ("licensed", "license"),
                       ("running", "run"), ("trained", "train"), ("recalling", "recall"), ("recalls", "recall")] {
            assert_eq!(st(a), st(b), "{a} vs {b}");
        }
        for same in ["class", "bus", "this", "gas", "fine-tune", "42", "café", "it"] {
            assert_eq!(st(same), same, "{same} must not change");
        }
        assert_eq!(String::from_utf8(stem_text(b"do tests pass")).unwrap(), "do test pass");
        // Known limit: a final "e" is only removed from words longer than four letters.
        assert_ne!(st("tuned"), st("tune"));
    }

    #[test]
    fn phrases_that_share_a_stem_count_once_with_the_heaviest_weight() {
        let text = "ZEROLM 1\nMETA 2 4\nINTENT 0 testing\nINTENT 1 unknown\n\
FEATURE 0 2.0 test\nFEATURE 0 2.5 tests\nFEATURE 0 1.0 testing\nFEATURE 0 3.0 benchmark\nSTART 0 0\nSTART 1 2\n\
NODE 0 1 hi\nNODE 1 -1 <eos>\nNODE 2 3 sorry\nNODE 3 -1 <eos>\n";
        let m = Model::parse(text).unwrap();
        let r = m.rank("do the tests and the benchmark pass", &[], 1);
        assert_eq!(r.score, 5.5, "2.5 (heaviest of test/tests/testing) + 3.0, not 8.5");
        assert_eq!(m.rank("testing", &[], 1).score, 2.5);
    }

    #[test]
    fn edit_distance_one_covers_swap_drop_insert_substitute() {
        assert!(within_one_edit(b"licence", b"licnece")); // swap
        assert!(within_one_edit(b"license", b"licens")); // drop
        assert!(within_one_edit(b"licens", b"license")); // insert
        assert!(within_one_edit(b"license", b"licenxe")); // substitute
        assert!(!within_one_edit(b"license", b"licenxx"));
        assert!(!within_one_edit(b"license", b"lisence plate"));
        assert!(!within_one_edit(b"abc", b"abcde"));
    }

    #[test]
    fn spell_snap_fixes_one_typo_but_not_short_or_ambiguous_words() {
        let mut m = Model::parse(TINY).unwrap();
        m.snap_words = vec![b"hello".to_vec(), b"licens".to_vec(), b"licenc".to_vec(), b"benchmark".to_vec()];
        let p = |t: &str| String::from_utf8(m.prepare(t.as_bytes())).unwrap();
        assert_eq!(p("benchmrk results"), "benchmark result");
        assert_eq!(p("benhcmark"), "benchmark"); // swap
        assert_eq!(p("hellp"), "hellp", "words under {MIN_SNAP_LEN} letters are never snapped");
        assert_eq!(p("licenxe"), "licenx", "two candidates one edit away: ambiguous, only stemmed");
        assert_eq!(p("xenchmark"), "xenchmark", "the first letter must match");
        assert_eq!(p("license"), "licens", "known words are only stemmed");
    }

    #[test]
    fn rank_orders_candidates_and_abstains() {
        let m = Model::parse(TINY).unwrap();
        let hit = m.rank("hello there", &[], 3);
        assert!(!hit.abstained);
        assert_eq!(m.intent_name[hit.intent], "greeting");
        assert_eq!(hit.candidates[0].intent, hit.intent);
        assert!(!hit.candidates[0].matched.is_empty());
        let miss = m.rank("qwerty", &[], 3);
        assert!(miss.abstained && miss.candidates.is_empty());
        assert_eq!(m.intent_name[miss.intent], "unknown");
        assert!(hit.to_json(&m).starts_with("{\"intent\":\"greeting\""));
    }

    #[test]
    fn classifies_and_abstains() {
        let m = Model::parse(TINY).unwrap();
        assert_eq!(m.decide("Hello!").0, 0);
        assert_eq!(m.decide("weather").0, 1);
    }

    #[test]
    fn whole_word_only() {
        assert!(contains_phrase(b"say hello now", b"hello"));
        assert!(!contains_phrase(b"othello", b"hello"));
    }

    #[test]
    fn walk_matches_faithful_softmax() {
        let m = Model::parse(TINY).unwrap();
        for i in 0..m.intents {
            assert_eq!(m.generate(i), m.generate_faithful(i));
        }
        assert_eq!(m.generate(0), vec!["hi"]);
    }

    #[test]
    fn rejects_bad_header_and_dangling_refs() {
        assert!(Model::parse("NOPE 1\n").is_err());
        let bad = TINY.replace("NODE 0 1 hi", "NODE 0 9 hi");
        assert!(Model::parse(&bad).is_err());
    }

    #[test]
    fn normalizes() {
        assert_eq!(normalize_text(b"  Hi, THERE!! ", MAX_TEXT), b"hi there");
    }
}
