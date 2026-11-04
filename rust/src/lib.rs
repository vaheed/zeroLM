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
pub const C_MODEL_STRUCT_BYTES: usize = 526_232;

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
    /// Normalized phrase bytes.
    pub phrase: Vec<u8>,
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
                        features.push(Feature {
                            intent: id as usize,
                            weight: w,
                            phrase: truncate_bytes(phrase, 191).into_bytes(),
                        });
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

        Ok(Model {
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

    /// Result of `classify`: (selected intent, raw logit score). Abstention
    /// (score below threshold) is applied by the caller via `decide`.
    pub fn classify(&self, normalized: &[u8]) -> (usize, f64) {
        self.classify_with(normalized, &[])
    }

    /// Like `classify`, with extra (learned) features added on top of the base
    /// weights. Extra features must reference valid intent ids.
    pub fn classify_with(&self, normalized: &[u8], extra: &[Feature]) -> (usize, f64) {
        let mut logits = vec![0.0f64; self.intents];
        for f in self.features.iter().chain(extra.iter()) {
            if contains_phrase(normalized, &f.phrase) {
                logits[f.intent] += f.weight;
            }
        }
        let best = argmax(&logits);
        (best, logits[best])
    }

    /// Distinct phrases (base + extra) that fired for `intent` on this input.
    pub fn matched_phrases(&self, normalized: &[u8], intent: usize, extra: &[Feature]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for f in self.features.iter().chain(extra.iter()) {
            if f.intent == intent && contains_phrase(normalized, &f.phrase) {
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
