//! Tokenization and embedding lookup. Generic: the vocabulary and embedding rows come from the
//! weight file; this module knows no word.

use super::network::Net;
use std::collections::HashMap;

pub const MAX_TOKENS: usize = 400;

/// Lowercase; `, . ;` are tokens; apostrophes vanish ("don't" -> "dont"); whitespace and the benign
/// marks `! ? : " ( ) -` separate; any other symbol becomes a token of its own (hence unknown),
/// so unexpected characters are never silently dropped.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() {
            out.push(std::mem::take(cur));
        }
    };
    for c in text.chars() {
        if c == '\'' || c == '\u{2019}' {
            continue;
        }
        if c.is_alphanumeric() {
            cur.extend(c.to_lowercase());
        } else if matches!(c, ',' | '.' | ';') {
            flush(&mut cur, &mut out);
            out.push(c.to_string());
        } else if c.is_whitespace() || matches!(c, '!' | '?' | ':' | '"' | '(' | ')' | '-' | '\u{201c}' | '\u{201d}') {
            flush(&mut cur, &mut out);
        } else {
            flush(&mut cur, &mut out);
            out.push(c.to_string());
        }
    }
    flush(&mut cur, &mut out);
    out
}

pub struct Encoder {
    index: HashMap<String, usize>,
    unk: usize,
}

impl Encoder {
    pub fn new(net: &Net) -> Result<Encoder, String> {
        let vocab = net.list("vocab");
        let index: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
        let unk = *index.get("<unk>").ok_or("vocab has no <unk>")?;
        Ok(Encoder { index, unk })
    }

    pub fn id(&self, tok: &str) -> usize {
        *self.index.get(tok).unwrap_or(&self.unk)
    }

    pub fn known(&self, tok: &str) -> bool {
        self.index.contains_key(tok)
    }
}
