//! Evaluation-only harnesses (roadmap Phase 8): BLiMP (does the model give the
//! acceptable sentence of a minimal pair the higher log-probability?) and
//! LAMBADA (does greedy decoding reproduce the final word of a passage?).
//!
//! A model is anything that implements [`ByteLm`]. Scoring never updates the
//! model, and the harness never feeds benchmark text back to it.

use crate::ngram::{Kn, Wb};
use std::collections::BTreeMap;
use zerolm_data::evalsets::{BlimpPair, LambadaItem};

/// A byte-level language model: probability of the next byte given all bytes before it.
pub trait ByteLm {
    fn prob(&self, context: &[u8], next: u8) -> f64;
}

/// Order-5 context: the n-gram models only look at the last four bytes.
const TAIL: usize = 4;

fn tail(ctx: &[u8]) -> &[u8] {
    &ctx[ctx.len().saturating_sub(TAIL)..]
}

pub struct KneserNey<'a>(pub &'a Kn);
pub struct WittenBell<'a>(pub &'a Wb);
pub struct PpmC<'a>(pub &'a Wb);

impl ByteLm for KneserNey<'_> {
    fn prob(&self, c: &[u8], b: u8) -> f64 {
        self.0.prob(tail(c), b)
    }
}
impl ByteLm for WittenBell<'_> {
    fn prob(&self, c: &[u8], b: u8) -> f64 {
        self.0.prob(tail(c), b)
    }
}
impl ByteLm for PpmC<'_> {
    fn prob(&self, c: &[u8], b: u8) -> f64 {
        self.0.ppm_prob(tail(c), b)
    }
}

/// Every sentence is scored after a newline, which stands for "start of a text".
const START: u8 = b'\n';

/// Total log2-probability of `sentence` (bits, at most 0).
pub fn sentence_log2(lm: &dyn ByteLm, sentence: &str) -> f64 {
    let mut ctx = vec![START];
    let mut total = 0.0;
    for &b in sentence.as_bytes() {
        total += lm.prob(&ctx, b).max(f64::MIN_POSITIVE).log2();
        ctx.push(b);
    }
    total
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlimpResult {
    pub pairs: usize,
    pub correct: usize,
    /// Pairs where good and bad had exactly equal scores (counted as wrong).
    pub ties: usize,
    /// Correct pairs over all pairs.
    pub accuracy: f64,
    /// Mean of per-paradigm accuracies (BLiMP's own aggregate).
    pub paradigm_mean: f64,
    pub per_paradigm: BTreeMap<String, (usize, usize)>,
    pub per_field: BTreeMap<String, (usize, usize)>,
}

pub fn blimp(lm: &dyn ByteLm, pairs: &[BlimpPair]) -> BlimpResult {
    let (mut correct, mut ties) = (0, 0);
    let mut per_paradigm: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut per_field: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for p in pairs {
        let (g, b) = (sentence_log2(lm, &p.good), sentence_log2(lm, &p.bad));
        let ok = g > b;
        ties += (g == b) as usize;
        correct += ok as usize;
        for (map, key) in [(&mut per_paradigm, &p.uid), (&mut per_field, &p.field)] {
            let e = map.entry(key.clone()).or_default();
            e.0 += ok as usize;
            e.1 += 1;
        }
    }
    let paradigm_mean = per_paradigm.values().map(|(c, n)| *c as f64 / *n as f64).sum::<f64>() / per_paradigm.len().max(1) as f64;
    BlimpResult { pairs: pairs.len(), correct, ties, accuracy: correct as f64 / pairs.len().max(1) as f64, paradigm_mean, per_paradigm, per_field }
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'\'' || b >= 0x80
}

fn argmax(lm: &dyn ByteLm, ctx: &[u8]) -> u8 {
    let mut best = (f64::NEG_INFINITY, 0u8);
    for b in 0..=255u8 {
        let p = lm.prob(ctx, b);
        if p > best.0 {
            best = (p, b);
        }
    }
    best.1
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambadaResult {
    pub items: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// Mean bits the model assigns to the target word's bytes given its context.
    pub mean_target_bits: f64,
}

/// An item is correct when greedy byte decoding spells exactly the target and
/// then ends the word: the byte after the target must not continue a word.
pub fn lambada(lm: &dyn ByteLm, items: &[LambadaItem]) -> LambadaResult {
    let (mut correct, mut bits) = (0, 0.0);
    for it in items {
        let mut ctx = it.context.clone();
        let mut ok = true;
        for &t in &it.target {
            ok &= argmax(lm, &ctx) == t;
            bits -= lm.prob(&ctx, t).max(f64::MIN_POSITIVE).log2();
            ctx.push(t);
        }
        ok &= !is_word_byte(argmax(lm, &ctx));
        correct += ok as usize;
    }
    LambadaResult { items: items.len(), correct, accuracy: correct as f64 / items.len().max(1) as f64, mean_target_bits: bits / items.len().max(1) as f64 }
}
