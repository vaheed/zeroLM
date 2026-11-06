//! Deterministic generated text streams with strict data roles.
//!
//! `exposure` is training input (the learner updates on it). `heldout`
//! contains only sentences whose exact text never appears in exposure.
//! `novel` contains only sentences that bind at least one held-out slot
//! combination, which exposure never contains. Contamination is checked,
//! not assumed (see [`Split::contamination`]).

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use zerolm_tensor::sha256::Sha256;
use zerolm_tensor::Rng;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamSpec {
    pub version: u32,
    pub id: String,
    pub description: String,
    pub separator: String,
    pub lexicon: BTreeMap<String, Vec<String>>,
    pub templates: Vec<String>,
    /// Each entry maps slot -> word; a sentence binding all of them is held out.
    pub held_out_combinations: Vec<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    Parse(String),
    UnknownSlot(String),
    UnknownWord { slot: String, word: String },
    EmptySlot(String),
    NoTemplates,
}

/// One generated sentence plus the slot bindings that produced it.
#[derive(Debug, Clone)]
pub struct Sentence {
    pub text: String,
    pub bindings: BTreeMap<String, String>,
}

impl StreamSpec {
    pub fn parse(json: &str) -> Result<Self, SpecError> {
        let s: StreamSpec = serde_json::from_str(json).map_err(|e| SpecError::Parse(e.to_string()))?;
        s.validate()?;
        Ok(s)
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        if self.templates.is_empty() {
            return Err(SpecError::NoTemplates);
        }
        for (slot, words) in &self.lexicon {
            if words.is_empty() {
                return Err(SpecError::EmptySlot(slot.clone()));
            }
        }
        for t in &self.templates {
            for slot in slots(t) {
                if !self.lexicon.contains_key(&slot) {
                    return Err(SpecError::UnknownSlot(slot));
                }
            }
        }
        for combo in &self.held_out_combinations {
            for (slot, word) in combo {
                match self.lexicon.get(slot) {
                    None => return Err(SpecError::UnknownSlot(slot.clone())),
                    Some(ws) if !ws.contains(word) => {
                        return Err(SpecError::UnknownWord { slot: slot.clone(), word: word.clone() })
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// SHA-256 of the canonical JSON of the spec.
    pub fn sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(serde_json::to_string(self).expect("serialize").as_bytes());
        h.finish_hex()
    }

    pub fn sample(&self, rng: &mut Rng) -> Sentence {
        let t = &self.templates[rng.below(self.templates.len())];
        let mut bindings = BTreeMap::new();
        let mut text = String::new();
        let mut rest = t.as_str();
        while let Some(open) = rest.find('{') {
            text.push_str(&rest[..open]);
            let close = open + rest[open..].find('}').expect("validated template");
            let slot = &rest[open + 1..close];
            let words = &self.lexicon[slot];
            let w = &words[rng.below(words.len())];
            text.push_str(w);
            bindings.insert(slot.to_string(), w.clone());
            rest = &rest[close + 1..];
        }
        text.push_str(rest);
        Sentence { text, bindings }
    }

    pub fn is_held_out(&self, s: &Sentence) -> bool {
        self.held_out_combinations
            .iter()
            .any(|c| c.iter().all(|(slot, w)| s.bindings.get(slot) == Some(w)))
    }

    /// Generate the three role-separated streams. Each role uses its own RNG
    /// stream derived from `seed`, so the exposure stream is identical no
    /// matter how much evaluation data is requested.
    pub fn split(&self, seed: u64, exposure_bytes: usize, eval_bytes: usize) -> Split {
        let mut exposure = String::new();
        let mut seen = BTreeSet::new();
        let mut rng = Rng::from_tag(seed, "stream.exposure");
        while exposure.len() < exposure_bytes {
            let s = self.sample(&mut rng);
            if self.is_held_out(&s) {
                continue;
            }
            exposure.push_str(&s.text);
            exposure.push_str(&self.separator);
            seen.insert(s.text);
        }
        let mut heldout = String::new();
        let mut rng = Rng::from_tag(seed, "stream.heldout");
        let mut guard = 0usize;
        while heldout.len() < eval_bytes && guard < 1_000_000 {
            guard += 1;
            let s = self.sample(&mut rng);
            if self.is_held_out(&s) || seen.contains(&s.text) {
                continue;
            }
            heldout.push_str(&s.text);
            heldout.push_str(&self.separator);
        }
        let mut novel = String::new();
        let mut rng = Rng::from_tag(seed, "stream.novel");
        guard = 0;
        while novel.len() < eval_bytes && guard < 1_000_000 {
            guard += 1;
            let s = self.sample(&mut rng);
            if !self.is_held_out(&s) {
                continue;
            }
            novel.push_str(&s.text);
            novel.push_str(&self.separator);
        }
        Split { exposure, heldout, novel, exposure_sentences: seen }
    }
}

fn slots(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = t;
    while let Some(open) = rest.find('{') {
        let Some(len) = rest[open..].find('}') else { break };
        out.push(rest[open + 1..open + len].to_string());
        rest = &rest[open + len + 1..];
    }
    out
}

pub struct Split {
    pub exposure: String,
    pub heldout: String,
    pub novel: String,
    exposure_sentences: BTreeSet<String>,
}

/// Leak counts between roles; all must be zero.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Contamination {
    pub heldout_sentences_in_exposure: usize,
    pub novel_sentences_in_exposure: usize,
    pub exposure_sentences_with_held_out_combination: usize,
}

impl Contamination {
    pub fn clean(&self) -> bool {
        self.heldout_sentences_in_exposure == 0
            && self.novel_sentences_in_exposure == 0
            && self.exposure_sentences_with_held_out_combination == 0
    }
}

impl Split {
    pub fn contamination(&self, spec: &StreamSpec) -> Contamination {
        let lines = |s: &str| s.split(spec.separator.as_str()).filter(|l| !l.is_empty()).map(str::to_string).collect::<Vec<_>>();
        let count = |s: &str| lines(s).iter().filter(|l| self.exposure_sentences.contains(*l)).count();
        // Re-parse exposure sentences against held-out word combinations by
        // whole-word containment (independent of the generator's bindings).
        let held_words: Vec<Vec<&str>> =
            spec.held_out_combinations.iter().map(|c| c.values().map(String::as_str).collect()).collect();
        let bad = lines(&self.exposure)
            .iter()
            .filter(|l| {
                let words: BTreeSet<&str> = l.split(|c: char| !c.is_ascii_alphanumeric()).collect();
                held_words.iter().any(|ws| ws.iter().all(|w| words.contains(w)))
            })
            .count();
        Contamination {
            heldout_sentences_in_exposure: count(&self.heldout),
            novel_sentences_in_exposure: count(&self.novel),
            exposure_sentences_with_held_out_combination: bad,
        }
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finish_hex()
}
