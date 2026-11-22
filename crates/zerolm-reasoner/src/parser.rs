//! Circuit binding and output rendering for the seeded bounded grammar.
//! The host copies bytes and tensor rows. Recognition and attribute decisions
//! are outputs of compiled programs; enum rendering remains an interface to
//! the existing binding/dispatch path during the uniform-engine migration.

use crate::{Reasoner, grammar_seed::KEY_BYTES, parser_seed::{TOKENS, FEATURES}};
use crate::circuits::with_consts;
use zerolm_engine::isa::{Buf, View};
use zerolm_engine::semantic_state::{Fact, Pred, Query, Rule, Term, Utterance};
use zerolm_wsp::Compiled;

pub(crate) struct Programs {
    pub byte_lookup: Compiled, pub number: Compiled, pub features: Compiled,
    pub whitespace: Compiled, pub tail: Compiled, pub compare: Compiled, pub matcher: Compiled,
}
impl Programs {
    pub fn new() -> Self {
        let get = |id| zerolm_wsp::shipped(id).expect("shipped parser program compiles");
        Self { byte_lookup: get("byte-lookup"), number: get("token-number"), features: get("token-features"), whitespace: get("masked-lookup"), tail: get("question-tail"), compare: get("row-compare"), matcher: get("template-match") }
    }
    pub fn all(&self) -> [&Compiled; 7] { [&self.byte_lookup, &self.number, &self.features, &self.whitespace, &self.tail, &self.compare, &self.matcher] }
}

impl Reasoner {
    fn lexical_rows(&self, words: &[String]) -> Buf {
        let g = &self.grammar;
        let mut rows = vec![0.0; words.len() * KEY_BYTES];
        for (i, word) in words.iter().enumerate() {
            for (dst, byte) in rows[i * KEY_BYTES..(i + 1) * KEY_BYTES].iter_mut().zip(word.bytes()) { *dst = byte as f32; }
        }
        let weights = with_consts(&g.lookup, vec![("table_keys", g.view("keys")), ("table_values", g.view("values"))]);
        self.exec(&g.lookup, vec![("keys", Buf::new(words.len(), KEY_BYTES, rows))], &weights).get("values").clone()
    }

    pub(crate) fn parser_features(&self, words: &[String]) -> Buf {
        let g = &self.grammar;
        let lexical = self.lexical_rows(words);
        let byte_weights = with_consts(&g.parser.byte_lookup, vec![("table_keys", g.view("byte_keys")), ("table_values", g.view("byte_values"))]);
        let number_weights = with_consts(&g.parser.number, vec![("limit", g.view("number_limit"))]);
        let feature_weights = with_consts(&g.parser.features, vec![]);
        let mut numbers = Vec::new(); let mut singular = Vec::new();
        for word in words {
            let keys: Vec<f32> = word.bytes().map(|b| b as f32).collect();
            let out = self.exec(&g.parser.byte_lookup, vec![("keys", Buf::new(keys.len(), 1, keys))], &byte_weights);
            let traits = out.get("values"); let length = word.len();
            // Place weights depend only on shape. Capping at 10^7 retains
            // exact values through the declared 10^6 bound and rejects any
            // nonzero higher-place digit, including arbitrarily long numerals.
            let places = (0..length).map(|i| 10u32.pow((length - 1 - i).min(7) as u32) as f32).collect();
            let out = self.exec(&g.parser.number, vec![
                ("traits", traits.clone()), ("places", Buf::row(places)),
                ("first", Buf::row(traits.data[..4].to_vec())),
                ("last", Buf::row(traits.data[(length-1)*4..length*4].to_vec())),
                ("length", Buf::row(vec![length as f32])),
            ], &number_weights);
            let n = out.get("singular_length").data[0] as usize;
            singular.push(word[..n].to_string());
            numbers.push(Buf::row(vec![out.get("number_valid").data[0], out.get("number").data[0], out.get("ends_s").data[0], n as f32]));
        }
        let singular_lexical = self.lexical_rows(&singular);
        let mut values = Vec::new();
        for (i, numeric) in numbers.into_iter().enumerate() {
            let out = self.exec(&g.parser.features, vec![
                ("lexical", Buf::row(lexical.data[i * FEATURES..(i+1)*FEATURES].to_vec())),
                ("singular", Buf::row(singular_lexical.data[i * FEATURES..(i+1)*FEATURES].to_vec())),
                ("numeric", numeric),
            ], &feature_weights);
            values.extend_from_slice(&out.get("features").data);
        }
        Buf::new(words.len(), FEATURES, values)
    }

    pub(crate) fn raw_question(&self, text: &str) -> Buf {
        let bytes: Vec<f32> = text.bytes().map(|b| b as f32).collect(); let n = bytes.len();
        if n == 0 { return Buf::row(vec![0.0]); }
        let g = &self.grammar;
        let mut windows = Vec::with_capacity(n * 3);
        for i in 0..n { for k in 0..3 { windows.push(bytes.get(i+k).copied().unwrap_or(0.0)); } }
        let weights = with_consts(&g.parser.whitespace, vec![("table_keys", g.view("whitespace_keys")), ("table_masks", g.view("whitespace_masks")), ("table_values", g.view("whitespace_values"))]);
        let out = self.exec(&g.parser.whitespace, vec![("keys", Buf::new(n, 3, windows))], &weights);
        let spans = &out.get("values").data;
        let mut coverage = Vec::with_capacity(n * 3);
        for i in 0..n { for k in 0..3 { coverage.push(i.checked_sub(k).map(|j| spans[j]).unwrap_or(0.0)); } }
        let weights = with_consts(&g.parser.tail, vec![("marker", g.view("question_marker"))]);
        self.exec(&g.parser.tail, vec![("bytes", Buf::new(n, 1, bytes)), ("spans", Buf::new(n, 3, coverage)), ("positions", Buf::row((1..=n).map(|i| i as f32).collect()))], &weights).get("question").clone()
    }

    fn token_equality(&self, words: &[String]) -> Buf {
        let c = &self.grammar.parser.compare;
        let mut same = Buf::new(TOKENS, TOKENS, vec![1.0; TOKENS * TOKENS]);
        let width = words.iter().map(String::len).max().unwrap_or(0);
        for offset in (0..width).step_by(KEY_BYTES) {
            let mut keys = vec![0.0; TOKENS * KEY_BYTES];
            for (i, word) in words.iter().enumerate() {
                for (dst, byte) in keys[i * KEY_BYTES..(i+1)*KEY_BYTES].iter_mut().zip(word.bytes().skip(offset)) { *dst = byte as f32; }
            }
            let weights = with_consts(c, vec![("table_keys", View::new(TOKENS, KEY_BYTES, &keys))]);
            same = self.exec(c, vec![("keys", Buf::new(TOKENS, KEY_BYTES, keys.clone())), ("previous", same)], &weights).get("mask").clone();
        }
        same
    }

    pub(crate) fn recognized_frame(&self, text: &str, words: &[String], features: &Buf, arity: usize) -> Buf {
        let g = &self.grammar; let c = &g.parser.matcher;
        let mut padded = vec![0.0; TOKENS * FEATURES]; padded[..features.data.len()].copy_from_slice(&features.data);
        let mut external = vec![("lengths", g.view("template_lengths")), ("questions", g.view("template_questions")), ("priorities", g.view("template_priorities")), ("literal_ids", g.view("template_literals")), ("literal_mask", g.view("template_literal_masks")), ("equalities", g.view("template_equalities")), ("payload", g.view("template_payload")), ("sources", g.view("template_sources"))];
        let names: Vec<_> = (0..TOKENS).map(|i| format!("required{i}")).collect();
        for (i, name) in names.iter().enumerate() { external.push((name.as_str(), g.view(&format!("template_required{i}")))); }
        let weights = with_consts(c, external);
        self.exec(c, vec![("features", Buf::new(TOKENS, FEATURES, padded)), ("same", self.token_equality(words)), ("length", Buf::row(vec![arity as f32])), ("raw_question", self.raw_question(text))], &weights).get("frame").clone()
    }

    pub(crate) fn parse_circuits(&self, text: &str) -> Option<Utterance> {
        let all_words = self.normalized_tokens(text);
        if all_words.is_empty() { return None; }
        let words: Vec<_> = all_words.iter().take(TOKENS).cloned().collect();
        let features = self.parser_features(&words);
        // The length input must retain the full arity even when only the
        // fixed-capacity prefix is classified; overlong clauses cannot match.
        let frame = self.recognized_frame(text, &words, &features, all_words.len());
        self.render_frame(&frame.data, &words, &features)
    }

    fn render_frame(&self, f: &[f32], words: &[String], features: &Buf) -> Option<Utterance> {
        let word = |i: usize, singular: bool| {
            let length = if singular { features.data[i * FEATURES + 14] as usize } else { words[i].len() };
            words[i][..length].to_string()
        };
        let subject = || if f[4] != 0.0 { Term::Pronoun } else { Term::Name(word(f[2] as usize, f[3] != 0.0)) };
        let dimension = || self.grammar.dimensions[f[7] as usize - 1].clone();
        let object = || word(f[5] as usize, f[6] != 0.0);
        let thing = || word(f[15] as usize, true);
        match f[0] as usize {
            0 => None,
            2 => Some(Utterance::Rule(Rule { scope: (f[11] != 0.0).then(|| word(f[2] as usize, false)), premise: word(f[12] as usize, false), conclusion: word(f[13] as usize, false) })),
            4 => Some(Utterance::Query(Query::Count { subj: subject(), thing: thing() })),
            5 => Some(Utterance::Query(Query::Which { dim: dimension(), higher: f[14] != 0.0, a: word(f[2] as usize, false), b: object() })),
            form @ (1 | 3) => {
                let pred = match f[1] as usize {
                    1 => Pred::Isa(object()), 2 => Pred::Can(object()), 3 => Pred::In(object()), 4 => Pred::Prop(object()),
                    5 => Pred::Greater { dim: dimension(), other: object() }, 6 => Pred::Before(object()),
                    7 => Pred::Count { thing: thing(), n: f[8] as i64 },
                    8 => Pred::More { thing: thing(), delta: f[8] as i64, other: object() },
                    _ => unreachable!("validated template predicate"),
                };
                Some(if form == 1 { Utterance::Fact(Fact { subj: subject(), pred, negated: f[9] != 0.0 }) } else { Utterance::Query(Query::Holds { subj: subject(), pred }) })
            }
            _ => unreachable!("validated template form"),
        }
    }
}

#[cfg(test)]
#[path = "parser_tests.rs"]
mod tests;
