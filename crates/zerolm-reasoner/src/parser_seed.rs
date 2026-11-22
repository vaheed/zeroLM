//! Initialization of declared grammar tables. All strings inspected here are
//! fixed seed data; this module never receives user text.

use serde::Deserialize;
use std::collections::BTreeMap;
use zerolm_memory::checkpoint::Tensor;

pub const TOKENS: usize = 9;
pub const FEATURES: usize = 20;
pub const FIELDS: usize = 16;
const GROUP: &str = "bootstrap_grammar";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Seed {
    reserved: Vec<String>, number_words: Vec<String>, pronouns: Vec<String>,
    articles: Vec<String>, quantifiers: Vec<String>, leading_questions: Vec<String>,
    pub dimensions: Vec<String>, comparatives: Vec<Comparative>,
    digit_bytes: Vec<u8>, minus_byte: u8, suffix_byte: u8,
    number_limit: usize, whitespace: Vec<Vec<u8>>, question_byte: u8,
    templates: Vec<Template>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Comparative { word: String, dimension: String, greater: bool }
#[derive(Deserialize)]
#[serde(untagged)]
enum Cell { Literal(String), Requirements { require: Vec<String> } }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Read { token: usize, feature: String, #[serde(default = "one")] scale: f32 }
fn one() -> f32 { 1.0 }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Template {
    tokens: Vec<Cell>, question: u8, frame: BTreeMap<String, f32>,
    reads: BTreeMap<String, Read>, same: Vec<[usize; 2]>,
}

fn feature(name: &str) -> usize {
    ["id", "name", "term", "pronoun", "singular_name", "ends_s", "number_valid", "number", "comparative", "dimension", "greater", "article", "quantifier", "leading_question", "singular_length", "present", "plural_subject"]
        .iter().position(|n| *n == name).expect("declared feature")
}
fn field(name: &str) -> usize {
    ["form", "pred", "subject", "subject_singular", "pronoun", "object", "object_singular", "dimension", "number", "negated", "unused", "scope", "premise", "conclusion", "higher", "thing"]
        .iter().position(|n| *n == name).expect("declared frame field")
}
fn tensor(name: &str, rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Tensor::new(&format!("bootstrap.reasoning.{name}"), GROUP, rows, cols, data)
}

pub fn augment(s: &Seed, lexicon: &[String], tensors: &mut Vec<Tensor>) {
    assert_eq!(s.digit_bytes.len(), 10);
    assert_eq!(s.number_limit, 1_000_000);
    let id = |word: &str| (lexicon.iter().position(|w| w == word).expect("declared literal") + 1) as f32;
    let mut values = vec![0.0; lexicon.len() * FEATURES];
    for (i, word) in lexicon.iter().enumerate() {
        let v = &mut values[i * FEATURES..(i + 1) * FEATURES];
        v[0] = id(word);
        for (at, words) in [(1, &s.reserved), (2, &s.pronouns), (8, &s.articles), (9, &s.quantifiers), (10, &s.leading_questions)] {
            v[at] = words.contains(word) as u8 as f32;
        }
        if let Some(n) = s.number_words.iter().position(|w| w == word) { v[3] = 1.0; v[4] = n as f32; }
        if let Some(c) = s.comparatives.iter().find(|c| c.word == *word) {
            v[5] = 1.0;
            v[6] = (s.dimensions.iter().position(|d| *d == c.dimension).expect("declared dimension") + 1) as f32;
            v[7] = c.greater as u8 as f32;
        }
    }
    tensors[1] = tensor("values", lexicon.len(), FEATURES, values);
    let mut bytes = vec![0.0; 256 * 4];
    for (value, byte) in s.digit_bytes.iter().enumerate() {
        bytes[*byte as usize * 4] = 1.0; bytes[*byte as usize * 4 + 1] = value as f32;
    }
    bytes[s.minus_byte as usize * 4 + 2] = 1.0;
    bytes[s.suffix_byte as usize * 4 + 3] = 1.0;
    tensors.push(tensor("byte_keys", 256, 1, (0..256).map(|i| i as f32).collect()));
    tensors.push(tensor("byte_values", 256, 4, bytes));
    tensors.push(tensor("number_limit", 1, 1, vec![s.number_limit as f32]));
    let mut keys = Vec::new(); let mut masks = Vec::new(); let mut lengths = Vec::new();
    for bytes in &s.whitespace {
        assert!(!bytes.is_empty() && bytes.len() <= 3);
        for i in 0..3 { keys.push(bytes.get(i).copied().unwrap_or(0) as f32); masks.push((i < bytes.len()) as u8 as f32); }
        lengths.push(bytes.len() as f32);
    }
    let n = s.whitespace.len();
    tensors.push(tensor("whitespace_keys", n, 3, keys));
    tensors.push(tensor("whitespace_masks", n, 3, masks));
    tensors.push(tensor("whitespace_values", n, 1, lengths));
    tensors.push(tensor("question_marker", 1, 1, vec![s.question_byte as f32]));

    let p = s.templates.len(); assert!(p > 0);
    let mut lens = Vec::new(); let mut questions = Vec::new();
    let mut literals = vec![0.0; p * TOKENS]; let mut literal_masks = literals.clone();
    let mut required = vec![vec![0.0; p * FEATURES]; TOKENS];
    let mut equal = vec![0.0; p * TOKENS * TOKENS];
    let mut payload = vec![0.0; p * FIELDS];
    let mut sources = vec![0.0; p * FIELDS * TOKENS * FEATURES];
    for (row, template) in s.templates.iter().enumerate() {
        assert!(!template.tokens.is_empty() && template.tokens.len() <= TOKENS && template.question <= 1);
        lens.push(template.tokens.len() as f32); questions.push(template.question as f32);
        for (i, cell) in template.tokens.iter().enumerate() {
            match cell {
                Cell::Literal(word) => { literals[row * TOKENS + i] = id(word); literal_masks[row * TOKENS + i] = 1.0; }
                Cell::Requirements { require } => for name in require { required[i][row * FEATURES + feature(name)] = 1.0; },
            }
        }
        for [a, b] in &template.same {
            assert!(*a < template.tokens.len() && *b < template.tokens.len());
            equal[row * TOKENS * TOKENS + a * TOKENS + b] = 1.0;
        }
        for (name, value) in &template.frame { assert!(value.is_finite()); payload[row * FIELDS + field(name)] = *value; }
        for (name, read) in &template.reads {
            assert!(read.token < template.tokens.len() && read.scale.is_finite());
            sources[(row * FIELDS + field(name)) * TOKENS * FEATURES + read.token * FEATURES + feature(&read.feature)] = read.scale;
        }
    }
    tensors.push(tensor("template_lengths", p, 1, lens));
    tensors.push(tensor("template_questions", p, 1, questions));
    tensors.push(tensor("template_priorities", p, 1, (0..p).map(|i| (p-i) as f32).collect()));
    tensors.push(tensor("template_literals", p, TOKENS, literals));
    tensors.push(tensor("template_literal_masks", p, TOKENS, literal_masks));
    for (i, values) in required.into_iter().enumerate() { tensors.push(tensor(&format!("template_required{i}"), p, FEATURES, values)); }
    tensors.push(tensor("template_equalities", p, TOKENS * TOKENS, equal));
    tensors.push(tensor("template_payload", p, FIELDS, payload));
    tensors.push(tensor("template_sources", p, FIELDS * TOKENS * FEATURES, sources));
}
