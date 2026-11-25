//! Initialization and binding of the already-declared bounded grammar prior.
//! Strings are inspected here only while initializing fixed seed data. The
//! model path compares input rows using generic compiled programs.

use serde::Deserialize;
use zerolm_engine::isa::View;
use zerolm_memory::checkpoint::Tensor;
use zerolm_tensor::sha256::Sha256;
use zerolm_wsp::Compiled;

pub(crate) const KEY_BYTES: usize = 16;
pub(crate) const WINDOW: usize = 4;
/// Dimension names in the seed (`dimensions` of the parser seed).
pub(crate) const DIMENSIONS: usize = 4;
const GROUP: &str = "bootstrap_grammar";
const SEED: &str = include_str!("../../../model/bootstrap/reasoning-grammar.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Seed {
    version: u32,
    key_bytes: usize,
    tensor_sha256: String,
    lexicon: Vec<String>,
    rewrites: Vec<Rewrite>,
    parser: crate::parser_seed::Seed,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rewrite {
    from: Vec<String>,
    to: Vec<String>,
    at_end: bool,
}

pub(crate) struct Grammar {
    pub lookup: Compiled,
    pub rewrite: Compiled,
    pub tensors: Vec<Tensor>,
    pub parser: crate::parser::Programs,
    pub dimensions: Vec<String>,
    /// Rendering only: index zero is the unknown/padding label.
    pub words: Vec<String>,
}

impl Grammar {
    pub fn new() -> Self {
        let seed: Seed = serde_json::from_str(SEED).expect("valid grammar seed");
        assert_eq!((seed.version, seed.key_bytes), (1, KEY_BYTES));
        assert!(!seed.lexicon.is_empty() && !seed.rewrites.is_empty());
        assert_eq!(seed.parser.dimensions.len(), DIMENSIONS);
        let mut keys = Vec::new();
        for (i, word) in seed.lexicon.iter().enumerate() {
            assert!(!word.is_empty() && word.len() < KEY_BYTES && word.is_ascii());
            assert!(seed.lexicon[..i].iter().all(|other| other != word), "duplicate lexical row");
            let mut row = vec![0.0; KEY_BYTES];
            for (dst, src) in row.iter_mut().zip(word.bytes()) {
                *dst = src as f32;
            }
            keys.extend(row);
        }
        let id = |word: &str| (seed.lexicon.iter().position(|w| w == word).expect("rewrite word declared") + 1) as f32;
        let mut from = Vec::new();
        let mut to = Vec::new();
        let mut from_mask = Vec::new();
        let mut to_mask = Vec::new();
        let mut at_end = Vec::new();
        for rule in &seed.rewrites {
            assert!(!rule.from.is_empty() && rule.from.len() <= WINDOW && rule.to.len() <= rule.from.len());
            for i in 0..WINDOW {
                from.push(rule.from.get(i).map(|w| id(w)).unwrap_or(0.0));
                to.push(rule.to.get(i).map(|w| id(w)).unwrap_or(0.0));
                from_mask.push((i < rule.from.len()) as u8 as f32);
                to_mask.push((i < rule.to.len()) as u8 as f32);
            }
            at_end.push(rule.at_end as u8 as f32);
        }
        let n = seed.lexicon.len();
        let r = seed.rewrites.len();
        let mut tensors = vec![
            Tensor::new("bootstrap.reasoning.keys", GROUP, n, KEY_BYTES, keys),
            Tensor::new("bootstrap.reasoning.values", GROUP, n, 1, (1..=n).map(|i| i as f32).collect()),
            Tensor::new("bootstrap.reasoning.from_ids", GROUP, r, WINDOW, from),
            Tensor::new("bootstrap.reasoning.from_mask", GROUP, r, WINDOW, from_mask),
            Tensor::new("bootstrap.reasoning.to_ids", GROUP, r, WINDOW, to),
            Tensor::new("bootstrap.reasoning.to_mask", GROUP, r, WINDOW, to_mask),
            Tensor::new("bootstrap.reasoning.at_end", GROUP, r, 1, at_end),
        ];
        crate::parser_seed::augment(&seed.parser, &seed.lexicon, &mut tensors);
        assert_eq!(Self::hash(&tensors), seed.tensor_sha256, "deliberate grammar changes require re-pinning");
        Grammar {
            lookup: zerolm_wsp::shipped("row-lookup").expect("lookup compiles"),
            rewrite: zerolm_wsp::shipped("sequence-rewrite").expect("rewrite compiles"),
            tensors,
            parser: crate::parser::Programs::new(),
            dimensions: seed.parser.dimensions,
            words: std::iter::once(String::new()).chain(seed.lexicon).collect(),
        }
    }

    fn hash(tensors: &[Tensor]) -> String {
        let mut h = Sha256::new();
        h.update(b"zerolm-reasoning-grammar-v1\0");
        for t in tensors {
            h.update(t.name.as_bytes());
            h.update(&(t.rows as u64).to_le_bytes());
            h.update(&(t.cols as u64).to_le_bytes());
            h.update_f32(&t.data);
        }
        h.finish_hex()
    }

    pub fn view(&self, name: &str) -> View<'_> {
        let full = format!("bootstrap.reasoning.{name}");
        let t = self.tensors.iter().find(|t| t.name == full).expect("declared parser table");
        View::new(t.rows, t.cols, &t.data)
    }

    pub fn row(&self, tensor: usize, row: usize) -> View<'_> {
        let t = &self.tensors[tensor];
        View::new(1, t.cols, &t.data[row * t.cols..(row + 1) * t.cols])
    }
}
