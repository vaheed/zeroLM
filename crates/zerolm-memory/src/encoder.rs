//! Key encoder table: a programmed (not learned) random-feature embedding bag.
//!
//! The encoding itself runs as the compiled `memory-encode` circuit
//! (`read_circuit`, `zerolm_wsp::circuits::key_encode`): text -> words (ASCII
//! letters lower-cased; ASCII non-alphanumerics separate words; bytes >= 0x80
//! are word bytes) -> byte trigrams of "#word#" -> FNV-1a-hashed rows of the
//! seeded table `core.memory_encoder` (buckets x key_dim) -> per-word sum,
//! unit normalized -> sum over words, unit normalized.
//!
//! Consequences stated plainly: the key is invariant to ASCII case,
//! punctuation, and word order (so "a likes b" and "b likes a" share a key);
//! extra words dilute similarity; there is no learned notion of synonymy.

use zerolm_tensor::Matrix;

pub const TENSOR: &str = "core.memory_encoder";

#[derive(Debug, Clone, PartialEq)]
pub struct KeyEncoder {
    pub table: Matrix,
}

impl KeyEncoder {
    pub fn new(seed: u64, buckets: usize, key_dim: usize) -> Self {
        KeyEncoder { table: Matrix::uniform(buckets, key_dim, seed, TENSOR, 1.0) }
    }

    pub fn key_dim(&self) -> usize {
        self.table.cols
    }

    pub fn view(&self) -> zerolm_engine::isa::View<'_> {
        zerolm_engine::isa::View::new(self.table.rows, self.table.cols, &self.table.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistent_memory::MemoryConfig;
    use crate::read_circuit::{self, TEXT_BYTES};
    use zerolm_tensor::{dot, fnv1a, norm};

    /// The former host encoder, kept only as an independent reference for
    /// the circuit (same arithmetic order, so the results must be bit-equal).
    fn host_words(table: &Matrix, text: &str) -> Vec<Vec<f32>> {
        let mut words = Vec::new();
        let mut cur = Vec::new();
        for &b in text.as_bytes().iter().chain(std::iter::once(&b' ')) {
            if b.is_ascii_alphanumeric() || b >= 0x80 {
                cur.push(b.to_ascii_lowercase());
            } else if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        }
        words
            .into_iter()
            .map(|w| {
                let padded: Vec<u8> = std::iter::once(b'#').chain(w).chain(std::iter::once(b'#')).collect();
                let mut wv = vec![0.0f32; table.cols];
                for t in padded.windows(3) {
                    let row = table.row((fnv1a(t) % table.rows as u64) as usize);
                    wv.iter_mut().zip(row).for_each(|(a, b)| *a += b);
                }
                let n = norm(&wv);
                wv.iter_mut().for_each(|x| *x /= n);
                wv
            })
            .collect()
    }

    fn host_pool(words: &[Vec<f32>], d: usize) -> Vec<f32> {
        let mut key = vec![0.0f32; d];
        for w in words {
            key.iter_mut().zip(w).for_each(|(a, b)| *a += b);
        }
        let n = norm(&key);
        key.iter_mut().for_each(|x| *x /= n);
        key
    }

    fn encode(c: &zerolm_wsp::Compiled, e: &KeyEncoder, text: &str) -> zerolm_engine::isa::Outputs {
        read_circuit::run_encoder(c, e.view(), read_circuit::text_input(text).unwrap()).unwrap()
    }

    #[test]
    fn the_circuit_equals_the_host_reference_bit_for_bit() {
        let cfg = MemoryConfig::default();
        let e = KeyEncoder::new(cfg.seed, cfg.buckets, cfg.key_dim);
        let c = read_circuit::build_encoder(&cfg);
        let long = "w ".repeat((TEXT_BYTES - 2) / 2);
        let texts = [
            "zorvan color",
            "  Zorvan, COLOR?!",
            "a",
            "What is the HOME of blixo-trel #42?",
            "naïve café über 東京",
            "x9 9x _ a_b a#b @[`{",
            "\u{1F600}smile",
            long.as_str(),
        ];
        for t in texts {
            let o = encode(&c, &e, t);
            let want = host_words(&e.table, t);
            let n = want.len();
            assert_eq!(o.scalar("n_words"), n as f32, "{t:?}");
            let words = o.get("words");
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            for (i, w) in want.iter().enumerate() {
                assert_eq!(bits(&words.data[i * cfg.key_dim..(i + 1) * cfg.key_dim]), bits(w), "{t:?} word {i}");
            }
            assert!(words.data[n * cfg.key_dim..].iter().all(|x| *x == 0.0), "{t:?}: padding rows are zero");
            let mask = &o.get("word_mask").data;
            assert!(mask.iter().enumerate().all(|(i, m)| *m == (i < n) as u8 as f32), "{t:?}");
            assert_eq!(bits(&o.get("key").data), bits(&host_pool(&want, cfg.key_dim)), "{t:?} key");
        }
        assert!(read_circuit::text_input(&format!("{long}ab")).is_none());
    }

    #[test]
    fn case_punctuation_and_order_do_not_change_the_key() {
        let cfg = MemoryConfig::default();
        let e = KeyEncoder::new(1, cfg.buckets, cfg.key_dim);
        let c = read_circuit::build_encoder(&cfg);
        let key = |t: &str| encode(&c, &e, t).get("key").data.clone();
        let a = key("zorvan color");
        assert_eq!(a, key("  Zorvan, COLOR?!"));
        assert!((dot(&a, &key("color zorvan")) - 1.0).abs() < 1e-5);
        let empty = encode(&c, &e, "?? !!");
        assert_eq!(empty.scalar("n_words"), 0.0);
        assert!(empty.get("key").data.iter().all(|x| *x == 0.0));
        assert!(dot(&a, &key("blixo home")).abs() < 0.4);
    }
}
