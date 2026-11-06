//! Key encoder: a programmed (not learned) random-feature embedding bag.
//!
//! text -> normalized words -> byte trigrams of "#word#" -> hashed rows of a
//! seeded table `memory.encoder` (buckets x key_dim) -> per-word sum, unit
//! normalized -> sum over words, unit normalized.
//!
//! Consequences stated plainly: the key is invariant to ASCII case,
//! punctuation, and word order (so "a likes b" and "b likes a" share a key);
//! extra words dilute similarity; there is no learned notion of synonymy.

use zerolm_tensor::{fnv1a, norm, Matrix};

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

    /// ASCII letters are lower-cased, ASCII non-alphanumerics separate words,
    /// and non-ASCII bytes are kept as word characters.
    pub fn words(text: &str) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut cur = Vec::new();
        for &b in text.as_bytes() {
            if b.is_ascii_alphanumeric() || b >= 0x80 {
                cur.push(b.to_ascii_lowercase());
            } else if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// Unit-norm key, or `None` when the text has no word characters.
    pub fn encode(&self, text: &str) -> Option<Vec<f32>> {
        let d = self.key_dim();
        let mut key = vec![0.0f32; d];
        let mut any = false;
        for w in Self::words(text) {
            let mut padded = Vec::with_capacity(w.len() + 2);
            padded.push(b'#');
            padded.extend_from_slice(&w);
            padded.push(b'#');
            let mut wv = vec![0.0f32; d];
            for t in padded.windows(3) {
                let row = self.table.row((fnv1a(t) % self.table.rows as u64) as usize);
                wv.iter_mut().zip(row).for_each(|(a, b)| *a += b);
            }
            let n = norm(&wv);
            if n > 0.0 {
                key.iter_mut().zip(&wv).for_each(|(a, b)| *a += b / n);
                any = true;
            }
        }
        let n = norm(&key);
        if !any || n == 0.0 {
            return None;
        }
        key.iter_mut().for_each(|x| *x /= n);
        Some(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerolm_tensor::dot;

    #[test]
    fn case_punctuation_and_order_do_not_change_the_key() {
        let e = KeyEncoder::new(1, 4096, 128);
        let a = e.encode("zorvan color").unwrap();
        assert_eq!(a, e.encode("  Zorvan, COLOR?!").unwrap());
        let r = e.encode("color zorvan").unwrap();
        assert!((dot(&a, &r) - 1.0).abs() < 1e-5);
        assert!(e.encode("?? !!").is_none());
        let other = e.encode("blixo home").unwrap();
        assert!(dot(&a, &other).abs() < 0.4);
    }
}
