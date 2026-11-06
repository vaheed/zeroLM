//! Byte n-gram baselines (interpolated Witten-Bell), learned online from the
//! same exposure stream and evaluated frozen. Order 1 is the byte-frequency
//! (unigram) baseline.

use std::collections::HashMap;

pub struct Ngram {
    order: usize,
    /// counts[k]: context of length k -> next-byte counts.
    counts: Vec<HashMap<Vec<u8>, Vec<u32>>>,
    history: Vec<u8>,
}

impl Ngram {
    pub fn new(order: usize) -> Self {
        assert!(order >= 1);
        Ngram { order, counts: (0..order).map(|_| HashMap::new()).collect(), history: Vec::new() }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    fn prob(&self, hist: &[u8], b: u8) -> f64 {
        let mut p = 1.0 / 256.0;
        for k in 0..self.order {
            if k > hist.len() {
                break;
            }
            let ctx = &hist[hist.len() - k..];
            let Some(c) = self.counts[k].get(ctx) else { break };
            let total: u64 = c.iter().map(|&x| x as u64).sum();
            let distinct = c.iter().filter(|&&x| x > 0).count() as f64;
            let lambda = total as f64 / (total as f64 + distinct);
            p = lambda * c[b as usize] as f64 / total as f64 + (1.0 - lambda) * p;
        }
        p
    }

    /// Learn from a stream chronologically.
    pub fn observe_all(&mut self, bytes: &[u8]) {
        for &b in bytes {
            for k in 0..self.order.min(self.history.len() + 1) {
                let ctx = self.history[self.history.len() - k..].to_vec();
                self.counts[k].entry(ctx).or_insert_with(|| vec![0; 256])[b as usize] += 1;
            }
            self.history.push(b);
            if self.history.len() > self.order {
                self.history.remove(0);
            }
        }
    }

    /// Bits per byte on `bytes` (fresh history, no learning).
    pub fn bpb(&self, bytes: &[u8]) -> f64 {
        let mut bits = 0.0;
        for (i, &b) in bytes.iter().enumerate() {
            let hist = &bytes[i.saturating_sub(self.order - 1)..i];
            bits -= self.prob(hist, b).log2();
        }
        bits / bytes.len().max(1) as f64
    }
}
