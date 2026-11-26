//! Novelty pilot, step 0: how much can an induction-style algorithmic prior add on top of a strong
//! count model? (`docs/experimental-design.md`, experiment N0.)
//!
//! The prior is a *match model*: find the longest earlier occurrence of the current context (at least
//! `MIN_LEN` bytes, at most `MAX_LEN`) and predict the byte that followed it. Its confidence is a
//! running hit rate per match-length bucket, updated only after each byte is scored (prequential, no
//! label leakage). It is mixed with a frozen count model as `p = (1-c) p_count + c [b = b*]`.
//! Research code: this is the host-side reference for the algorithm; a compiled circuit would have to
//! reproduce it on the neural ISA (constitution invariant 10) before any mainline claim.
//! Nothing here is learned from a corpus: the only state is the history and ten counters.

use std::collections::HashMap;

pub const MIN_LEN: usize = 5;
pub const MAX_LEN: usize = 32;
const BUCKETS: usize = 7;

fn bucket(len: usize) -> usize {
    match len {
        0..=5 => 0,
        6 => 1,
        7 => 2,
        8..=11 => 3,
        12..=15 => 4,
        16..=23 => 5,
        _ => 6,
    }
}

/// Longest-recent-match predictor over a growing byte history.
pub struct MatchModel {
    hist: Vec<u8>,
    last: HashMap<u64, u32>,
    hits: [[u32; 4]; BUCKETS],
    tries: [[u32; 4]; BUCKETS],
}

fn key(w: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in w {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

impl MatchModel {
    pub fn new() -> Self {
        MatchModel { hist: Vec::new(), last: HashMap::new(), hits: [[0; 4]; BUCKETS], tries: [[0; 4]; BUCKETS] }
    }

    /// Predicted next byte and the verified match length, or None.
    pub fn predict(&self) -> Option<(u8, usize)> {
        let n = self.hist.len();
        if n < MIN_LEN {
            return None;
        }
        let &p = self.last.get(&key(&self.hist[n - MIN_LEN..]))?;
        let p = p as usize; // index of the byte that followed the earlier occurrence
        let mut len = 0;
        while len < MAX_LEN && len < p && self.hist[p - 1 - len] == self.hist[n - 1 - len] {
            len += 1;
        }
        if len < MIN_LEN {
            return None;
        }
        Some((self.hist[p], len))
    }

    /// Confidence that the prediction of a match of length `len` is right.
    /// `q` is the quantized count-model probability of the predicted byte (0 for the v1 mixer).
    pub fn confidence(&self, len: usize, q: usize) -> f64 {
        let b = bucket(len);
        (self.hits[b][q] as f64 + 1.0) / (self.tries[b][q] as f64 + 2.0)
    }

    /// Record the true byte: update the bucket's hit rate, then extend the history and the index.
    pub fn observe(&mut self, byte: u8, q: usize) {
        if let Some((pred, len)) = self.predict() {
            let b = bucket(len);
            self.tries[b][q] += 1;
            if pred == byte {
                self.hits[b][q] += 1;
            }
        }
        self.hist.push(byte);
        let n = self.hist.len();
        if n > MIN_LEN {
            // the window ending at n-1 (exclusive of the new byte) was followed by hist[n-1]
            self.last.insert(key(&self.hist[n - 1 - MIN_LEN..n - 1]), (n - 1) as u32);
        }
    }

    /// Absorb bytes without scoring (exposure history).
    pub fn absorb(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hist.push(b);
            let n = self.hist.len();
            if n > MIN_LEN {
                self.last.insert(key(&self.hist[n - 1 - MIN_LEN..n - 1]), (n - 1) as u32);
            }
        }
    }
}

impl Default for MatchModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Bits per byte of every position under (count model, count model mixed with the match model).
/// `count_prob(i, b)` must return the count model's probability of byte `b` at position `i` given its history.
/// With `conditional` the match confidence is also conditioned on the count model's own probability of the
/// predicted byte (4 levels), so redundant matches are not double counted (mixer v2).
pub fn score_pair(mm: &mut MatchModel, bytes: &[u8], conditional: bool, mut count_prob: impl FnMut(usize, u8) -> f64) -> (Vec<f32>, Vec<f32>) {
    let (mut base, mut mixed) = (Vec::with_capacity(bytes.len()), Vec::with_capacity(bytes.len()));
    for (i, &b) in bytes.iter().enumerate() {
        let pc = count_prob(i, b);
        base.push(-pc.log2() as f32);
        let mut q = 0;
        let p = match mm.predict() {
            Some((pred, len)) => {
                if conditional {
                    let pp = count_prob(i, pred);
                    q = if pp < 0.1 { 0 } else if pp < 0.4 { 1 } else if pp < 0.8 { 2 } else { 3 };
                }
                let c = mm.confidence(len, q);
                // renormalised mixture: mass c moves to the predicted byte
                (1.0 - c) * pc + if pred == b { c } else { 0.0 }
            }
            None => pc,
        };
        mixed.push(-p.log2() as f32);
        mm.observe(b, q);
    }
    (base, mixed)
}

/// Mean of paired differences with a block-bootstrap 95% interval (blocks of `block` bytes).
pub fn paired_ci(base: &[f32], mixed: &[f32], block: usize, resamples: usize, seed: u64) -> (f64, f64, f64) {
    let n = base.len();
    let nb = n.div_ceil(block);
    let sums: Vec<(f64, usize)> = (0..nb).map(|k| {
        let r = k * block..((k + 1) * block).min(n);
        (r.clone().map(|i| (base[i] - mixed[i]) as f64).sum(), r.len())
    }).collect();
    let mean = sums.iter().map(|s| s.0).sum::<f64>() / n as f64;
    let mut rng = zerolm_tensor::Rng::new(seed);
    let mut ms: Vec<f64> = (0..resamples).map(|_| {
        let (mut s, mut c) = (0.0, 0usize);
        for _ in 0..nb {
            let (a, l) = sums[rng.below(nb)];
            s += a;
            c += l;
        }
        s / c as f64
    }).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (mean, ms[resamples * 25 / 1000], ms[resamples * 975 / 1000])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_a_repeat_and_learns_confidence() {
        let mut mm = MatchModel::new();
        let text = b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over";
        for &b in text.iter() {
            mm.observe(b, 0);
        }
        let (pred, len) = mm.predict().unwrap();
        assert_eq!(pred, b' ');
        assert!(len >= 20);
    }

    #[test]
    fn no_prediction_without_a_repeat() {
        let mut mm = MatchModel::new();
        for &b in b"abcdefghijklmnop" {
            mm.observe(b, 0);
        }
        assert!(mm.predict().is_none());
    }

    #[test]
    fn random_bytes_gain_nothing_and_repeats_gain_a_lot() {
        let mut r = zerolm_tensor::Rng::new(7);
        let rand: Vec<u8> = (0..4000).map(|_| r.below(256) as u8).collect();
        let uniform = |_: usize, _: u8| 1.0 / 256.0;
        let (a, b) = score_pair(&mut MatchModel::new(), &rand, false, uniform);
        let d: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) as f64).sum::<f64>() / 4000.0;
        assert!(d.abs() < 0.01, "{d}");
        let mut rep = rand[..1000].to_vec();
        rep.extend_from_slice(&rand[..1000]);
        let (a, b) = score_pair(&mut MatchModel::new(), &rep, false, uniform);
        let d: f64 = a[1000..].iter().zip(&b[1000..]).map(|(x, y)| (x - y) as f64).sum::<f64>() / 1000.0;
        assert!(d > 4.0, "{d}");
    }

    #[test]
    fn scoring_is_prequential() {
        // the confidence used for byte i never depends on byte i: changing it changes only later bits
        let t = b"abcdefabcdefabcdefabcdefXabcdef".to_vec();
        let mut t2 = t.clone();
        *t2.last_mut().unwrap() = b'Z';
        let u = |_: usize, _: u8| 1.0 / 256.0;
        let (_, m1) = score_pair(&mut MatchModel::new(), &t, true, u);
        let (_, m2) = score_pair(&mut MatchModel::new(), &t2, true, u);
        assert_eq!(m1[..m1.len() - 1], m2[..m2.len() - 1]);
    }
}
