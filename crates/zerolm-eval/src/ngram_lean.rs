//! Memory-lean versions of the Phase 8 count-based baselines, for streams of 10^8 bytes.
//!
//! The reference models in [`crate::ngram`] keep one follower list per context in a hash map,
//! which needs tens of gigabytes at 10^8 bytes. These keep the same counts in open-addressing
//! tables keyed by 64-bit hashes and give the same probabilities (a unit test checks this
//! against the reference on generated text):
//!
//! - [`LeanWb`]: interpolated Witten-Bell for every order up to `order`, and PPM-C with
//!   exclusion, from one set of tables (a lower order uses only the shorter contexts);
//! - [`LeanKn`]: interpolated Kneser-Ney (fixed discount 0.75), highest order on raw counts,
//!   lower orders on continuation counts.
//!
//! Followers of a context are not stored as lists: PPM enumerates the 256 byte values and
//! stops once it has found all of the context's distinct followers.

/// Open-addressing map from a non-zero 64-bit key to a `u32` count.
pub struct Counts {
    keys: Vec<u64>,
    vals: Vec<u32>,
    len: usize,
}

const EMPTY: u64 = 0;

impl Counts {
    pub fn new() -> Self {
        Counts { keys: vec![EMPTY; 1 << 16], vals: vec![0; 1 << 16], len: 0 }
    }

    fn slot(&self, key: u64) -> usize {
        let mask = self.keys.len() - 1;
        let mut i = (key >> 7) as usize & mask;
        while self.keys[i] != EMPTY && self.keys[i] != key {
            i = (i + 1) & mask;
        }
        i
    }

    pub fn get(&self, key: u64) -> u32 {
        let i = self.slot(key);
        if self.keys[i] == key { self.vals[i] } else { 0 }
    }

    /// Add one; returns the count before the increment.
    pub fn bump(&mut self, key: u64) -> u32 {
        if (self.len + 1) * 4 > self.keys.len() * 3 {
            self.grow();
        }
        let i = self.slot(key);
        if self.keys[i] == EMPTY {
            self.keys[i] = key;
            self.len += 1;
        }
        let before = self.vals[i];
        self.vals[i] = before + 1;
        before
    }

    fn grow(&mut self) {
        let n = self.keys.len() * 2;
        let (ok, ov) = (std::mem::replace(&mut self.keys, vec![EMPTY; n]), std::mem::replace(&mut self.vals, vec![0; n]));
        for (k, v) in ok.into_iter().zip(ov) {
            if k != EMPTY {
                let i = self.slot(k);
                self.keys[i] = k;
                self.vals[i] = v;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for Counts {
    fn default() -> Self {
        Counts::new()
    }
}

/// 64-bit hash of `tag`, the context bytes, and optionally one more byte; never zero.
fn hash(tag: u8, ctx: &[u8], extra: Option<u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (tag as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h = (h ^ tag as u64).wrapping_mul(0x0000_0100_0000_01b3);
    for &b in ctx {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(e) = extra {
        h = (h ^ 0x100 ^ e as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= h >> 32;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 29;
    h | 1
}

/// Per-context summary: total count and number of distinct followers.
struct CtxSummary {
    total: Counts,
    distinct: Counts,
}

impl CtxSummary {
    fn new() -> Self {
        CtxSummary { total: Counts::new(), distinct: Counts::new() }
    }
}

/// Witten-Bell and PPM-C from one set of tables; `order` is the longest model (contexts of up to `order - 1` bytes).
pub struct LeanWb {
    order: usize,
    pairs: Counts,
    ctx: CtxSummary,
}

impl LeanWb {
    pub fn new(order: usize) -> Self {
        LeanWb { order, pairs: Counts::new(), ctx: CtxSummary::new() }
    }

    pub fn observe(&mut self, hist: &[u8], b: u8) {
        for k in 0..self.order.min(hist.len() + 1) {
            let c = &hist[hist.len() - k..];
            let ck = hash(k as u8, c, None);
            if self.pairs.bump(hash(k as u8, c, Some(b))) == 0 {
                self.ctx.distinct.bump(ck);
            }
            self.ctx.total.bump(ck);
        }
    }

    /// Interpolated Witten-Bell of order `order` (at most this model's order).
    pub fn prob(&self, hist: &[u8], b: u8, order: usize) -> f64 {
        let mut p = 1.0 / 256.0;
        for k in 0..order.min(hist.len() + 1) {
            let c = &hist[hist.len() - k..];
            let ck = hash(k as u8, c, None);
            let total = self.ctx.total.get(ck);
            if total == 0 {
                break;
            }
            let distinct = self.ctx.distinct.get(ck);
            let (total, distinct) = (total as f64, distinct as f64);
            let lambda = total / (total + distinct);
            let n = self.pairs.get(hash(k as u8, c, Some(b))) as f64;
            p = lambda * n / total + (1.0 - lambda) * p;
        }
        p
    }

    /// PPM-C with exclusion at this model's order.
    pub fn ppm_prob(&self, hist: &[u8], b: u8) -> f64 {
        let mut excluded = [false; 256];
        let mut escape = 1.0;
        for k in (0..self.order.min(hist.len() + 1)).rev() {
            let c = &hist[hist.len() - k..];
            let ck = hash(k as u8, c, None);
            let distinct_all = self.ctx.distinct.get(ck) as usize;
            if distinct_all == 0 {
                continue;
            }
            let (mut found, mut live, mut total_live, mut hit) = (0usize, 0usize, 0.0f64, 0u32);
            for x in 0..=255u8 {
                if found == distinct_all {
                    break;
                }
                let n = self.pairs.get(hash(k as u8, c, Some(x)));
                if n == 0 {
                    continue;
                }
                found += 1;
                if excluded[x as usize] {
                    continue;
                }
                live += 1;
                total_live += n as f64;
                if x == b {
                    hit = n;
                }
            }
            if live == 0 {
                continue;
            }
            let d = live as f64;
            if hit > 0 {
                return escape * hit as f64 / (total_live + d);
            }
            escape *= d / (total_live + d);
            for x in 0..=255u8 {
                if self.pairs.get(hash(k as u8, c, Some(x))) > 0 {
                    excluded[x as usize] = true;
                }
            }
        }
        let left = 256 - excluded.iter().filter(|e| **e).count();
        escape / left as f64
    }

    pub fn entries(&self) -> usize {
        self.pairs.len() + self.ctx.total.len() + self.ctx.distinct.len()
    }

    pub fn bpb_wb(&self, bytes: &[u8], order: usize) -> f64 {
        self.bpb(bytes, |h, b| self.prob(h, b, order))
    }

    pub fn bpb_ppm(&self, bytes: &[u8]) -> f64 {
        self.bpb(bytes, |h, b| self.ppm_prob(h, b))
    }

    fn bpb(&self, bytes: &[u8], f: impl Fn(&[u8], u8) -> f64) -> f64 {
        let bits: f64 = bytes.iter().enumerate().map(|(i, &b)| -f(&bytes[i.saturating_sub(self.order - 1)..i], b).log2()).sum();
        bits / bytes.len().max(1) as f64
    }
}

const DISCOUNT: f64 = 0.75;

/// Interpolated Kneser-Ney, fixed discount 0.75.
pub struct LeanKn {
    order: usize,
    /// Raw counts at the highest order, continuation counts below.
    pairs: Counts,
    ctx: CtxSummary,
    /// Distinct (j+1)-byte-context-and-byte occurrences already counted into the continuation tables.
    seen: Counts,
}

impl LeanKn {
    pub fn new(order: usize) -> Self {
        LeanKn { order, pairs: Counts::new(), ctx: CtxSummary::new(), seen: Counts::new() }
    }

    fn bump(&mut self, j: usize, c: &[u8], b: u8) {
        let ck = hash(j as u8, c, None);
        if self.pairs.bump(hash(j as u8, c, Some(b))) == 0 {
            self.ctx.distinct.bump(ck);
        }
        self.ctx.total.bump(ck);
    }

    pub fn observe(&mut self, hist: &[u8], b: u8) {
        let top = self.order - 1;
        if hist.len() < top {
            return;
        }
        self.bump(top, &hist[hist.len() - top..], b);
        for j in 0..top {
            // `seen` is keyed by the (j+1)-byte context and the byte, with tags apart from the pair tables.
            if self.seen.bump(hash(100 + (j + 1) as u8, &hist[hist.len() - j - 1..], Some(b))) == 0 {
                self.bump(j, &hist[hist.len() - j..], b);
            }
        }
    }

    pub fn prob(&self, hist: &[u8], b: u8) -> f64 {
        let mut p = 1.0 / 256.0;
        for j in 0..self.order.min(hist.len() + 1) {
            let c = &hist[hist.len() - j..];
            let ck = hash(j as u8, c, None);
            let total = self.ctx.total.get(ck);
            if total == 0 {
                continue;
            }
            let (total, distinct) = (total as f64, self.ctx.distinct.get(ck) as f64);
            let n = self.pairs.get(hash(j as u8, c, Some(b))) as f64;
            p = (n - DISCOUNT).max(0.0) / total + DISCOUNT * distinct / total * p;
        }
        p
    }

    pub fn entries(&self) -> usize {
        self.pairs.len() + self.ctx.total.len() + self.ctx.distinct.len() + self.seen.len()
    }

    pub fn bpb(&self, bytes: &[u8]) -> f64 {
        let bits: f64 = bytes.iter().enumerate().map(|(i, &b)| -self.prob(&bytes[i.saturating_sub(self.order - 1)..i], b).log2()).sum();
        bits / bytes.len().max(1) as f64
    }
}
