//! Count-based real-text baselines (roadmap Phase 8): interpolated Witten-Bell,
//! PPM-C with exclusion, and interpolated Kneser-Ney, all over bytes with sparse counts.

use std::collections::HashMap;
use zerolm_tensor::fnv1a;

/// Interpolated Witten-Bell with sparse next-byte counts.
pub struct Wb {
    order: usize,
    counts: HashMap<u64, Vec<(u8, u32)>>,
}

pub fn key(k: usize, ctx: &[u8]) -> u64 {
    let mut v = Vec::with_capacity(ctx.len() + 1);
    v.push(k as u8);
    v.extend_from_slice(ctx);
    fnv1a(&v)
}

impl Wb {
    pub fn new(order: usize) -> Self {
        Wb { order, counts: HashMap::new() }
    }

    pub fn observe(&mut self, hist: &[u8], b: u8) {
        for k in 0..self.order.min(hist.len() + 1) {
            let e = self.counts.entry(key(k, &hist[hist.len() - k..])).or_default();
            match e.iter_mut().find(|(x, _)| *x == b) {
                Some((_, c)) => *c += 1,
                None => e.push((b, 1)),
            }
        }
    }

    pub fn prob(&self, hist: &[u8], b: u8) -> f64 {
        let mut p = 1.0 / 256.0;
        for k in 0..self.order.min(hist.len() + 1) {
            let Some(c) = self.counts.get(&key(k, &hist[hist.len() - k..])) else { break };
            let total: u64 = c.iter().map(|x| x.1 as u64).sum();
            let lambda = total as f64 / (total as f64 + c.len() as f64);
            let n = c.iter().find(|x| x.0 == b).map(|x| x.1).unwrap_or(0) as f64;
            p = lambda * n / total as f64 + (1.0 - lambda) * p;
        }
        p
    }

    /// PPM-C with exclusion: escape through shorter contexts, never counting a
    /// byte already ruled out by a longer context; the last resort is uniform
    /// over the bytes not yet excluded.
    pub fn ppm_prob(&self, hist: &[u8], b: u8) -> f64 {
        let mut excluded = [false; 256];
        let mut escape = 1.0;
        for k in (0..self.order.min(hist.len() + 1)).rev() {
            let Some(c) = self.counts.get(&key(k, &hist[hist.len() - k..])) else { continue };
            let live: Vec<&(u8, u32)> = c.iter().filter(|x| !excluded[x.0 as usize]).collect();
            if live.is_empty() {
                continue;
            }
            let total: f64 = live.iter().map(|x| x.1 as f64).sum();
            let d = live.len() as f64;
            if let Some(x) = live.iter().find(|x| x.0 == b) {
                return escape * x.1 as f64 / (total + d);
            }
            escape *= d / (total + d);
            for x in live {
                excluded[x.0 as usize] = true;
            }
        }
        let left = 256 - excluded.iter().filter(|e| **e).count();
        escape / left as f64
    }

    pub fn bpb(&self, bytes: &[u8], ppm: bool) -> f64 {
        let bits: f64 = bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let h = &bytes[i.saturating_sub(self.order - 1)..i];
                -(if ppm { self.ppm_prob(h, b) } else { self.prob(h, b) }).log2()
            })
            .sum();
        bits / bytes.len().max(1) as f64
    }

}

/// Interpolated Kneser-Ney (fixed discount 0.75): the highest order uses raw
/// counts, lower orders use continuation counts (distinct left extensions).
pub struct Kn {
    order: usize,
    raw: HashMap<u64, Vec<(u8, u32)>>,
    cont: Vec<HashMap<u64, Vec<(u8, u32)>>>,
    seen: std::collections::HashSet<u64>,
}

pub const DISCOUNT: f64 = 0.75;

fn bump(m: &mut HashMap<u64, Vec<(u8, u32)>>, k: u64, b: u8) {
    let e = m.entry(k).or_default();
    match e.iter_mut().find(|(x, _)| *x == b) {
        Some((_, c)) => *c += 1,
        None => e.push((b, 1)),
    }
}

impl Kn {
    pub fn new(order: usize) -> Self {
        Kn { order, raw: HashMap::new(), cont: (0..order - 1).map(|_| HashMap::new()).collect(), seen: Default::default() }
    }

    pub fn observe(&mut self, hist: &[u8], b: u8) {
        let top = self.order - 1;
        if hist.len() < top {
            return;
        }
        bump(&mut self.raw, key(top, &hist[hist.len() - top..]), b);
        for j in 0..top {
            let mut g = key(j + 1, &hist[hist.len() - j - 1..]).to_le_bytes().to_vec();
            g.push(b);
            if self.seen.insert(fnv1a(&g)) {
                bump(&mut self.cont[j], key(j, &hist[hist.len() - j..]), b);
            }
        }
    }

    pub fn prob(&self, hist: &[u8], b: u8) -> f64 {
        let mut p = 1.0 / 256.0;
        for j in 0..self.order.min(hist.len() + 1) {
            let table = if j == self.order - 1 { &self.raw } else { &self.cont[j] };
            let Some(c) = table.get(&key(j, &hist[hist.len() - j..])) else { continue };
            let total: f64 = c.iter().map(|x| x.1 as f64).sum();
            let n = c.iter().find(|x| x.0 == b).map(|x| x.1 as f64).unwrap_or(0.0);
            p = (n - DISCOUNT).max(0.0) / total + DISCOUNT * c.len() as f64 / total * p;
        }
        p
    }

    pub fn bpb(&self, bytes: &[u8]) -> f64 {
        let bits: f64 = bytes.iter().enumerate().map(|(i, &b)| -self.prob(&bytes[i.saturating_sub(self.order - 1)..i], b).log2()).sum();
        bits / bytes.len().max(1) as f64
    }
}

impl Wb {
    pub fn learn_range(&mut self, bytes: &[u8], from: usize, to: usize) {
        for i in from..to {
            self.observe(&bytes[i.saturating_sub(self.order - 1)..i], bytes[i]);
        }
    }
}
