//! Probes and accounting for the gated-delta memory experiment (`docs/gdm-experiment.md`).
//!
//! The recall probe asks one question: after the model has read some key-value lines, can it say which value
//! belongs to a key, as measured by the bits it needs for the first letter of the value? It separates what lives
//! in the recurrent state `h`, in the persistent fast-weight state `S`, and in the learned weights:
//!
//! * `full`: state as left after the lines and the unrelated text.
//! * `reset_h`: `h` and the byte history zeroed, `S` kept (what the fast-weight memory alone retains).
//! * `reset_s`: `S` zeroed, `h` kept.
//! * `restart`: everything zeroed (a process restart that restores only weights).
//! * `weights`: the weights were updated by online learning while reading the lines and the unrelated text
//!   (a throw-away copy restored from a snapshot), then the state is zeroed: what the weights retained.
//!
//! All conditions score the same queries; the control query uses a key that never appeared. Keys are three random
//! lowercase letters, values come from a fixed pool of 16 three-letter pseudo-words with distinct first letters (known words in
//! unfamiliar pairings), so the answer cannot be guessed from a key. Queries are scored with frozen weights and a
//! copy of the state per query, so no query or answer reaches the weights or another query.
use crate::bptt::{snapshot::CarryState, Bptt};
use zerolm_tensor::Rng;

pub const POOL: usize = 16;

fn pool() -> Vec<Vec<u8>> {
    let mut r = Rng::from_tag(0, "gdm-probe-pool-v1");
    (0..POOL)
        .map(|i| {
            let mut w = vec![b'a' + i as u8];
            for _ in 0..2 {
                w.push(b'a' + r.below(26) as u8);
            }
            w
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Trial {
    /// The presented lines, `key value.\n` each.
    pub lines: Vec<u8>,
    pub keys: Vec<[u8; 3]>,
    pub control_keys: Vec<[u8; 3]>,
    /// First letter of each fact's value.
    pub answers: Vec<u8>,
}

/// One deterministic trial: `facts` lines with keys of pairwise distinct last two letters.
pub fn make_trial(seed: u64, facts: usize) -> Trial {
    let words = pool();
    let mut r = Rng::from_tag(seed, "gdm-probe-trial");
    let mut used: Vec<[u8; 2]> = Vec::new();
    let fresh_key = |r: &mut Rng, used: &mut Vec<[u8; 2]>| loop {
        let k = [b'a' + r.below(26) as u8, b'a' + r.below(26) as u8, b'a' + r.below(26) as u8];
        let tail = [k[1], k[2]];
        if !used.contains(&tail) {
            used.push(tail);
            return k;
        }
    };
    let keys: Vec<[u8; 3]> = (0..facts).map(|_| fresh_key(&mut r, &mut used)).collect();
    let control_keys: Vec<[u8; 3]> = (0..facts).map(|_| fresh_key(&mut r, &mut used)).collect();
    let mut lines = vec![b'\n'];
    let mut answers = Vec::new();
    for k in &keys {
        let w = &words[r.below(POOL)];
        lines.extend_from_slice(k);
        lines.push(b' ');
        lines.extend_from_slice(w);
        lines.extend_from_slice(b".\n");
        answers.push(w[0]);
    }
    Trial { lines, keys, control_keys, answers }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Tally {
    pub shown_bits: f64,
    pub control_bits: f64,
    pub shown_hits: usize,
    pub control_hits: usize,
    pub n: usize,
}

impl Tally {
    /// Share of the control's surprisal removed by having seen the line: 1 is perfect recall, 0 is none.
    pub fn recall(&self) -> f64 {
        1.0 - self.shown_bits / self.control_bits.max(1e-9)
    }
    pub fn add(&mut self, o: &Tally) {
        self.shown_bits += o.shown_bits;
        self.control_bits += o.control_bits;
        self.shown_hits += o.shown_hits;
        self.control_hits += o.control_hits;
        self.n += o.n;
    }
}

/// Score every query of `t` from `st` (a copy per query); frozen weights.
pub fn score_queries(m: &Bptt, st: &CarryState, t: &Trial) -> Tally {
    let mut tally = Tally::default();
    for i in 0..t.keys.len() {
        for (shown, key) in [(true, t.keys[i]), (false, t.control_keys[i])] {
            let mut q = key.to_vec();
            q.push(b' ');
            q.push(t.answers[i]);
            let out = m.probs_from(st, &q);
            let (bits, arg) = out[out.len() - 1];
            if shown {
                tally.shown_bits += bits as f64;
                tally.shown_hits += (arg == t.answers[i]) as usize;
            } else {
                tally.control_bits += bits as f64;
                tally.control_hits += (arg == t.answers[i]) as usize;
            }
        }
        tally.n += 1;
    }
    tally
}

pub const CONDITIONS: [&str; 5] = ["full", "reset_h", "reset_s", "restart", "weights"];

/// Run the recall probe at one distractor length. `distractor` supplies unrelated real text (consecutive
/// windows are used per trial). `make` builds a model of the same genome as `m`; `snap` is `m`'s snapshot,
/// from which the `weights` condition restores a throw-away copy per trial.
pub fn recall_probe(m: &Bptt, make: &dyn Fn() -> Bptt, snap: &[u8], distractor: &[u8], dlen: usize, trials: usize, facts: usize, seed: u64) -> Vec<(&'static str, Tally)> {
    let mut tallies = vec![Tally::default(); CONDITIONS.len()];
    for ti in 0..trials {
        let t = make_trial(seed.wrapping_mul(1_000_003).wrapping_add(ti as u64), facts);
        let mut pres = t.lines.clone();
        if dlen > 0 {
            let start = (ti * 7919 * 13) % distractor.len().saturating_sub(dlen + 1).max(1);
            pres.extend_from_slice(&distractor[start..start + dlen]);
        }
        pres.push(b'\n');
        let full = m.advance(&m.fresh_state(), &pres);
        let fresh = m.fresh_state();
        let reset_h = CarryState { h: fresh.h.clone(), fast: full.fast.clone(), tail: fresh.tail.clone() };
        let reset_s = CarryState { h: full.h.clone(), fast: fresh.fast.clone(), tail: full.tail.clone() };
        let before = m.weights_sha256();
        for (c, st) in [&full, &reset_h, &reset_s, &fresh].into_iter().enumerate() {
            let tl = score_queries(m, st, &t);
            tallies[c].add(&tl);
        }
        assert_eq!(before, m.weights_sha256(), "a probe must not change the weights");
        // Weights condition: learn the lines and the unrelated text online in a throw-away copy.
        let mut copy = make();
        copy.restore(snap).expect("snapshot of the same genome");
        copy.observe_all(&pres);
        let tl = score_queries(&copy, &copy.fresh_state(), &t);
        tallies[4].add(&tl);
    }
    CONDITIONS.iter().cloned().zip(tallies).collect()
}

/// Approximate multiply-add floating-point operations per byte for the GRU part (forward, backward = 2x forward).
pub fn gru_flops_per_byte(e: usize, d: usize) -> f64 {
    2.0 * (3 * d * e + 3 * d * d) as f64 + 2.0 * (256 * (d + e)) as f64 + 8.0 * d as f64
}

/// The same for the gated-delta memory path: two bundles, the delta update, the read, the decoder.
pub fn gdm_flops_per_byte(r: usize, lags: usize) -> f64 {
    (2 * 2 * lags * r + 2 * r * r + 4 * r * r + 2 * r * r + 2 * 256 * r) as f64
}

/// `n` windows of `each` bytes spread evenly over `text`.
pub fn spaced(text: &[u8], n: usize, each: usize) -> Vec<u8> {
    if text.len() <= n * each {
        return text.to_vec();
    }
    let gap = (text.len() - each) / (n - 1).max(1);
    (0..n).flat_map(|i| text[i * gap..i * gap + each].to_vec()).collect()
}

/// Peak resident set size of this process in kB (Linux), 0 if unavailable.
pub fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trials_are_deterministic_and_keys_unique() {
        let (a, b) = (make_trial(5, 8), make_trial(5, 8));
        assert_eq!(a.lines, b.lines);
        let mut tails: Vec<[u8; 2]> = a.keys.iter().chain(&a.control_keys).map(|k| [k[1], k[2]]).collect();
        tails.sort();
        tails.dedup();
        assert_eq!(tails.len(), 16);
        assert_ne!(make_trial(6, 8).lines, a.lines);
    }

    #[test]
    fn spaced_windows() {
        let t: Vec<u8> = (0..=255).cycle().take(1000).collect();
        assert_eq!(spaced(&t, 4, 50).len(), 200);
        assert_eq!(spaced(&t[..100], 4, 50).len(), 100);
    }
}
