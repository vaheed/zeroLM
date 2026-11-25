//! Real-text baselines (roadmap Phase 8): byte frequency and interpolated
//! Witten-Bell orders 3 and 5 (sparse counts), and the online-learning
//! milestone's delta-rule predictor, all learning chronologically from the
//! same exposure bytes and scored on the hidden split's whole documents.
//!
//! `zerolm-realtext --root DATA_DIR [--manifest PATH] [--exposure N] [--hidden N] [--out DIR]`
//! Not yet run here: PPM, Kneser-Ney, and the backpropagation CONTROL.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_learning::online_predictor::{Constants, OnlinePredictor, Rule};
use zerolm_tensor::fnv1a;

/// Interpolated Witten-Bell with sparse next-byte counts.
struct Wb {
    order: usize,
    counts: HashMap<u64, Vec<(u8, u32)>>,
}

fn key(k: usize, ctx: &[u8]) -> u64 {
    let mut v = Vec::with_capacity(ctx.len() + 1);
    v.push(k as u8);
    v.extend_from_slice(ctx);
    fnv1a(&v)
}

impl Wb {
    fn new(order: usize) -> Self {
        Wb { order, counts: HashMap::new() }
    }

    fn observe(&mut self, hist: &[u8], b: u8) {
        for k in 0..self.order.min(hist.len() + 1) {
            let e = self.counts.entry(key(k, &hist[hist.len() - k..])).or_default();
            match e.iter_mut().find(|(x, _)| *x == b) {
                Some((_, c)) => *c += 1,
                None => e.push((b, 1)),
            }
        }
    }

    fn prob(&self, hist: &[u8], b: u8) -> f64 {
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

    fn bpb(&self, bytes: &[u8]) -> f64 {
        let bits: f64 = bytes.iter().enumerate().map(|(i, &b)| -self.prob(&bytes[i.saturating_sub(self.order - 1)..i], b).log2()).sum();
        bits / bytes.len().max(1) as f64
    }

}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let mpath = arg("--manifest").unwrap_or_else(|| "research/corpus/manifest.json".into());
    let m = Manifest::parse(&std::fs::read_to_string(mpath).expect("manifest")).expect("valid manifest");
    let problems = corpus::verify(&m, &root);
    assert!(problems.is_empty(), "{problems:?}");
    let n_exp: usize = arg("--exposure").map(|v| v.parse().expect("number")).unwrap_or(1_000_000);
    let n_hid: usize = arg("--hidden").map(|v| v.parse().expect("number")).unwrap_or(200_000);
    let exposure_all = corpus::role_stream(&m, &root, corpus::Role::Exposure).expect("exposure");
    let hidden_all = corpus::role_stream(&m, &root, corpus::Role::Hidden).expect("hidden");
    let exposure = &exposure_all[..n_exp.min(exposure_all.len())];
    let hidden = &hidden_all[..n_hid.min(hidden_all.len())];
    println!("exposure {} of {} bytes (sha256 {}), hidden {} of {} bytes", exposure.len(), exposure_all.len(), corpus::sha256_hex(exposure), hidden.len(), hidden_all.len());
    let points: Vec<usize> = [100_000usize, 1_000_000, 10_000_000, 100_000_000].into_iter().filter(|p| *p <= exposure.len()).chain(std::iter::once(exposure.len())).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let mut rows = Vec::new();
    let t = Instant::now();
    for order in [1usize, 3, 5] {
        let mut wb = Wb::new(order);
        let mut done = 0;
        for &p in &points {
            wb.learn_range(exposure, done, p);
            done = p;
            rows.push((format!("witten-bell-{order}"), p, wb.bpb(hidden)));
        }
    }
    println!("n-gram baselines {:.1}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let mut m_delta = OnlinePredictor::new(1, Rule::Delta, Constants::default(), "real-text-exposure");
    let mut done = 0;
    for &p in &points {
        m_delta.observe_all(&exposure[done..p]).expect("finite");
        done = p;
        rows.push(("online-delta (phase 4 learner)".into(), p, m_delta.evaluate(hidden).expect("finite").bpb));
    }
    println!("online learner {:.1}s", t.elapsed().as_secs_f64());
    let mut out = String::from("## Real-text baselines (held-out BPB on hidden whole documents)\n\n| baseline | exposure bytes | hidden BPB |\n|---|---|---|\n");
    for (n, p, b) in &rows {
        out += &format!("| {n} | {p} | {b:.3} |\n");
    }
    out += "\nNot yet run: PPM, Kneser-Ney order 5, and the online-backpropagation CONTROL.\n";
    print!("{out}");
    let dir = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/real-text".into()));
    std::fs::create_dir_all(&dir).expect("out dir");
    std::fs::write(dir.join("brief.md"), &out).expect("write");
    let json: Vec<_> = rows.iter().map(|(n, p, b)| serde_json::json!({"baseline": n, "exposure_bytes": p, "hidden_bpb": b})).collect();
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&serde_json::json!({"report": "real-text", "exposure_sha256": corpus::sha256_hex(exposure), "rows": json})).unwrap()).expect("write");
}

impl Wb {
    fn learn_range(&mut self, bytes: &[u8], from: usize, to: usize) {
        for i in from..to {
            self.observe(&bytes[i.saturating_sub(self.order - 1)..i], bytes[i]);
        }
    }
}
