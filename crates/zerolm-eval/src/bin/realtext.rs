//! Real-text baselines (roadmap Phase 8): byte frequency, interpolated
//! Witten-Bell orders 3 and 5, PPM-C order 5, Kneser-Ney order 5 (sparse counts), and the online-learning
//! milestone's delta-rule predictor, all learning chronologically from the
//! same exposure bytes and scored on the hidden split's whole documents.
//!
//! `zerolm-realtext --root DATA_DIR [--manifest PATH] [--exposure N] [--hidden N] [--learner-limit N] [--control] [--control-limit N] [--out DIR]`
//! With `--control`, also the online-backpropagation CONTROL (labelled, excluded from strict results).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_eval::bptt::{Bptt, Config as BpttConfig};
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

    /// PPM-C with exclusion: escape through shorter contexts, never counting a
    /// byte already ruled out by a longer context; the last resort is uniform
    /// over the bytes not yet excluded.
    fn ppm_prob(&self, hist: &[u8], b: u8) -> f64 {
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

    fn bpb(&self, bytes: &[u8], ppm: bool) -> f64 {
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
struct Kn {
    order: usize,
    raw: HashMap<u64, Vec<(u8, u32)>>,
    cont: Vec<HashMap<u64, Vec<(u8, u32)>>>,
    seen: std::collections::HashSet<u64>,
}

const DISCOUNT: f64 = 0.75;

fn bump(m: &mut HashMap<u64, Vec<(u8, u32)>>, k: u64, b: u8) {
    let e = m.entry(k).or_default();
    match e.iter_mut().find(|(x, _)| *x == b) {
        Some((_, c)) => *c += 1,
        None => e.push((b, 1)),
    }
}

impl Kn {
    fn new(order: usize) -> Self {
        Kn { order, raw: HashMap::new(), cont: (0..order - 1).map(|_| HashMap::new()).collect(), seen: Default::default() }
    }

    fn observe(&mut self, hist: &[u8], b: u8) {
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

    fn prob(&self, hist: &[u8], b: u8) -> f64 {
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
            rows.push((format!("witten-bell-{order}"), p, wb.bpb(hidden, false)));
            if order == 5 {
                rows.push(("ppm-c-5".into(), p, wb.bpb(hidden, true)));
            }
        }
    }
    let mut kn = Kn::new(5);
    let mut done = 0;
    for &p in &points {
        for i in done..p {
            kn.observe(&exposure[i.saturating_sub(4)..i], exposure[i]);
        }
        done = p;
        rows.push(("kneser-ney-5".into(), p, kn.bpb(hidden)));
    }
    println!("n-gram baselines {:.1}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let mut m_delta = OnlinePredictor::new(1, Rule::Delta, Constants::default(), "real-text-exposure");
    let learner_limit: usize = arg("--learner-limit").map(|v| v.parse().expect("number")).unwrap_or(1_000_000);
    let mut done = 0;
    for &p in points.iter().filter(|p| **p <= learner_limit) {
        m_delta.observe_all(&exposure[done..p]).expect("finite");
        done = p;
        rows.push(("online-delta (phase 4 learner)".into(), p, m_delta.evaluate(hidden).expect("finite").bpb));
    }
    println!("online learner {:.1}s", t.elapsed().as_secs_f64());
    if std::env::args().any(|a| a == "--control") {
        // Learning rate chosen on the dev role only (first 200k exposure bytes in, first 200k dev bytes scored).
        let dev_all = corpus::role_stream(&m, &root, corpus::Role::Dev).expect("dev");
        let dev = &dev_all[..dev_all.len().min(200_000)];
        let pilot = &exposure[..exposure.len().min(200_000)];
        let mut best = (f64::INFINITY, 3e-3f32);
        for lr in [1e-3f32, 3e-3, 1e-2] {
            let t = Instant::now();
            let mut c = Bptt::new(9001, BpttConfig { lr, ..Default::default() });
            c.observe_all(pilot);
            let b = c.evaluate(dev);
            println!("control pilot lr {lr}: dev BPB {b:.3} ({:.0}s)", t.elapsed().as_secs_f64());
            if b < best.0 {
                best = (b, lr);
            }
        }
        let control_limit: usize = arg("--control-limit").map(|v| v.parse().expect("number")).unwrap_or(1_000_000);
        let t = Instant::now();
        let mut c = Bptt::new(1, BpttConfig { lr: best.1, ..Default::default() });
        let mut done = 0;
        for &p in points.iter().filter(|p| **p <= control_limit) {
            c.observe_all(&exposure[done..p]);
            done = p;
            rows.push((format!("CONTROL backprop-through-time GRU (lr {})", best.1), p, c.evaluate(hidden)));
            println!("control prequential BPB over the first {p} bytes: {:.3}", c.prequential_bpb());
        }
        println!("control {:.1}s", t.elapsed().as_secs_f64());
    }
    let mut out = String::from("## Real-text baselines (held-out BPB on hidden whole documents)\n\n| baseline | exposure bytes | hidden BPB |\n|---|---|---|\n");
    for (n, p, b) in &rows {
        out += &format!("| {n} | {p} | {b:.3} |\n");
    }
    out += "\nThe CONTROL row (if present) is online backpropagation through time, excluded from strict results.\n";
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
