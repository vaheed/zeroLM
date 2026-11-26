//! Real-text baselines (roadmap Phase 8): byte frequency, interpolated
//! Witten-Bell orders 3 and 5, PPM-C order 5, Kneser-Ney order 5 (sparse counts), and the online-learning
//! milestone's delta-rule predictor, all learning chronologically from the
//! same exposure bytes and scored on the hidden split's whole documents.
//!
//! `zerolm-realtext --root DATA_DIR [--manifest PATH] [--exposure N] [--hidden N] [--learner-limit N] [--control] [--control-limit N] [--all-sources] [--out DIR]`
//! With `--control`, also the online-backpropagation CONTROL (labelled, excluded from strict results).

use std::path::PathBuf;
use std::time::Instant;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_eval::bptt::{Bptt, Config as BpttConfig};
use zerolm_learning::online_predictor::{Constants, OnlinePredictor, Rule};
use zerolm_eval::levels::Levels;
use zerolm_eval::ngram::{Kn, Wb};

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let mpath = arg("--manifest").unwrap_or_else(|| "research/corpus/manifest.json".into());
    let manifest_text = std::fs::read_to_string(mpath).expect("manifest");
    let m = Manifest::parse(&manifest_text).expect("valid manifest");
    let problems = corpus::verify(&m, &root);
    assert!(problems.is_empty(), "{problems:?}");
    let n_exp: usize = arg("--exposure").map(|v| v.parse().expect("number")).unwrap_or(1_000_000);
    let n_hid: usize = arg("--hidden").map(|v| v.parse().expect("number")).unwrap_or(200_000);
    // Default: the first exposure source only (the Phase 8 table); `--all-sources` streams every source.
    let all = std::env::args().any(|a| a == "--all-sources");
    let first = m.exposure_order[0].clone();
    let role = |r| if all { corpus::role_stream(&m, &root, r) } else { corpus::source_role_stream(&m, &root, &first, r) };
    let exposure_all = role(corpus::Role::Exposure).expect("exposure");
    let hidden_all = role(corpus::Role::Hidden).expect("hidden");
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
        let dev_all = role(corpus::Role::Dev).expect("dev");
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
    let ngram_rows: Vec<(String, usize, f64)> = rows.clone();
    let l1: Vec<_> = points
        .iter()
        .filter_map(|p| Levels::l1_bar(&ngram_rows, *p).map(|bar| serde_json::json!({"exposure_bytes": p, "bar_bpb": bar})))
        .collect();
    let summary = zerolm_eval::levels::realtext_summary(
        &rows,
        &corpus::sha256_hex(exposure),
        &corpus::sha256_hex(hidden),
        hidden.len(),
        &corpus::sha256_hex(manifest_text.as_bytes()),
        &Levels::spec_sha256(),
        l1,
        std::env::args().any(|a| a == "--control"),
    );
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).expect("write");
}

