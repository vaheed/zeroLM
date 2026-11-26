//! `zerolm-curves --root DATA [--points 100000,1000000,10000000,100000000] [--out DIR]`
//!
//! Real-text baseline learning curves up to 10^8 exposure bytes (roadmap Phase 8). One
//! chronological pass over the exposure stream (every source in `exposure_order`: fiction,
//! non-fiction, Simple English Wikipedia), scored on three held-out roles at each point with the
//! memory-lean count models of `zerolm_eval::ngram_lean` (tested equal to the Phase 8 reference
//! models): byte frequency, Witten-Bell orders 3 and 5, PPM-C order 5, Kneser-Ney order 5.
//! The first 10^7 bytes are the fiction stream, so those rows reproduce the Phase 8 table.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_eval::levels::Levels;
use zerolm_eval::ngram_lean::{LeanKn, LeanWb};

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let manifest_text = std::fs::read_to_string(arg("--manifest").unwrap_or_else(|| "research/corpus/manifest.json".into())).expect("manifest");
    let m = Manifest::parse(&manifest_text).expect("valid manifest");
    let problems = corpus::verify(&m, &root);
    assert!(problems.is_empty(), "{problems:?}");
    let points: Vec<usize> = arg("--points").map(|v| v.split(',').map(|x| x.parse().expect("number")).collect()).unwrap_or_else(|| vec![100_000, 1_000_000, 10_000_000, 100_000_000]);
    let top = *points.iter().max().unwrap();
    let mut exposure = Vec::new();
    for id in &m.exposure_order {
        exposure.extend(corpus::source_role_stream(&m, &root, id, corpus::Role::Exposure).expect("exposure"));
        if exposure.len() >= top {
            break;
        }
    }
    exposure.truncate(top);
    let cap = |mut v: Vec<u8>, n: usize| {
        v.truncate(n);
        v
    };
    let heldout: Vec<(&str, Vec<u8>)> = [("fiction", "gutenberg-books", usize::MAX), ("nonfiction", "gutenberg-nonfiction", 1_000_000), ("wikipedia", "simple-wikipedia", 2_000_000)]
        .into_iter()
        .map(|(name, id, n)| (name, cap(corpus::source_role_stream(&m, &root, id, corpus::Role::Hidden).expect("hidden"), n)))
        .collect();
    println!("exposure {} bytes (sha256 {}); held-out: {}", exposure.len(), corpus::sha256_hex(&exposure), heldout.iter().map(|(n, v)| format!("{n} {}", v.len())).collect::<Vec<_>>().join(", "));
    let points: Vec<usize> = points.into_iter().filter(|p| *p <= exposure.len()).collect();
    let mut rows: Vec<Value> = Vec::new();
    let mut resources: Vec<Value> = Vec::new();
    let add = |name: &str, p: usize, scores: Vec<f64>, rows: &mut Vec<Value>| {
        for ((role, text), bpb) in heldout.iter().zip(scores) {
            rows.push(json!({"baseline": name, "exposure_bytes": p, "heldout_role": role, "heldout_bytes": text.len(), "bpb": bpb}));
        }
    };
    // Pass 1: Witten-Bell (orders 1, 3, 5) and PPM-C from one set of tables.
    let t = Instant::now();
    let mut wb = LeanWb::new(5);
    let mut done = 0;
    for &p in &points {
        for i in done..p {
            wb.observe(&exposure[i.saturating_sub(4)..i], exposure[i]);
        }
        done = p;
        for (name, f) in [("byte-frequency", 1usize), ("witten-bell-3", 3), ("witten-bell-5", 5)] {
            let s = heldout.iter().map(|(_, h)| wb.bpb_wb(h, f)).collect();
            add(name, p, s, &mut rows);
        }
        let s = heldout.iter().map(|(_, h)| wb.bpb_ppm(h)).collect();
        add("ppm-c-5", p, s, &mut rows);
        println!("witten-bell and ppm-c at {p} bytes: {} table entries, {:.0}s", wb.entries(), t.elapsed().as_secs_f64());
    }
    resources.push(json!({"models": "witten-bell, ppm-c", "wall_s": t.elapsed().as_secs_f64(), "table_entries": wb.entries(), "peak_rss_kb": zerolm_eval::peak_rss_kb()}));
    drop(wb);
    // Pass 2: Kneser-Ney.
    let t = Instant::now();
    let mut kn = LeanKn::new(5);
    let mut done = 0;
    for &p in &points {
        for i in done..p {
            kn.observe(&exposure[i.saturating_sub(4)..i], exposure[i]);
        }
        done = p;
        let s = heldout.iter().map(|(_, h)| kn.bpb(h)).collect();
        add("kneser-ney-5", p, s, &mut rows);
        println!("kneser-ney at {p} bytes: {} table entries, {:.0}s", kn.entries(), t.elapsed().as_secs_f64());
    }
    resources.push(json!({"models": "kneser-ney", "wall_s": t.elapsed().as_secs_f64(), "table_entries": kn.entries(), "peak_rss_kb": zerolm_eval::peak_rss_kb()}));
    // L1 bars (better of PPM-C 5 and Kneser-Ney 5) and the Wikipedia BPB bars of L2 and L3.
    let levels = Levels::load(std::path::Path::new(".")).expect("levels");
    let (l2, l3) = (levels.0["levels"]["L2"]["wikipedia_heldout_bpb_max"].as_f64().unwrap(), levels.0["levels"]["L3"]["wikipedia_heldout_bpb_max"].as_f64().unwrap());
    let mut l1_bars = Vec::new();
    for &p in &points {
        for (role, _) in &heldout {
            let best = rows.iter().filter(|r| r["exposure_bytes"] == p && r["heldout_role"] == *role && (r["baseline"] == "ppm-c-5" || r["baseline"] == "kneser-ney-5")).map(|r| r["bpb"].as_f64().unwrap()).fold(f64::INFINITY, f64::min);
            l1_bars.push(json!({"exposure_bytes": p, "heldout_role": role, "bar_bpb": best}));
        }
    }
    let wiki_best = rows.iter().filter(|r| r["heldout_role"] == "wikipedia" && r["exposure_bytes"] == *points.last().unwrap()).map(|r| r["bpb"].as_f64().unwrap()).fold(f64::INFINITY, f64::min);
    let summary = json!({
        "report": "real-text-curves",
        "exposure_sha256": corpus::sha256_hex(&exposure), "exposure_bytes": exposure.len(),
        "manifest_sha256": corpus::sha256_hex(manifest_text.as_bytes()), "levels_sha256": Levels::spec_sha256(),
        "points": points, "rows": rows, "l1_bars": l1_bars, "resources": resources,
        "wikipedia_level_bars": {"l2_max_bpb": l2, "l3_max_bpb": l3, "best_count_baseline_bpb_at_last_point": wiki_best, "count_baseline_meets_l2": wiki_best <= l2, "count_baseline_meets_l3": wiki_best <= l3},
        "scope_note": "Count-based baselines only, one chronological pass, one split. The first 10^7 exposure bytes are fiction. The backpropagation CONTROL and the online-learning predictor are reported by zerolm-realtext up to 10^7 and 10^6 bytes, not here. No ZeroLM model is claimed to reach any level by this report."
    });
    let dir = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/real-text-curves".into()));
    std::fs::create_dir_all(&dir).expect("out dir");
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).expect("write");
    let mut brief = String::from("## Real-text baseline curves (held-out BPB)\n\n| baseline | exposure bytes | fiction | non-fiction | Wikipedia |\n|---|---|---|---|---|\n");
    for name in ["byte-frequency", "witten-bell-3", "witten-bell-5", "ppm-c-5", "kneser-ney-5"] {
        for &p in &points {
            let get = |role: &str| summary["rows"].as_array().unwrap().iter().find(|r| r["baseline"] == name && r["exposure_bytes"] == p && r["heldout_role"] == role).map(|r| r["bpb"].as_f64().unwrap()).unwrap_or(f64::NAN);
            brief += &format!("| {name} | {p} | {:.3} | {:.3} | {:.3} |\n", get("fiction"), get("nonfiction"), get("wikipedia"));
        }
    }
    brief += &format!("\nWikipedia BPB bars: L2 {l2}, L3 {l3}; best count baseline at the last point {wiki_best:.3}.\n");
    std::fs::write(dir.join("brief.md"), &brief).expect("write");
    print!("{brief}");
}
