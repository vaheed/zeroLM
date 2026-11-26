//! `zerolm-headroom --root DATA [--points 100000,1000000,10000000] [--role dev|hidden] [--mixer v1|v2] [--out DIR]`
//!
//! Experiment N0 (`docs/experimental-design.md`): paired headroom of a match-model (induction) prior over
//! Kneser-Ney order 5 and PPM-C order 5, with (a) in-context history only and (b) exposure + in-context
//! history. Default role is `dev`; the hidden role is read only by the pre-registered final run.
use serde_json::{json, Value};
use std::path::PathBuf;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_eval::headroom::{paired_ci, score_pair, MatchModel};
use zerolm_eval::ngram_lean::{LeanKn, LeanWb};

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let manifest_text = std::fs::read_to_string("research/corpus/manifest.json").expect("manifest");
    let m = Manifest::parse(&manifest_text).expect("manifest");
    assert!(corpus::verify(&m, &root).is_empty());
    let points: Vec<usize> = arg("--points").map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or_else(|| vec![100_000, 1_000_000, 10_000_000]);
    let role = match arg("--role").as_deref() { Some("hidden") => corpus::Role::Hidden, _ => corpus::Role::Dev };
    let conditional = arg("--mixer").as_deref() == Some("v2");
    let top = *points.iter().max().unwrap();
    let mut exposure = Vec::new();
    for id in &m.exposure_order {
        exposure.extend(corpus::source_role_stream(&m, &root, id, corpus::Role::Exposure).unwrap());
        if exposure.len() >= top { break; }
    }
    exposure.truncate(top);
    let held: Vec<(&str, Vec<u8>)> = [("fiction", "gutenberg-books"), ("nonfiction", "gutenberg-nonfiction")].into_iter().map(|(n, id)| {
        let mut v = corpus::source_role_stream(&m, &root, id, role).unwrap();
        v.truncate(1_000_000);
        (n, v)
    }).collect();
    let mut rows: Vec<Value> = Vec::new();
    let mut kn = LeanKn::new(5);
    let mut wb = LeanWb::new(5);
    let mut done = 0;
    for &p in &points {
        for i in done..p {
            let h = &exposure[i.saturating_sub(4)..i];
            kn.observe(h, exposure[i]);
            wb.observe(h, exposure[i]);
        }
        done = p;
        for (name, text) in &held {
            for (cname, with_exposure) in [("in-context", false), ("exposure+in-context", true)] {
                for (model, is_kn) in [("kneser-ney-5", true), ("ppm-c-5", false)] {
                    let mut mm = MatchModel::new();
                    if with_exposure { mm.absorb(&exposure[..p]); }
                    let (base, mixed) = score_pair(&mut mm, text, conditional, |i, b| {
                        let h = &text[i.saturating_sub(4)..i];
                        if is_kn { kn.prob(h, b) } else { wb.ppm_prob(h, b) }
                    });
                    let n = text.len() as f64;
                    let (d, lo, hi) = paired_ci(&base, &mixed, 10_000, 1000, 20261127);
                    let b = base.iter().map(|x| *x as f64).sum::<f64>() / n;
                    let x = mixed.iter().map(|x| *x as f64).sum::<f64>() / n;
                    println!("{p:>9} {name:<10} {cname:<20} {model:<13} base {b:.3} +match {x:.3} gain {d:.4} [{lo:.4},{hi:.4}]");
                    rows.push(json!({"exposure_bytes": p, "role": name, "history": cname, "count_model": model, "heldout_bytes": text.len(), "base_bpb": b, "with_match_bpb": x, "gain_bpb": d, "gain_ci95": [lo, hi]}));
                }
            }
        }
    }
    let summary = json!({"report": "headroom", "split_role": if matches!(role, corpus::Role::Hidden) {"hidden"} else {"dev"}, "exposure_sha256": corpus::sha256_hex(&exposure), "manifest_sha256": corpus::sha256_hex(manifest_text.as_bytes()), "mixer": if conditional {"v2"} else {"v1"}, "min_len": zerolm_eval::headroom::MIN_LEN, "rows": rows,
        "scope_note": "Host-side reference match model mixed with frozen count models; paired block-bootstrap (10,000-byte blocks) over one held-out stream. Not a ZeroLM model result."});
    let dir = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/headroom".into()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
}
