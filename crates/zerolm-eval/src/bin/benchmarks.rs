//! `zerolm-benchmarks --root DATA_DIR [--manifest PATH] [--exposure N] [--out DIR]`
//!
//! BLiMP and LAMBADA (evaluation-only, hashed in the manifest) for the count-based
//! baselines after one chronological pass over the first N exposure bytes. Writes
//! `reports/benchmarks/{summary.json,brief.md}` (schema
//! `research/schemas/benchmarks-report.schema.json`).

use std::path::PathBuf;
use zerolm_data::{corpus, evalsets, manifest::{EvalKind, Manifest}};
use zerolm_eval::benchmarks::{blimp, lambada, ByteLm, KneserNey, PpmC, WittenBell};
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
    let eval = |kind: EvalKind| {
        let e = m.evaluation_sets.iter().find(|e| e.kind == kind).unwrap_or_else(|| panic!("manifest has no {kind:?} evaluation set"));
        (e.sha256.clone(), std::fs::read(root.join(&e.path)).expect("evaluation file"))
    };
    let (blimp_sha, blimp_bytes) = eval(EvalKind::Blimp);
    let (lambada_sha, lambada_bytes) = eval(EvalKind::Lambada);
    let pairs = evalsets::parse_blimp(&blimp_bytes).expect("blimp");
    let (items, skipped) = evalsets::parse_lambada(&lambada_bytes).expect("lambada");
    let n_exp: usize = arg("--exposure").map(|v| v.parse().expect("number")).unwrap_or(1_000_000);
    let exposure_all = corpus::role_stream(&m, &root, corpus::Role::Exposure).expect("exposure");
    let exposure = &exposure_all[..n_exp.min(exposure_all.len())];
    println!("exposure {} bytes; {} BLiMP pairs; {} LAMBADA items ({skipped} skipped)", exposure.len(), pairs.len(), items.len());
    let (mut wb, mut kn) = (Wb::new(5), Kn::new(5));
    for i in 0..exposure.len() {
        let h = &exposure[i.saturating_sub(4)..i];
        wb.observe(h, exposure[i]);
        kn.observe(h, exposure[i]);
    }
    let models: Vec<(&str, Box<dyn ByteLm + '_>)> = vec![("witten-bell-5", Box::new(WittenBell(&wb))), ("ppm-c-5", Box::new(PpmC(&wb))), ("kneser-ney-5", Box::new(KneserNey(&kn)))];
    let levels = Levels::load(std::path::Path::new(".")).expect("levels spec");
    let (blimp_l2, blimp_l3) = (levels.0["levels"]["L2"]["blimp_min"].as_f64().unwrap(), levels.0["levels"]["L3"]["blimp_min"].as_f64().unwrap());
    let lambada_l3 = levels.0["levels"]["L3"]["lambada_accuracy_min"].as_f64().unwrap();
    let mut rows = Vec::new();
    let mut brief = String::from("## BLiMP and LAMBADA (evaluation-only)\n\n| model | exposure bytes | BLiMP pair accuracy | BLiMP paradigm mean | LAMBADA accuracy | LAMBADA target bits |\n|---|---|---|---|---|---|\n");
    for (name, lm) in &models {
        let b = blimp(lm.as_ref(), &pairs);
        let l = lambada(lm.as_ref(), &items);
        println!("{name}: BLiMP {:.3} (paradigm mean {:.3}), LAMBADA {:.4} ({:.2} bits/word)", b.accuracy, b.paradigm_mean, l.accuracy, l.mean_target_bits);
        brief += &format!("| {name} | {} | {:.3} | {:.3} | {:.4} | {:.2} |\n", exposure.len(), b.accuracy, b.paradigm_mean, l.accuracy, l.mean_target_bits);
        let fields: serde_json::Map<String, serde_json::Value> = b.per_field.iter().map(|(k, (c, n))| (k.clone(), serde_json::json!(*c as f64 / *n as f64))).collect();
        rows.push(serde_json::json!({
            "model": name, "exposure_bytes": exposure.len(),
            "blimp": {"pairs": b.pairs, "correct": b.correct, "ties": b.ties, "accuracy": b.accuracy, "paradigm_mean": b.paradigm_mean, "per_field_accuracy": fields, "meets_l2": b.paradigm_mean >= blimp_l2, "meets_l3": b.paradigm_mean >= blimp_l3},
            "lambada": {"items": l.items, "correct": l.correct, "accuracy": l.accuracy, "mean_target_bits": l.mean_target_bits, "meets_l3": l.accuracy >= lambada_l3},
        }));
    }
    brief += "\\nChance for BLiMP is 0.5. LAMBADA is scored as exact greedy byte decoding of the final word (detokenized, lower-cased text). L2/L3 flags use `model/specs/levels.json`; the BPB and rating parts of those levels are not measured here.\n".replace("\\n", "\n").as_str();
    print!("{brief}");
    let summary = serde_json::json!({
        "report": "benchmarks",
        "manifest_sha256": corpus::sha256_hex(manifest_text.as_bytes()),
        "levels_sha256": Levels::spec_sha256(),
        "exposure_sha256": corpus::sha256_hex(exposure),
        "blimp_sha256": blimp_sha, "lambada_sha256": lambada_sha,
        "lambada_skipped": skipped,
        "rows": rows,
        "scope_note": "Count-based baselines only, one split; evaluation sets were never in exposure (see the contamination check). No ZeroLM model is claimed to reach any level by this report."
    });
    let dir = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/benchmarks".into()));
    std::fs::create_dir_all(&dir).expect("out dir");
    std::fs::write(dir.join("brief.md"), &brief).expect("write");
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).expect("write");
}
