//! `zerolm-data check [--manifest PATH]` validates the manifest;
//! `zerolm-data verify --root DIR [--manifest PATH]` also checks every shard
//! and evaluation file against its SHA-256; `zerolm-data contamination --root
//! DIR [--n 13]` fails if any evaluation set shares an n-word window with
//! exposure text.

use std::path::PathBuf;
use zerolm_data::{contamination, corpus, manifest::Manifest};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("check");
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let path = flag("--manifest").unwrap_or_else(|| "research/corpus/manifest.json".into());
    let m = match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| Manifest::parse(&t)) {
        Ok(m) => m,
        Err(e) => fail(&format!("{path}: {e}")),
    };
    match cmd {
        "check" => println!("manifest ok: {} sources, {} evaluation sets", m.sources.len(), m.evaluation_sets.len()),
        "verify" | "contamination" => {
            let root = PathBuf::from(flag("--root").unwrap_or_else(|| fail("--root DIR is required")));
            let problems = corpus::verify(&m, &root);
            if !problems.is_empty() {
                fail(&problems.join("\n"));
            }
            if cmd == "verify" {
                println!("all shards and evaluation files match the manifest");
                return;
            }
            let n: usize = flag("--n").map(|v| v.parse().unwrap_or_else(|_| fail("--n must be a number"))).unwrap_or(13);
            let exposure = corpus::role_stream(&m, &root, corpus::Role::Exposure).unwrap_or_else(|e| fail(&e));
            let index = contamination::ngram_hashes(&exposure, n);
            let mut bad = false;
            let mut lower_index = None;
            for e in &m.evaluation_sets {
                let data = std::fs::read(root.join(&e.path)).unwrap_or_else(|e| fail(&e.to_string()));
                use zerolm_data::{evalsets, manifest::EvalKind};
                let (what, shared, total) = match e.kind {
                    EvalKind::Blimp => {
                        let pairs = evalsets::parse_blimp(&data).unwrap_or_else(|e| fail(&e));
                        let all: Vec<&[u8]> = pairs.iter().flat_map(|p| [p.good.as_bytes(), p.bad.as_bytes()]).collect();
                        let checked = all.iter().filter(|s| s.split(|b| b.is_ascii_whitespace()).filter(|w| !w.is_empty()).count() >= evalsets::MIN_VERBATIM_WORDS).count();
                        (format!("sentences of at least {} words found verbatim in exposure ({} shorter sentences not checked)", evalsets::MIN_VERBATIM_WORDS, all.len() - checked), evalsets::verbatim_overlap(&exposure, &all), checked)
                    }
                    EvalKind::Lambada => {
                        // LAMBADA is lower-cased and word-tokenized: compare detokenized, lower-cased text.
                        let text = String::from_utf8_lossy(&data).lines().map(evalsets::detokenize).collect::<Vec<_>>().join("\n");
                        let lower = lower_index.get_or_insert_with(|| contamination::ngram_hashes(&exposure.to_ascii_lowercase(), n));
                        let o = contamination::overlap(lower, text.as_bytes(), n);
                        (format!("{n}-word windows shared with exposure"), o.shared, o.windows)
                    }
                    EvalKind::HeldoutBpb => {
                        let o = contamination::overlap(&index, &data, n);
                        (format!("{n}-word windows shared with exposure"), o.shared, o.windows)
                    }
                };
                println!("{}: {shared}/{total} {what}", e.id);
                bad |= shared > 0;
            }
            if bad {
                fail("contamination: an evaluation set overlaps exposure");
            }
        }
        other => fail(&format!("unknown command '{other}'")),
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("zerolm-data: {msg}");
    std::process::exit(1);
}
