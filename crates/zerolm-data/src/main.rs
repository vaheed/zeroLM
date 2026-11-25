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
            for e in &m.evaluation_sets {
                let data = std::fs::read(root.join(&e.path)).unwrap_or_else(|e| fail(&e.to_string()));
                let o = contamination::overlap(&index, &data, n);
                println!("{}: {}/{} {n}-word windows shared with exposure", e.id, o.shared, o.windows);
                bad |= o.shared > 0;
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
