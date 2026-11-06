use std::path::PathBuf;
use std::process::ExitCode;
use zerolm_eval::*;
use zerolm_nn::Kind;

/// Usage: zerolm-bakeoff [--quick] [--out DIR] [--check ARCH_JSON]
fn main() -> ExitCode {
    let mut cfg = Config::full();
    let mut out = PathBuf::from("reports/bakeoff");
    let mut check: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--quick" => cfg = Config::quick(),
            "--out" => out = PathBuf::from(args.next().unwrap_or_default()),
            "--check" => check = args.next().map(PathBuf::from),
            _ => {
                eprintln!("usage: zerolm-bakeoff [--quick] [--out DIR] [--check model/specs/architecture.json]");
                return ExitCode::from(2);
            }
        }
    }
    let mut results: Vec<CandidateResult> = Kind::ALL.iter().map(|&k| evaluate(k, &cfg)).collect();
    score_all(&mut results);
    let selected = select(&results).filter(|r| within_cpu_budget(r)).map(|r| r.kind.clone());
    let mut failures = Vec::new();
    for r in &results {
        if !r.stable {
            failures.push(format!("{}: unstable", r.kind));
        }
        if !within_cpu_budget(r) {
            failures.push(format!("{}: p95 latency over {P95_LATENCY_BUDGET_US}us budget", r.kind));
        }
        let ok_ablation = r.ablations.iter().all(|a| a.horizon_severed < a.horizon_intact.max(1) || a.horizon_intact == 0);
        if !ok_ablation {
            failures.push(format!("{}: severing the recurrent path did not reduce retention", r.kind));
        }
    }
    if selected.is_none() {
        failures.push("no eligible candidate selected".into());
    }
    if let (Some(path), Some(sel)) = (&check, &selected) {
        match std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) {
            Some(v) if v["selected"] == sel.as_str() => {}
            Some(v) => failures.push(format!("architecture.json selects {} but the bake-off selects {sel}", v["selected"])),
            None => failures.push(format!("cannot read {}", path.display())),
        }
    }
    let report = serde_json::json!({
        "report": "bakeoff",
        "quick": cfg.budget != Config::full().budget,
        "budget": cfg.budget,
        "seeds": cfg.seeds,
        "embed_dim": EMBED_DIM,
        "selected": selected,
        "passed": failures.is_empty(),
        "failures": failures,
        "candidates": results,
        "scope_note": "Metrics are measured at seeded initialization; no learning occurred. They measure architectural capacity, not language ability.",
    });
    std::fs::create_dir_all(&out).expect("create report dir");
    std::fs::write(out.join("benchmark.json"), serde_json::to_string_pretty(&report).unwrap() + "\n").expect("write");
    std::fs::write(out.join("summary.md"), summary(&report)).expect("write");
    println!("{}", summary(&report));
    if report["passed"] == true { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

fn summary(r: &serde_json::Value) -> String {
    let mut s = format!(
        "## Architecture bake-off: {}\n\nSelected: **{}** (cell budget {} params, seeds {})\n\n| candidate | eligible | hidden | cell params | state bytes | horizon | p50 us | p95 us | tok/s | score |\n|---|---|---|---|---|---|---|---|---|---|\n",
        if r["passed"] == true { "PASS" } else { "FAIL" },
        r["selected"].as_str().unwrap_or("none"),
        r["budget"],
        r["seeds"]
    );
    for c in r["candidates"].as_array().unwrap() {
        let cpu = &c["cpu"][0];
        s += &format!(
            "| {} | {} | {} | {} | {} | {} | {:.1} | {:.1} | {:.0} | {:.3} |\n",
            c["kind"].as_str().unwrap(), c["eligible"], c["hidden_dim"], c["cell_params"], c["session_state_bytes"],
            c["median_horizon"], cpu["p50_us"].as_f64().unwrap(), cpu["p95_us"].as_f64().unwrap(),
            cpu["tokens_per_sec"].as_f64().unwrap(), c["score"].as_f64().unwrap()
        );
    }
    for f in r["failures"].as_array().unwrap() {
        s += &format!("\n- FAILURE: {}", f.as_str().unwrap());
    }
    s + "\n\nNo learning occurred: these are initialization-time capacity measurements.\n"
}
