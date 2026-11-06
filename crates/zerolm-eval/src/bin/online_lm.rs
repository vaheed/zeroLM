use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zerolm_eval::online::{learning_curve_csv, run, brief_md, summary_md, Protocol, Report};

const USAGE: &str = "usage: zerolm-online-lm [--pilot] [--out DIR]\n  main run (default): main seeds, fails unless the frozen gate passes\n  --pilot: pilot seeds only, never gates (used to set the floor)";

/// Online-learning gate: chronological online exposure and held-out evaluation.
fn main() -> ExitCode {
    let mut pilot = false;
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pilot" => pilot = true,
            "--out" => match args.next() {
                Some(d) => out = Some(PathBuf::from(d)),
                None => return usage(),
            },
            _ => return usage(),
        }
    }
    let root = Path::new(".");
    let out = out.unwrap_or_else(|| PathBuf::from(if pilot { "reports/online-learning-pilot" } else { "reports/online-learning" }));
    let protocol = match Protocol::load(root) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let seeds = if pilot { protocol.pilot_seeds.clone() } else { protocol.main_seeds.clone() };
    let report = match run(root, &protocol, &seeds, !pilot) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = write(root, &out, &report) {
        eprintln!("error writing reports: {e}");
        return ExitCode::from(2);
    }
    let md = summary_md(&report);
    println!("{md}");
    if report.passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn write(root: &Path, out: &Path, r: &Report) -> std::io::Result<()> {
    std::fs::create_dir_all(out.join("trace-samples"))?;
    std::fs::write(out.join("summary.json"), json(r))?;
    std::fs::write(out.join("summary.md"), summary_md(r))?;
    std::fs::write(out.join("brief.md"), brief_md(r))?;
    std::fs::write(out.join("environment.json"), json(&zerolm_policy::baseline::environment(root)))?;
    let metrics = serde_json::json!({
        "verdicts": r.verdicts,
        "gates": r.gates,
        "baselines": r.baselines,
        "recurrence_ablation_delta_bpb": r.recurrence_ablation_delta_bpb,
    });
    std::fs::write(out.join("metrics.json"), json(&metrics))?;
    std::fs::write(out.join("learning-curve.csv"), learning_curve_csv(r))?;
    let hashes = |f: &dyn Fn(&zerolm_eval::online::RunResult) -> &str| {
        let mut s = String::new();
        for x in &r.runs {
            s += &format!("{}  seed={} arm={}\n", f(x), x.seed, x.arm);
        }
        s
    };
    std::fs::write(out.join("initial-weight.sha256"), hashes(&|x| &x.initial_sha256))?;
    std::fs::write(out.join("final-weight.sha256"), hashes(&|x| &x.final_sha256))?;
    // Trace samples: the first 200 update events and every checkpoint record
    // of each learning run on the first seed.
    if let Some(seed) = r.seeds.first().map(|s| s.seed) {
        for run in r.runs.iter().filter(|x| x.seed == seed && !x.events.is_empty()) {
            let ev: String = run.events.iter().take(200).map(|e| serde_json::to_string(e).unwrap() + "\n").collect();
            std::fs::write(out.join(format!("trace-samples/events-seed{seed}-{}.jsonl", run.arm)), ev)?;
            let cp: String = run.checkpoint_log.iter().map(|c| serde_json::to_string(c).unwrap() + "\n").collect();
            std::fs::write(out.join(format!("trace-samples/checkpoints-seed{seed}-{}.jsonl", run.arm)), cp)?;
        }
    }
    Ok(())
}

fn json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string_pretty(v).expect("serialize") + "\n"
}
