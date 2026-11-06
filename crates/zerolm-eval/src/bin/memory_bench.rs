use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zerolm_eval::memory::{curves_csv, run, brief_md, summary_md, Report, Spec};

const USAGE: &str = "usage: zerolm-memory [--pilot] [--out DIR]\n  main run (default): main seeds, fails unless every check and frozen gate passes\n  --pilot: pilot seeds only, never gates (used to set the gates)";

/// Memory gate: persistent weight-resident memory across a restart.
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
    let out = out.unwrap_or_else(|| PathBuf::from(if pilot { "reports/memory-pilot" } else { "reports/memory" }));
    let spec = match Spec::load(root) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let seeds = if pilot { spec.pilot_seeds.clone() } else { spec.main_seeds.clone() };
    let report = match run(root, &spec, &seeds, !pilot, &out.join("work")) {
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
    println!("{}", summary_md(&report));
    if report.passed || pilot {
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
    std::fs::create_dir_all(out)?;
    let _ = std::fs::remove_dir_all(out.join("work"));
    std::fs::write(out.join("summary.json"), json(r))?;
    std::fs::write(out.join("summary.md"), summary_md(r))?;
    std::fs::write(out.join("brief.md"), brief_md(r))?;
    std::fs::write(out.join("environment.json"), json(&zerolm_policy::baseline::environment(root)))?;
    let (inter, sweep, fast) = curves_csv(r);
    std::fs::write(out.join("interference-capacity.csv"), inter)?;
    std::fs::write(out.join("precision-recall.csv"), sweep)?;
    std::fs::write(out.join("fast-weights.csv"), fast)?;
    let mut hashes = String::new();
    for x in &r.seeds {
        hashes += &format!("{}  seed={} checkpoint\n{}  seed={} core\n", x.checkpoint_sha256, x.seed, x.final_core_sha256, x.seed);
    }
    std::fs::write(out.join("checkpoint.sha256"), hashes)
}

fn json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string_pretty(v).expect("serialize") + "\n"
}
