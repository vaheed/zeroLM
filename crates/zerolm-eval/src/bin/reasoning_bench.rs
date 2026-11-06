use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zerolm_eval::reasoning::{run, summary_md, Report, Spec};

const USAGE: &str = "usage: zerolm-reasoning [--pilot] [--out DIR]\n  main run (default): main seeds plus one hidden seed; fails unless every frozen gate passes\n  --pilot: pilot seeds only, never gates (used to set the gates)";

/// Reasoning gate: compositional answers from stored primitives, with ablations.
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
    let out = out.unwrap_or_else(|| PathBuf::from(if pilot { "reports/reasoning-pilot" } else { "reports/reasoning" }));
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
    let _ = std::fs::remove_dir_all(out.join("work"));
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
    std::fs::write(out.join("summary.json"), json(r))?;
    std::fs::write(out.join("summary.md"), summary_md(r))?;
    std::fs::write(out.join("environment.json"), json(&zerolm_policy::baseline::environment(root)))?;
    let mut csv = String::from("seed,ablation,kind,category,agreement\n");
    for x in r.seeds.iter().filter(|x| !x.hidden) {
        for a in &x.ablations {
            for (c, v) in &a.per_category {
                csv += &format!("{},{},{},{c},{v:.5}\n", x.seed, a.name, a.kind);
            }
        }
    }
    std::fs::write(out.join("ablations.csv"), csv)
}

fn json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string_pretty(v).expect("serialize") + "\n"
}
