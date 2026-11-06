use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zerolm_eval::uniform::{brief_md, run};

/// Uniform-engine gate: reproducible circuits, patch round trips, reference
/// replay of every circuit run, and the host audit.
fn main() -> ExitCode {
    let mut out = PathBuf::from("reports/uniform-engine");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match (a.as_str(), args.next()) {
            ("--out", Some(d)) => out = PathBuf::from(d),
            _ => {
                eprintln!("usage: zerolm-uniform [--out DIR]");
                return ExitCode::from(2);
            }
        }
    }
    let report = match run(Path::new(".")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let brief = brief_md(&report);
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(&out)?;
        std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&report).expect("serialize") + "\n")?;
        std::fs::write(out.join("summary.md"), &brief)?;
        std::fs::write(out.join("brief.md"), &brief)
    };
    if let Err(e) = write() {
        eprintln!("error writing reports: {e}");
        return ExitCode::from(2);
    }
    println!("{brief}");
    if report.passed { ExitCode::SUCCESS } else { ExitCode::from(1) }
}
