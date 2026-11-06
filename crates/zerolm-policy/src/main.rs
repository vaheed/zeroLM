use std::path::PathBuf;
use std::process::ExitCode;

/// Usage: zerolm-audit [--root DIR] [--json]
fn main() -> ExitCode {
    let mut root = PathBuf::from(".");
    let mut json = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root" => match args.next() {
                Some(d) => root = PathBuf::from(d),
                None => return usage(),
            },
            "--json" => json = true,
            _ => return usage(),
        }
    }
    let report = zerolm_policy::audit(&root);
    if json {
        println!("{}", serde_json::to_string_pretty(&report).expect("serialize"));
    } else {
        println!("phase-0 audit: {}", if report.passed { "PASS" } else { "FAIL" });
        for c in &report.checks_run {
            println!("  ran {c}");
        }
        for v in &report.violations {
            println!("  VIOLATION [{}] {}", v.check, v.detail);
        }
    }
    if report.passed { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

fn usage() -> ExitCode {
    eprintln!("usage: zerolm-audit [--root DIR] [--json]");
    ExitCode::from(2)
}
