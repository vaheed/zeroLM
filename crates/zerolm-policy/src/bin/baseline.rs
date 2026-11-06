use std::path::PathBuf;
use std::process::ExitCode;

/// Usage: zerolm-baseline (generate|check) [--root DIR]
fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_default();
    let mut root = PathBuf::from(".");
    while let Some(a) = args.next() {
        match (a.as_str(), args.next()) {
            ("--root", Some(d)) => root = PathBuf::from(d),
            _ => return usage(),
        }
    }
    let violations = match mode.as_str() {
        "generate" => match zerolm_policy::baseline::generate(&root) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
        "check" => zerolm_policy::baseline::check(&root),
        _ => return usage(),
    };
    println!("baseline {mode}: {}", if violations.is_empty() { "PASS" } else { "FAIL" });
    for v in &violations {
        println!("  VIOLATION [{}] {}", v.check, v.detail);
    }
    if violations.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

fn usage() -> ExitCode {
    eprintln!("usage: zerolm-baseline (generate|check) [--root DIR]");
    ExitCode::from(2)
}
