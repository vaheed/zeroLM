//! `zerolm reason`: statements of the bounded language are stored in the
//! relational tensor memory; questions are answered by the Phase 6 operators,
//! with the operators and premise count shown for derived answers. With
//! `--dir`, memory is committed atomically after each stored statement.

use serde::Serialize;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use zerolm_memory::checkpoint::CheckpointError;
use zerolm_reasoner::{Answer, Observed, Reasoner, ReasonerConfig, ReasonerError};

#[derive(Serialize)]
struct Event<'a> {
    seq: usize,
    line: &'a str,
    observed: &'a Observed,
    response: &'a str,
    checkpoint_sha256: Option<String>,
}

fn describe(a: &Answer) -> String {
    let v = a.verdict.text();
    if a.derived {
        format!("{v} (derived: {}; {} premises; confidence {:.2})", a.operators.join(", "), a.premises.len(), a.confidence)
    } else {
        v
    }
}

pub fn run(dir: Option<&Path>, trace: Option<String>, stream: &str) -> ExitCode {
    let mut r = match dir.map(Reasoner::load) {
        Some(Ok(r)) => {
            eprintln!("reason: loaded {} statements from {}", r.n_slots(), dir.expect("dir").display());
            r
        }
        Some(Err(ReasonerError::Checkpoint(CheckpointError::NotFound))) | None => Reasoner::new(ReasonerConfig::default()).expect("default config"),
        Some(Err(e)) => {
            eprintln!("error: cannot load reasoning memory: {e}");
            return ExitCode::from(2);
        }
    };
    let mut sink = match trace.map(std::fs::File::create).transpose() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot create trace file: {e}");
            return ExitCode::from(2);
        }
    };
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let mut failed = false;
    for (seq, line) in stdin.lock().split(b'\n').enumerate() {
        let Ok(raw) = line else { break };
        let text = String::from_utf8_lossy(&raw).into_owned();
        let text = text.strip_suffix('\r').unwrap_or(&text);
        let observed = match r.observe(text, stream) {
            Ok(o) => o,
            Err(e) => {
                let _ = writeln!(out, "error: {e}");
                failed = true;
                continue;
            }
        };
        let response = match &observed {
            Observed::Stored { .. } => "noted".to_string(),
            Observed::Duplicate { .. } => "already noted".to_string(),
            Observed::Unresolved => "cannot tell what 'it' refers to; not stored".to_string(),
            Observed::Unparsed => "not understood (outside the bounded language)".to_string(),
            Observed::Answer(a) => describe(a),
        };
        let mut checkpoint_sha256 = None;
        if let (Some(d), Observed::Stored { .. }) = (dir, &observed) {
            match r.save(d) {
                Ok(sha) => checkpoint_sha256 = Some(sha),
                Err(e) => {
                    let _ = writeln!(out, "error: not saved ({e})");
                    failed = true;
                    continue;
                }
            }
        }
        let _ = writeln!(out, "{response}");
        if let Some(f) = sink.as_mut() {
            let ev = Event { seq, line: text, observed: &observed, response: &response, checkpoint_sha256 };
            let _ = writeln!(f, "{}", serde_json::to_string(&ev).expect("serialize"));
        }
    }
    eprintln!("reason: {} statements, {} symbols in memory", r.n_slots(), r.n_symbols());
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
