//! `zerolm reason`: statements of the bounded language are stored in the
//! relational tensor memory; questions are answered by the Phase 6 operators,
//! with the operators and premise count shown for derived answers. With
//! `--dir`, memory is committed atomically after each stored statement.

use serde::Serialize;
use crate::verbose::{short, View};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use zerolm_engine::semantic_state::parse;
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

pub fn run(dir: Option<&Path>, trace: Option<String>, stream: &str, view: &View) -> ExitCode {
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
    view.note("memory", format!("{} statements, {} symbols; capacity {} slots, {} symbols, {}-dim phasors", r.n_slots(), r.n_symbols(), r.cfg.slots, r.cfg.symbols, r.cfg.dim));
    // Display only: the model stores vectors, not sentences, so this map lets
    // verbose output show which input line wrote each slot in this session.
    let mut written_by: HashMap<usize, String> = HashMap::new();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let mut failed = false;
    for (seq, line) in stdin.lock().split(b'\n').enumerate() {
        let Ok(raw) = line else { break };
        let text = String::from_utf8_lossy(&raw).into_owned();
        let text = text.strip_suffix('\r').unwrap_or(&text);
        view.input(seq, text);
        if view.on() {
            match parse(text) {
                Some(u) => view.note("parsed", serde_json::to_string(&u).expect("serialize")),
                None => view.note("parsed", "no match in the bounded grammar"),
            }
        }
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
        if view.on() {
            explain(view, &r, &observed, &written_by);
        }
        if let Observed::Stored { slot } = observed {
            written_by.insert(slot, text.to_string());
        }
        let mut checkpoint_sha256 = None;
        if let (Some(d), Observed::Stored { .. }) = (dir, &observed) {
            match r.save(d) {
                Ok(sha) => {
                    view.note("checkpoint", format!("{} saved atomically to {}", short(&sha), d.display()));
                    checkpoint_sha256 = Some(sha)
                }
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

/// Verbose detail for one observation: the write it caused or the derivation behind an answer.
fn explain(view: &View, r: &Reasoner, observed: &Observed, written_by: &HashMap<usize, String>) {
    match observed {
        Observed::Stored { slot } => {
            view.note("stored", format!("slot {slot} (bound role-filler vector)"));
            if let Some(ev) = r.events().last() {
                view.note("update", format!("rule={} signal={} stream={}", ev.rule, ev.signal, ev.input_stream));
                view.note("changed", format!("{} (rows sha256 {})", ev.changed_tensors.join(", "), short(&ev.rows_sha256)));
            }
        }
        Observed::Duplicate { slot } => view.note("duplicate", format!("matches slot {slot}; nothing written")),
        Observed::Unresolved => view.note("pronoun", "'it' matched no recent subject; nothing written"),
        Observed::Unparsed => view.note("result", "nothing stored, nothing changed"),
        Observed::Answer(a) => {
            view.note("verdict", a.verdict.text());
            let how = if a.derived { "derived by operators" } else { "direct retrieval" };
            view.note("how", format!("{how}: {}", if a.operators.is_empty() { "none".to_string() } else { a.operators.join(" -> ") }));
            for p in &a.premises {
                let src = written_by.get(p).map(|s| format!("\"{s}\"")).unwrap_or_else(|| "(stored before this session)".to_string());
                view.note("premise", format!("slot {p} <- {src}"));
            }
            view.note("confidence", format!("{:.2}", a.confidence));
            view.note("note", "answers are never written back to memory");
        }
    }
}
