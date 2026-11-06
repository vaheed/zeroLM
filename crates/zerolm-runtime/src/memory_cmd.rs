//! `zerolm memory`: the teaching protocol served from weight-resident memory.
//!
//! Unlike `zerolm run` (session ledger, nothing persists), every answer here
//! comes from the checkpoint tensors in `--dir` and the retrieval computation;
//! each line that changes memory is committed atomically before the reply,
//! so a later process answers from the same facts.

use crate::verbose::{short, View};
use serde::Serialize;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use zerolm_engine::bootstrap::{parse_line, Command, ParseError, Span};
use zerolm_memory::{MemoryConfig, MemoryError, MemoryEvent, MemoryStore, Outcome};

#[derive(Serialize)]
struct Event<'a> {
    seq: usize,
    command: &'a str,
    key_span: Option<Span>,
    value_span: Option<Span>,
    outcome: String,
    response: String,
    /// "memory" when the reply was computed from memory tensors, else "none".
    source: &'static str,
    score: Option<f32>,
    coverage: Option<f32>,
    memory_events: &'a [MemoryEvent],
    checkpoint_sha256: Option<String>,
}

pub fn run(dir: &Path, seed: Option<u64>, trace: Option<String>, stream: &str, view: &View) -> ExitCode {
    let mut cfg = MemoryConfig::default();
    if let Some(s) = seed {
        cfg.seed = s;
    }
    let (mut store, info) = match MemoryStore::open(dir, cfg) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: cannot open memory in {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    match &info {
        Some(i) => eprintln!("memory: loaded {:?} checkpoint {} ({} active facts){}", i.source, i.checkpoint_sha256, store.stats().active, i.current_error.as_ref().map(|e| format!("; current file rejected: {e}")).unwrap_or_default()),
        None => eprintln!("memory: no checkpoint in {}; starting from the clean model", dir.display()),
    }
    let core_before = store.digest("core.");
    view.note("core", format!("frozen core tensors digest {}", short(&core_before)));
    let mut sink = match trace.map(std::fs::File::create).transpose() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot create trace file: {e}");
            return ExitCode::from(2);
        }
    };
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let (mut offset, mut last, mut seq) = (0usize, String::new(), 0usize);
    let mut failed = false;
    for line in stdin.lock().split(b'\n') {
        let Ok(raw) = line else { break };
        let text = String::from_utf8_lossy(&raw).into_owned();
        let text = text.strip_suffix('\r').unwrap_or(&text);
        let n_events = store.events().len();
        let (command, key_span, value_span, result) = handle(&mut store, text, offset, &last, stream);
        let (outcome, response, source, score, coverage, changed) = match result {
            Ok(r) => r,
            Err(e) => ("error".to_string(), format!("error: {e}"), "none", None, None, false),
        };
        if view.on() {
            view.input(seq, text);
            view.note("command", command);
            view.note("outcome", format!("{outcome} (answer source: {source})"));
            if let (Some(s), Some(c)) = (score, coverage) {
                view.note("retrieval", format!("attention score {s:.3}, word coverage {c:.3}"));
            }
            for ev in &store.events()[n_events..] {
                view.note("update", format!("{} rule={} signal={} slot={}", ev.op, ev.rule, ev.signal, ev.slot.map_or("-".to_string(), |s| s.to_string())));
                view.note("changed", format!("{} (rows sha256 {})", ev.changed_tensors.join(", "), short(&ev.rows_sha256)));
            }
        }
        let mut checkpoint_sha256 = None;
        let mut response = response;
        if changed {
            match store.save(dir) {
                Ok(sha) => {
                    view.note("checkpoint", format!("{} saved atomically to {}", short(&sha), dir.display()));
                    checkpoint_sha256 = Some(sha)
                }
                Err(e) => {
                    response = format!("error: not saved ({e}); the last valid checkpoint is unchanged");
                    failed = true;
                }
            }
        }
        let _ = writeln!(out, "{response}");
        if let Some(f) = sink.as_mut() {
            let ev = Event { seq, command, key_span, value_span, outcome, response: response.clone(), source, score, coverage, memory_events: &store.events()[n_events..], checkpoint_sha256 };
            let _ = writeln!(f, "{}", serde_json::to_string(&ev).expect("serialize"));
        }
        if command != "repeat" {
            last = response;
        }
        offset += raw.len() + 1;
        seq += 1;
    }
    let core_after = store.digest("core.");
    let st = store.stats();
    eprintln!(
        "memory: {} active, {} superseded, {} deleted, {} free slots; core unchanged={}",
        st.active,
        st.superseded,
        st.deleted,
        st.free,
        core_before == core_after
    );
    if core_before == core_after && !failed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

type Reply = (String, String, &'static str, Option<f32>, Option<f32>, bool);

fn handle(store: &mut MemoryStore, line: &str, base: usize, last: &str, stream: &str) -> (&'static str, Option<Span>, Option<Span>, Result<Reply, MemoryError>) {
    let text = |s: Span| &line[s.start - base..s.end - base];
    let cmd = match parse_line(line, base) {
        Err(ParseError::NotACommand) => {
            return ("observe", None, None, Ok(("observed".into(), format!("observed {} bytes (not stored)", line.len()), "none", None, None, false)));
        }
        Err(ParseError::Malformed(why)) => return ("error", None, None, Ok(("malformed".into(), format!("error: {why}"), "none", None, None, false))),
        Ok(c) => c,
    };
    let name = cmd.name();
    let simple = |o: &str, r: String, changed: bool| Ok((o.to_string(), r, "memory", None, None, changed));
    match cmd {
        Command::Teach { key, value } => {
            let (k, v) = (text(key).trim(), text(value).trim());
            let r = store.teach(k, v, stream).and_then(|o| match o {
                Outcome::Stored { .. } => simple("stored", format!("noted {k} (unconfirmed)"), true),
                Outcome::Unchanged { .. } => simple("unchanged", format!("already noted {k}"), true),
                Outcome::Conflict { .. } => simple("conflict", format!("conflict: {k} already has a different value; use correct"), true),
                other => unreachable!("teach returned {other:?}"),
            });
            (name, Some(key), Some(value), r)
        }
        Command::Correct { key, value } => {
            let (k, v) = (text(key).trim(), text(value).trim());
            let r = store.correct(k, v, stream).and_then(|o| match o {
                Outcome::Superseded { .. } => simple("replaced", format!("corrected {k} (unconfirmed)"), true),
                Outcome::Stored { .. } => simple("stored", format!("corrected {k} (unconfirmed)"), true),
                Outcome::Unchanged { .. } => simple("unchanged", format!("already noted {k}"), true),
                other => unreachable!("correct returned {other:?}"),
            });
            (name, Some(key), Some(value), r)
        }
        Command::Confirm { key } => {
            let k = text(key).trim();
            let r = store.confirm(k, stream).and_then(|o| match o {
                Outcome::Confirmed { .. } => simple("confirmed", format!("confirmed {k}"), true),
                _ => simple("unknown", format!("unknown {k}"), false),
            });
            (name, Some(key), None, r)
        }
        Command::Forget { key } => {
            let k = text(key).trim();
            let r = store.forget(k, stream).and_then(|o| match o {
                Outcome::Forgotten { .. } => simple("forgotten", format!("forgot {k}"), true),
                _ => simple("unknown", format!("unknown {k}"), false),
            });
            (name, Some(key), None, r)
        }
        Command::Ask { key } => {
            let k = text(key).trim();
            let r = store.query(k).map(|a| match &a.value {
                Some(v) => {
                    let tag = if a.confirmed() { "confirmed" } else { "unconfirmed" };
                    (tag.to_string(), format!("{k} => {v} ({tag})"), "memory", Some(a.score), Some(a.coverage), false)
                }
                None => ("unknown".to_string(), format!("unknown {k}"), "memory", Some(a.score), Some(a.coverage), false),
            });
            (name, Some(key), None, r)
        }
        Command::Repeat => (name, None, None, Ok(("repeated".into(), last.to_string(), "none", None, None, false))),
    }
}
