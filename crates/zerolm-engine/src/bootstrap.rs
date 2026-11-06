//! Programmed bootstrap: the explicit teaching/control protocol.
//!
//! The model is *read-only* here (`&Model`), so a session cannot change
//! weights; this is enforced by the type system and checked by hash in tests.
//! The protocol keeps a small session-local **ledger** so teach/ask/correct/
//! confirm/forget are observable. The ledger is bootstrap scaffolding, not
//! model memory and not learning. Weight-resident memory is `zerolm-memory`
//! (CLI: `zerolm memory --dir`), which persists facts as checkpoint tensors.
//! The model state still receives every input byte, but with no learning rule
//! it carries no understanding of English.

use crate::Model;
use serde::Serialize;
use std::collections::BTreeMap;
use zerolm_nn::Kind;

pub const COMMANDS: [&str; 6] = ["teach", "ask", "correct", "confirm", "forget", "repeat"];
pub const DELIMITER: &str = " => ";
pub const DEFAULT_SEED: u64 = 1;
pub const EMBED_DIM: usize = 16;
pub const CELL_BUDGET: usize = 20_000;
pub const DEFAULT_KIND: Kind = Kind::Gru;

/// The deterministic clean model every session starts from.
pub fn default_model(seed: u64) -> Model {
    Model::new(DEFAULT_KIND, seed, EMBED_DIM, CELL_BUDGET)
}

/// Half-open byte range in the full input stream (not just the line).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Teach { key: Span, value: Span },
    Ask { key: Span },
    Correct { key: Span, value: Span },
    Confirm { key: Span },
    Forget { key: Span },
    Repeat,
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Command::Teach { .. } => "teach",
            Command::Ask { .. } => "ask",
            Command::Correct { .. } => "correct",
            Command::Confirm { .. } => "confirm",
            Command::Forget { .. } => "forget",
            Command::Repeat => "repeat",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Not a protocol line; treated as plain observed text.
    NotACommand,
    /// Command word recognised but the arguments are malformed.
    Malformed(&'static str),
}

/// Parse one line. `base` is the byte offset of the line start in the stream.
pub fn parse_line(line: &str, base: usize) -> Result<Command, ParseError> {
    let (word, rest) = match line.split_once(' ') {
        Some((w, r)) => (w, Some(r)),
        None => (line, None),
    };
    if !COMMANDS.contains(&word) {
        return Err(ParseError::NotACommand);
    }
    let arg_start = base + word.len() + 1;
    let span = |s: usize, e: usize| Span { start: s, end: e };
    let key_only = |rest: Option<&str>| match rest {
        Some(r) if !r.trim().is_empty() && !r.contains(DELIMITER) => Ok(span(arg_start, arg_start + r.len())),
        _ => Err(ParseError::Malformed("expected: <command> <key>")),
    };
    let pair = |rest: Option<&str>| match rest.and_then(|r| r.split_once(DELIMITER)) {
        Some((k, v)) if !k.trim().is_empty() && !v.trim().is_empty() && !v.contains(DELIMITER) => {
            let vs = arg_start + k.len() + DELIMITER.len();
            Ok((span(arg_start, arg_start + k.len()), span(vs, vs + v.len())))
        }
        _ => Err(ParseError::Malformed("expected: <command> <key> => <value>")),
    };
    match word {
        "teach" => pair(rest).map(|(key, value)| Command::Teach { key, value }),
        "correct" => pair(rest).map(|(key, value)| Command::Correct { key, value }),
        "ask" => key_only(rest).map(|key| Command::Ask { key }),
        "confirm" => key_only(rest).map(|key| Command::Confirm { key }),
        "forget" => key_only(rest).map(|key| Command::Forget { key }),
        _ if rest.is_none() => Ok(Command::Repeat),
        _ => Err(ParseError::Malformed("repeat takes no arguments")),
    }
}

#[derive(Debug, Clone)]
struct Entry {
    value: String,
    confirmed: bool,
}

/// One inspectable trace record per input line.
#[derive(Debug, Clone, Serialize)]
pub struct Event {
    pub seq: usize,
    pub line_span: Span,
    pub command: &'static str,
    pub key_span: Option<Span>,
    pub value_span: Option<Span>,
    pub outcome: String,
    pub response: String,
    /// Where the answer came from: always the session ledger or "none".
    pub source: &'static str,
    /// Weights are never modified in a session; the hash proves it.
    pub weights_sha256: String,
}

pub struct Session<'m> {
    model: &'m Model,
    state: Vec<f32>,
    ledger: BTreeMap<String, Entry>,
    events: Vec<Event>,
    offset: usize,
    last_response: String,
    weights_sha256: String,
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model) -> Self {
        Session {
            state: model.init_state(),
            ledger: BTreeMap::new(),
            events: Vec::new(),
            offset: 0,
            last_response: String::new(),
            weights_sha256: model.weights_sha256(),
            model,
        }
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Reset the session-local state and ledger ("forget everything").
    pub fn reset(&mut self) {
        *self = Session::new(self.model);
    }

    /// Process one line (without its trailing newline) and return the response.
    pub fn handle_line(&mut self, line: &str) -> String {
        let base = self.offset;
        let text = |s: Span| &line[s.start - base..s.end - base];
        let parsed = parse_line(line, base);
        let (command, key_span, value_span, outcome, response, source) = match &parsed {
            Err(ParseError::NotACommand) => (
                "observe",
                None,
                None,
                "observed".to_string(),
                format!("observed {} bytes", line.len()),
                "none",
            ),
            Err(ParseError::Malformed(why)) => ("error", None, None, "malformed".to_string(), format!("error: {why}"), "none"),
            Ok(cmd) => {
                let name = cmd.name();
                match cmd {
                    Command::Teach { key, value } => {
                        let (k, v) = (text(*key).to_string(), text(*value).to_string());
                        match self.ledger.get(&k) {
                            Some(e) if e.value != v => {
                                (name, Some(*key), Some(*value), "conflict".into(), format!("conflict: {k} already has a value; use correct"), "ledger")
                            }
                            Some(_) => (name, Some(*key), Some(*value), "unchanged".into(), format!("already noted {k}"), "ledger"),
                            None => {
                                self.ledger.insert(k.clone(), Entry { value: v, confirmed: false });
                                (name, Some(*key), Some(*value), "stored".into(), format!("noted {k} (unconfirmed)"), "ledger")
                            }
                        }
                    }
                    Command::Correct { key, value } => {
                        let (k, v) = (text(*key).to_string(), text(*value).to_string());
                        let existed = self.ledger.insert(k.clone(), Entry { value: v, confirmed: false }).is_some();
                        let out = if existed { "replaced" } else { "stored" };
                        (name, Some(*key), Some(*value), out.into(), format!("corrected {k} (unconfirmed)"), "ledger")
                    }
                    Command::Confirm { key } => {
                        let k = text(*key);
                        match self.ledger.get_mut(k) {
                            Some(e) => {
                                e.confirmed = true;
                                (name, Some(*key), None, "confirmed".into(), format!("confirmed {k}"), "ledger")
                            }
                            None => (name, Some(*key), None, "unknown".into(), format!("unknown {k}"), "none"),
                        }
                    }
                    Command::Ask { key } => {
                        let k = text(*key);
                        match self.ledger.get(k) {
                            Some(e) => {
                                let tag = if e.confirmed { "confirmed" } else { "unconfirmed" };
                                (name, Some(*key), None, tag.into(), format!("{k} => {} ({tag})", e.value), "ledger")
                            }
                            None => (name, Some(*key), None, "unknown".into(), format!("unknown {k}"), "none"),
                        }
                    }
                    Command::Forget { key } => {
                        let k = text(*key);
                        match self.ledger.remove(k) {
                            Some(_) => (name, Some(*key), None, "forgotten".into(), format!("forgot {k}"), "ledger"),
                            None => (name, Some(*key), None, "unknown".into(), format!("unknown {k}"), "none"),
                        }
                    }
                    Command::Repeat => (name, None, None, "repeated".into(), self.last_response.clone(), "none"),
                }
            }
        };
        // The model sees every byte (plus the newline); weights stay untouched.
        let _ = self.model.feed(&mut self.state, line.as_bytes());
        let _ = self.model.feed(&mut self.state, b"\n");
        self.offset += line.len() + 1;
        self.events.push(Event {
            seq: self.events.len(),
            line_span: Span { start: base, end: base + line.len() },
            command,
            key_span,
            value_span,
            outcome,
            response: response.clone(),
            source,
            weights_sha256: self.weights_sha256.clone(),
        });
        if command != "repeat" {
            self.last_response = response.clone();
        }
        response
    }
}
