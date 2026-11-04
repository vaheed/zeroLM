//! Live session: the model plus a persistent, decaying memory (the "LLLM" layer).
//! The base `weights.zlm` is never modified; all life lives in the life file.

use crate::memory::Memory;
use crate::{normalize_text, Model, MAX_TEXT};
use std::path::PathBuf;

pub struct Session<'a> {
    model: &'a Model,
    pub memory: Memory,
    path: Option<PathBuf>,
}

fn strip_prefix_ci<'s>(s: &'s str, prefix: &str) -> Option<&'s str> {
    let n = prefix.len();
    if s.len() >= n && s.is_char_boundary(n) && s[..n].eq_ignore_ascii_case(prefix) {
        Some(&s[n..])
    } else {
        None
    }
}

fn norm_string(s: &str) -> String {
    String::from_utf8_lossy(&normalize_text(s.as_bytes(), MAX_TEXT)).into_owned()
}

const QUESTION_OPENERS: [&str; 10] = [
    "what is ", "what s ", "what are ", "who is ", "who s ", "where is ", "when is ",
    "recall ", "tell me about ", "do you remember ",
];

impl<'a> Session<'a> {
    pub fn new(model: &'a Model, memory: Memory, path: Option<PathBuf>) -> Session<'a> {
        Session { model, memory, path }
    }

    /// Handle one user turn and return the reply.
    pub fn turn(&mut self, raw: &str) -> String {
        self.memory.tick();
        let raw = raw.trim();
        let reply = self.dispatch(raw);
        if let Some(p) = &self.path {
            if let Err(e) = self.memory.save(p) {
                eprintln!("warning: cannot save life file {}: {e}", p.display());
            }
        }
        reply
    }

    fn dispatch(&mut self, raw: &str) -> String {
        if let Some(rest) = strip_prefix_ci(raw, "remember ") {
            return self.cmd_remember(rest);
        }
        if let Some(rest) = strip_prefix_ci(raw, "forget ") {
            let rest = strip_prefix_ci(rest, "about ").unwrap_or(rest);
            let key = norm_string(rest);
            return if !key.is_empty() && self.memory.forget(&key) {
                format!("Forgot {key}.")
            } else {
                format!("I do not remember {key}.")
            };
        }
        let norm = normalize_text(raw.as_bytes(), MAX_TEXT);
        let norm_str = String::from_utf8_lossy(&norm);
        if QUESTION_OPENERS.iter().any(|o| norm_str.starts_with(o)) {
            if let Some((key, value)) = self.memory.recall(&norm) {
                return format!("{key} is {value}.");
            }
        }
        let (intent, _) = self.model.decide(raw);
        if self.model.intent_name[intent] == "memory" {
            return format!(
                "I keep up to {} memories, and unused ones fade as we talk. \
Say \"remember X is Y\", ask \"what is X\", or say \"forget X\".",
                self.memory.capacity
            );
        }
        self.model.generate(intent).join(" ")
    }

    fn cmd_remember(&mut self, rest: &str) -> String {
        let usage = "Say it like: remember <thing> is <value>.".to_string();
        let rest = strip_prefix_ci(rest, "that ").unwrap_or(rest);
        let lower = rest.to_ascii_lowercase();
        let split = [" is ", " are "]
            .iter()
            .filter_map(|sep| lower.find(sep).map(|i| (i, sep.len())))
            .min_by_key(|&(i, _)| i);
        let Some((i, len)) = split else { return usage };
        let key = norm_string(&rest[..i]);
        let value = rest[i + len..].trim().trim_end_matches(['.', '!', '?']).trim_end();
        if key.is_empty() || value.is_empty() {
            return usage;
        }
        let evicted = self.memory.remember(&key, value);
        match evicted {
            Some(old) => format!("Okay, I will remember that {key} is {value}. (I let go of {old}.)"),
            None => format!("Okay, I will remember that {key} is {value}."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "ZEROLM 1\nMETA 2 4\nINTENT 0 memory\nINTENT 1 unknown\n\
FEATURE 0 8.0 memory\nSTART 0 0\nSTART 1 2\n\
NODE 0 1 canned\nNODE 1 -1 <eos>\nNODE 2 3 sorry\nNODE 3 -1 <eos>\n";

    #[test]
    fn remember_recall_forget() {
        let m = Model::parse(TINY).unwrap();
        let mut s = Session::new(&m, Memory::new(8), None);
        assert!(s.turn("Remember that my name is Sam.").starts_with("Okay"));
        assert_eq!(s.turn("what is my name?"), "my name is Sam.");
        assert_eq!(s.turn("forget my name"), "Forgot my name.");
        assert_eq!(s.turn("what is my name?"), "sorry"); // falls through to unknown
    }

    #[test]
    fn memory_intent_is_live_not_canned() {
        let m = Model::parse(TINY).unwrap();
        let mut s = Session::new(&m, Memory::new(8), None);
        assert!(s.turn("memory").contains("8 memories"));
    }

    #[test]
    fn life_survives_restart() {
        let m = Model::parse(TINY).unwrap();
        let dir = std::env::temp_dir().join(format!("zerolm-live-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("life.zlm");
        {
            let mut s = Session::new(&m, Memory::load(&p, 8).unwrap(), Some(p.clone()));
            s.turn("remember the code word is plum");
        }
        let mut s = Session::new(&m, Memory::load(&p, 8).unwrap(), Some(p.clone()));
        assert_eq!(s.turn("what is the code word"), "the code word is plum.");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn bad_remember_gives_usage() {
        let m = Model::parse(TINY).unwrap();
        let mut s = Session::new(&m, Memory::new(8), None);
        assert!(s.turn("remember nothing").starts_with("Say it like"));
    }
}
