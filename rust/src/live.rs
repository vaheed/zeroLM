//! Live session: the model plus a persistent, decaying memory and a rolling
//! transcript (the "LLLM" layer). The base `weights.zlm` is never modified;
//! all life lives in the life file.

use crate::context::{Transcript, Turn, DEFAULT_CONTEXT_BUDGET};
use crate::memory::Memory;
use crate::{normalize_text, Feature, Model, ABSTAIN_THRESHOLD, MAX_TEXT};
use std::path::PathBuf;

/// Weight given to a taught phrase (same scale as the base model's +8).
const TEACH_WEIGHT: f64 = 8.0;
const GOOD_DELTA: f64 = 1.0;
const BAD_DELTA: f64 = -4.0;

/// A short, unclear turn may borrow its meaning from the previous one.
const FOLLOW_UP_MAX_WORDS: usize = 4;

pub struct Session<'a> {
    model: &'a Model,
    pub memory: Memory,
    path: Option<PathBuf>,
    pub transcript: Transcript,
    /// Intent name chosen on the most recent turn (None for commands).
    pub last_intent: Option<String>,
}

/// What a turn produced, for the transcript.
struct Outcome {
    text: String,
    intent: Option<usize>,
    matched: Vec<String>,
    meta: bool,
}

impl Outcome {
    fn plain(text: String) -> Outcome {
        Outcome { text, intent: None, matched: Vec::new(), meta: false }
    }
    fn meta(text: String) -> Outcome {
        Outcome { text, intent: None, matched: Vec::new(), meta: true }
    }
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
        Session {
            model,
            memory,
            path,
            transcript: Transcript::new(DEFAULT_CONTEXT_BUDGET),
            last_intent: None,
        }
    }

    pub fn set_context_budget(&mut self, bytes: usize) {
        self.transcript = Transcript::new(bytes);
    }

    /// Handle one user turn and return the reply.
    pub fn turn(&mut self, raw: &str) -> String {
        self.memory.tick();
        let raw = raw.trim();
        self.last_intent = None;
        let out = self.dispatch(raw);
        self.last_intent = out.intent.map(|i| self.model.intent_name[i].clone());
        self.transcript.push(Turn {
            user: raw.to_string(),
            bot: out.text.clone(),
            intent: out.intent,
            matched: out.matched,
            meta: out.meta,
        });
        if let Some(p) = &self.path {
            if let Err(e) = self.memory.save(p) {
                eprintln!("warning: cannot save life file {}: {e}", p.display());
            }
        }
        out.text
    }

    fn dispatch(&mut self, raw: &str) -> Outcome {
        let norm = normalize_text(raw.as_bytes(), MAX_TEXT);
        let norm_str = String::from_utf8_lossy(&norm).into_owned();

        // Context commands: they read the transcript, not the classifier.
        if matches!(norm_str.as_str(), "again" | "repeat" | "repeat that" | "say that again") {
            return Outcome::meta(match self.transcript.recent().find(|t| !t.meta) {
                Some(t) => t.bot.clone(),
                None => "There is nothing to repeat yet.".to_string(),
            });
        }
        if matches!(norm_str.as_str(), "what did i say" | "what did i just say" | "what was my last message") {
            return Outcome::meta(match self.transcript.recent().find(|t| !t.meta) {
                Some(t) => format!("You said: {}", t.user),
                None => "You have not said anything yet.".to_string(),
            });
        }

        if let Some(rest) = strip_prefix_ci(raw, "teach ") {
            return Outcome::meta(self.cmd_teach(rest));
        }
        match norm_str.as_str() {
            "good" | "good answer" | "good bot" => return Outcome::meta(self.cmd_feedback(true)),
            "bad" | "bad answer" | "bad bot" | "wrong" => return Outcome::meta(self.cmd_feedback(false)),
            _ => {}
        }

        if let Some(rest) = strip_prefix_ci(raw, "remember ") {
            return Outcome::plain(self.cmd_remember(rest));
        }
        if let Some(rest) = strip_prefix_ci(raw, "forget ") {
            let rest = strip_prefix_ci(rest, "about ").unwrap_or(rest);
            let key = norm_string(rest);
            return Outcome::plain(if !key.is_empty() && self.memory.forget(&key) {
                format!("Forgot {key}.")
            } else {
                format!("I do not remember {key}.")
            });
        }
        if QUESTION_OPENERS.iter().any(|o| norm_str.starts_with(o)) {
            if let Some((key, value)) = self.memory.recall(&norm) {
                return Outcome::plain(format!("{key} is {value}."));
            }
        }

        let extra = self.overlay();
        let (mut intent, score) = self.model.decide_with(raw, &extra);
        let mut basis = norm.clone();
        // Follow-up: a short turn that the classifier cannot place on its own
        // is read together with the last real question.
        if score < ABSTAIN_THRESHOLD && norm_str.split(' ').count() <= FOLLOW_UP_MAX_WORDS {
            let unknown = self.model.intents - 1;
            let prev = self
                .transcript
                .recent()
                .find(|t| !t.meta && t.intent.is_some_and(|i| i != unknown))
                .map(|t| t.user.clone());
            if let Some(prev) = prev {
                let combined = format!("{prev} {raw}");
                let (ci, cs) = self.model.decide_with(&combined, &extra);
                if cs >= ABSTAIN_THRESHOLD {
                    intent = ci;
                    basis = normalize_text(combined.as_bytes(), MAX_TEXT);
                }
            }
        }
        let matched = self.model.matched_phrases(&basis, intent, &extra);

        if self.model.intent_name[intent] == "memory" {
            return Outcome {
                text: format!(
                    "I keep up to {} memories, and unused ones fade as we talk. \
Say \"remember X is Y\", ask \"what is X\", or say \"forget X\".",
                    self.memory.capacity
                ),
                intent: Some(intent),
                matched,
                meta: false,
            };
        }
        Outcome {
            text: self.model.generate(intent).join(" "),
            intent: Some(intent),
            matched,
            meta: false,
        }
    }

    /// Learned reflexes as extra model features (unknown intent names are skipped).
    fn overlay(&self) -> Vec<Feature> {
        self.memory
            .learned
            .iter()
            .filter_map(|l| {
                self.model.intent_index(&l.intent).map(|i| Feature {
                    intent: i,
                    weight: l.weight,
                    phrase: l.phrase.clone().into_bytes(),
                })
            })
            .collect()
    }

    /// `teach <phrase> as <intent>`
    fn cmd_teach(&mut self, rest: &str) -> String {
        let usage = "Say it like: teach <phrase> as <intent>.".to_string();
        let lower = rest.to_ascii_lowercase();
        let Some(i) = lower.rfind(" as ") else { return usage };
        let phrase = norm_string(&rest[..i]);
        let name = rest[i + 4..].trim().trim_end_matches(['.', '!']).to_ascii_lowercase();
        if phrase.is_empty() || name.is_empty() {
            return usage;
        }
        if self.model.intent_index(&name).is_none() {
            return format!("I have no intent called {name}. I know: {}.", self.model.intent_name.join(", "));
        }
        self.memory.learn(&name, &phrase, TEACH_WEIGHT);
        format!("Okay, \"{phrase}\" now means {name}.")
    }

    /// Hebbian-style nudge of the phrases behind the last classified answer.
    fn cmd_feedback(&mut self, good: bool) -> String {
        let last = self
            .transcript
            .recent()
            .find(|t| !t.meta && t.intent.is_some())
            .map(|t| (t.intent.unwrap_or(0), t.matched.clone()));
        let Some((intent, matched)) = last else {
            return "There is no answer to rate yet.".to_string();
        };
        if matched.is_empty() {
            return "Noted, but no phrase of mine produced that answer, so there is nothing to adjust.".to_string();
        }
        let name = self.model.intent_name[intent].clone();
        let delta = if good { GOOD_DELTA } else { BAD_DELTA };
        for p in &matched {
            self.memory.learn(&name, p, delta);
        }
        if good {
            "Thanks, I will lean that way again.".to_string()
        } else {
            "Understood, I will trust that wording less.".to_string()
        }
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

    const TINY: &str = "ZEROLM 1\nMETA 3 6\nINTENT 0 memory\nINTENT 1 weights\nINTENT 2 unknown\n\
FEATURE 0 8.0 memory\nFEATURE 1 8.0 weights\nSTART 0 0\nSTART 1 2\nSTART 2 4\n\
NODE 0 1 canned\nNODE 1 -1 <eos>\nNODE 2 3 w\nNODE 3 -1 <eos>\nNODE 4 5 sorry\nNODE 5 -1 <eos>\n";

    fn session(m: &Model) -> Session<'_> {
        Session::new(m, Memory::new(8), None)
    }

    #[test]
    fn remember_recall_forget() {
        let m = Model::parse(TINY).unwrap();
        let mut s = session(&m);
        assert!(s.turn("Remember that my name is Sam.").starts_with("Okay"));
        assert_eq!(s.turn("what is my name?"), "my name is Sam.");
        assert_eq!(s.turn("forget my name"), "Forgot my name.");
        assert_eq!(s.turn("what is my name?"), "sorry"); // falls through to unknown
    }

    #[test]
    fn memory_intent_is_live_not_canned() {
        let m = Model::parse(TINY).unwrap();
        assert!(session(&m).turn("memory").contains("8 memories"));
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
        assert!(session(&m).turn("remember nothing").starts_with("Say it like"));
    }

    #[test]
    fn repeat_and_recall_of_last_message() {
        let m = Model::parse(TINY).unwrap();
        let mut s = session(&m);
        assert_eq!(s.turn("again"), "There is nothing to repeat yet.");
        assert_eq!(s.turn("show weights"), "w");
        assert_eq!(s.turn("say that again"), "w");
        assert_eq!(s.turn("what did I say?"), "You said: show weights");
        // meta turns do not become the thing that is repeated
        assert_eq!(s.turn("again"), "w");
    }

    #[test]
    fn teach_changes_behaviour_and_validates() {
        let m = Model::parse(TINY).unwrap();
        let mut s = session(&m);
        assert_eq!(s.turn("show me the goods"), "sorry");
        assert!(s.turn("teach show me the goods as weights").contains("now means weights"));
        assert_eq!(s.turn("show me the goods"), "w");
        assert!(s.turn("teach foo as nonsense").contains("no intent called nonsense"));
        assert!(s.turn("teach foo").starts_with("Say it like"));
    }

    #[test]
    fn bad_feedback_erodes_a_reflex_and_good_strengthens() {
        let m = Model::parse(TINY).unwrap();
        let mut s = session(&m);
        assert_eq!(s.turn("bad"), "There is no answer to rate yet.");
        assert_eq!(s.turn("show weights"), "w");
        s.turn("bad");
        assert_eq!(s.turn("show weights"), "w"); // 8 - 4 = 4: still above threshold
        s.turn("bad");
        assert_eq!(s.turn("show weights"), "sorry"); // eroded below the abstain line
        // the abstained turn matched no phrase, so rating it changes nothing
        assert!(s.turn("good").contains("nothing to adjust"));
        let learned = s.memory.learned.iter().find(|l| l.phrase == "weights").unwrap();
        assert!(learned.weight < -7.0 && learned.weight > -8.0);
    }

    #[test]
    fn learned_reflexes_never_touch_the_base_model() {
        let m = Model::parse(TINY).unwrap();
        let before = format!("{:?}", m.features);
        let mut s = session(&m);
        s.turn("teach hello as weights");
        s.turn("show weights");
        s.turn("bad");
        assert_eq!(before, format!("{:?}", m.features));
    }

    #[test]
    fn short_follow_up_borrows_previous_intent() {
        let m = Model::parse(TINY).unwrap();
        let mut s = session(&m);
        assert_eq!(s.turn("yes"), "sorry"); // nothing to borrow from yet
        assert_eq!(s.turn("show weights"), "w");
        assert_eq!(s.turn("why?"), "w"); // alone: unknown; with context: weights
        assert_eq!(s.last_intent.as_deref(), Some("weights"));
        // a long unrelated sentence must not borrow context
        assert_eq!(s.turn("tell me something about the weather outside today"), "sorry");
    }
}
