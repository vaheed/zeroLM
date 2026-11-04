//! Rolling transcript with a byte budget: the "whole context" the live model sees.
//! The oldest turns are dropped first; the newest turn is always kept.

use std::collections::VecDeque;

pub const DEFAULT_CONTEXT_BUDGET: usize = 16 * 1024;

#[derive(Debug, Clone)]
pub struct Turn {
    pub user: String,
    pub bot: String,
    /// Classified intent, if the turn went through the classifier.
    pub intent: Option<usize>,
    /// Phrases that fired for `intent` (used for feedback).
    pub matched: Vec<String>,
    /// Meta turns (again / what did I say / good / bad) are skipped when
    /// looking back for something to repeat or rate.
    pub meta: bool,
}

#[derive(Debug, Clone)]
pub struct Transcript {
    budget: usize,
    used: usize,
    turns: VecDeque<Turn>,
}

impl Transcript {
    pub fn new(budget: usize) -> Transcript {
        Transcript { budget: budget.max(1), used: 0, turns: VecDeque::new() }
    }

    fn cost(t: &Turn) -> usize {
        t.user.len() + t.bot.len()
    }

    pub fn push(&mut self, t: Turn) {
        self.used += Self::cost(&t);
        self.turns.push_back(t);
        while self.used > self.budget && self.turns.len() > 1 {
            if let Some(old) = self.turns.pop_front() {
                self.used -= Self::cost(&old);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Newest first.
    pub fn recent(&self) -> impl Iterator<Item = &Turn> {
        self.turns.iter().rev()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(u: &str) -> Turn {
        Turn { user: u.into(), bot: "ok".into(), intent: None, matched: vec![], meta: false }
    }

    #[test]
    fn drops_oldest_over_budget_but_keeps_newest() {
        let mut t = Transcript::new(20);
        for u in ["aaaaaaaa", "bbbbbbbb", "cccccccc"] {
            t.push(turn(u));
        }
        assert_eq!(t.recent().next().unwrap().user, "cccccccc");
        assert!(t.len() < 3);
        let mut tiny = Transcript::new(1);
        tiny.push(turn("a very long single turn"));
        assert_eq!(tiny.len(), 1);
    }
}
