//! Language -> semantic structures for a bounded statement language, and the
//! recurrent discourse state used to resolve "it".
//!
//! The grammar and its small lexicon of function words, comparatives, and
//! number words are programmed seed knowledge (seed entry `reasoning-grammar`).
//! Content words (entities, classes, actions, places, events, properties,
//! counted things) are open: any word that is not a function word. Parsing
//! produces role-filler structures; no sentence is stored anywhere.
//!
//! Bounded-language conventions: names do not end in "s" (a trailing "s" on a
//! plural class or counted noun is stripped), each name is one word, and
//! function words cannot be names.

use crate::binding::{cosine, Hv};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Term {
    Name(String),
    /// "it": resolved against the recurrent discourse state.
    Pronoun,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Pred {
    /// Class membership or class inclusion ("x is a y", "every x is a y").
    Isa(String),
    /// Ability ("x can fly").
    Can(String),
    /// Containment ("x is in y").
    In(String),
    /// Property ("x is wet").
    Prop(String),
    /// Strict order on a dimension: subject > other ("x is taller than y").
    Greater { dim: String, other: String },
    /// Temporal order: subject happened before other.
    Before(String),
    /// "x has n things".
    Count { thing: String, n: i64 },
    /// "x has delta more things than other" (negative delta = fewer).
    More { thing: String, delta: i64, other: String },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Fact {
    pub subj: Term,
    pub pred: Pred,
    pub negated: bool,
}

/// "if something is p then it is q" (scope None) or "if x is p then x is q".
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Rule {
    pub scope: Option<String>,
    pub premise: String,
    pub conclusion: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Query {
    /// Yes/no question about a predicate (never Count/More).
    Holds { subj: Term, pred: Pred },
    /// "how many things does x have?"
    Count { subj: Term, thing: String },
    /// "who is taller, a or b?" (`higher` false for "shorter").
    Which { dim: String, higher: bool, a: String, b: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Utterance {
    Fact(Fact),
    Rule(Rule),
    Query(Query),
}

/// Comparative word -> (dimension, true if it means "greater").
pub const COMPARATIVES: [(&str, &str, bool); 8] = [
    ("taller", "height", true),
    ("shorter", "height", false),
    ("older", "age", true),
    ("younger", "age", false),
    ("heavier", "weight", true),
    ("lighter", "weight", false),
    ("bigger", "size", true),
    ("smaller", "size", false),
];

pub const NUMBER_WORDS: [&str; 21] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
];

pub const FUNCTION_WORDS: [&str; 45] = [
    "a", "an", "the", "is", "are", "not", "no", "every", "each", "all", "can", "cannot", "in", "inside", "than", "before",
    "after", "has", "have", "owns", "own", "got", "more", "fewer", "less", "if", "then", "something", "anything", "thing",
    "it", "did", "does", "do", "how", "many", "who", "which", "or", "right", "true", "that", "able", "to", "located",
];

/// Words reserved by the grammar (function words, comparatives, number words,
/// and the verbs it normalizes). Generators must never use them as names.
pub fn is_reserved(w: &str) -> bool {
    FUNCTION_WORDS.contains(&w)
        || COMPARATIVES.iter().any(|c| c.0 == w)
        || NUMBER_WORDS.contains(&w)
        || ["happened", "happen", "came", "come", "preceded", "followed", "occurred", "unable", "lives", "so"].contains(&w)
}

pub fn tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn normalize(mut t: Vec<String>) -> Vec<String> {
    let phrases: [(&[&str], &[&str]); 21] = [
        (&["is", "it", "true", "that"], &[]),
        (&["can", "not"], &["cannot"]),
        (&["is", "unable", "to"], &["cannot"]),
        (&["is", "able", "to"], &["can"]),
        (&["able", "to"], &["can"]),
        (&["is", "located", "in"], &["is", "in"]),
        (&["inside"], &["in"]),
        (&["lives", "in"], &["is", "in"]),
        (&["happened", "before"], &["before"]),
        (&["happen", "before"], &["before"]),
        (&["came", "before"], &["before"]),
        (&["come", "before"], &["before"]),
        (&["occurred", "before"], &["before"]),
        (&["preceded"], &["before"]),
        (&["happened", "after"], &["after"]),
        (&["happen", "after"], &["after"]),
        (&["came", "after"], &["after"]),
        (&["come", "after"], &["after"]),
        (&["followed"], &["after"]),
        (&["a", "thing"], &["something"]),
        (&["anything"], &["something"]),
    ];
    for (from, to) in phrases {
        let mut i = 0;
        while i + from.len() <= t.len() {
            if t[i..i + from.len()].iter().zip(from.iter()).all(|(a, b)| a == b) {
                t.splice(i..i + from.len(), to.iter().map(|s| s.to_string()));
                i += to.len();
            } else {
                i += 1;
            }
        }
    }
    if t.last().map(String::as_str) == Some("right") {
        t.pop();
    }
    t.retain(|w| w != "the");
    t
}

fn content(w: &str) -> bool {
    !is_reserved(w)
}

fn name(w: &str) -> Option<String> {
    content(w).then(|| w.to_string())
}

/// Strip a plural "s" (names never end in "s").
fn singular(w: &str) -> Option<String> {
    let s = w.strip_suffix('s').filter(|s| s.len() >= 2).unwrap_or(w);
    name(s)
}

fn term(w: &str) -> Option<Term> {
    if w == "it" {
        Some(Term::Pronoun)
    } else {
        name(w).map(Term::Name)
    }
}

fn number(w: &str) -> Option<i64> {
    w.parse::<i64>().ok().filter(|n| (0..=1_000_000).contains(n)).or_else(|| NUMBER_WORDS.iter().position(|x| *x == w).map(|n| n as i64))
}

fn comparative(w: &str) -> Option<(&'static str, bool)> {
    COMPARATIVES.iter().find(|c| c.0 == w).map(|c| (c.1, c.2))
}

fn article(w: &str) -> bool {
    w == "a" || w == "an"
}

fn fact(subj: Term, pred: Pred, negated: bool) -> Option<Fact> {
    Some(Fact { subj, pred, negated })
}

/// Parse a declarative clause (also used for "x is a y?" questions).
fn parse_fact(t: &[&str]) -> Option<Fact> {
    use Pred::*;
    match t {
        // Class-level membership and inclusion.
        [q, c, "is", a, c2] if (*q == "every" || *q == "each") && article(a) => fact(Term::Name(name(c)?), Isa(name(c2)?), false),
        ["all", cs, "are", c2s] => fact(Term::Name(singular(cs)?), Isa(singular(c2s)?), false),
        ["no", c, "is", a, c2] if article(a) => fact(Term::Name(name(c)?), Isa(name(c2)?), true),
        ["no", cs, "are", c2s] => fact(Term::Name(singular(cs)?), Isa(singular(c2s)?), true),
        [cs, "are", c2s] if cs.ends_with('s') => fact(Term::Name(singular(cs)?), Isa(singular(c2s)?), false),
        [cs, "are", "not", c2s] if cs.ends_with('s') => fact(Term::Name(singular(cs)?), Isa(singular(c2s)?), true),
        // Abilities.
        [q, c, "can", v] if *q == "every" || *q == "each" || article(q) => fact(Term::Name(name(c)?), Can(name(v)?), false),
        [q, c, "cannot", v] if *q == "every" || *q == "each" || article(q) => fact(Term::Name(name(c)?), Can(name(v)?), true),
        ["all", cs, "can", v] => fact(Term::Name(singular(cs)?), Can(name(v)?), false),
        ["no", c, "can", v] => fact(Term::Name(singular(c)?), Can(name(v)?), true),
        [s, "can", v] => fact(subj_or_plural(s)?, Can(name(v)?), false),
        [s, "cannot", v] => fact(subj_or_plural(s)?, Can(name(v)?), true),
        // Membership, containment, comparison, property.
        [s, "is", a, c] if article(a) => fact(term(s)?, Isa(name(c)?), false),
        [s, "is", "not", a, c] if article(a) => fact(term(s)?, Isa(name(c)?), true),
        [s, "is", "in", y] => fact(term(s)?, In(name(y)?), false),
        [s, "is", "not", "in", y] => fact(term(s)?, In(name(y)?), true),
        [s, "is", cmp, "than", o] => {
            let (dim, greater) = comparative(cmp)?;
            let (s, o) = (term(s)?, name(o)?);
            if greater {
                fact(s, Greater { dim: dim.into(), other: o }, false)
            } else {
                // "x is shorter than y" == "y is taller than x" (needs a name subject).
                let Term::Name(x) = s else { return None };
                fact(Term::Name(o), Greater { dim: dim.into(), other: x }, false)
            }
        }
        [s, "before", y] => fact(term(s)?, Before(name(y)?), false),
        [s, "after", y] => {
            let Term::Name(x) = term(s)? else { return None };
            fact(Term::Name(name(y)?), Before(x), false)
        }
        [s, h, n, thing] if *h == "has" || *h == "owns" => fact(term(s)?, Count { thing: singular(thing)?, n: number(n)? }, false),
        [s, h, n, m, thing, "than", o] if *h == "has" || *h == "owns" => {
            let n = number(n)?;
            let delta = match *m {
                "more" => n,
                "fewer" | "less" => -n,
                _ => return None,
            };
            fact(term(s)?, More { thing: singular(thing)?, delta, other: name(o)? }, false)
        }
        [s, "is", p] => fact(term(s)?, Prop(name(p)?), false),
        [s, "is", "not", p] => fact(term(s)?, Prop(name(p)?), true),
        _ => None,
    }
}

fn subj_or_plural(s: &str) -> Option<Term> {
    if s == "it" {
        Some(Term::Pronoun)
    } else {
        singular(s).map(Term::Name)
    }
}

fn parse_rule(t: &[&str]) -> Option<Rule> {
    // if <s> is <p> [then] <s2> is <q>
    let rest = t.strip_prefix(&["if"])?;
    let (s, p, rest) = match rest {
        [s, "is", p, rest @ ..] => (*s, *p, rest),
        _ => return None,
    };
    let rest = rest.strip_prefix(&["then"]).unwrap_or(rest);
    let (s2, q) = match rest {
        [s2, "is", q] => (*s2, *q),
        _ => return None,
    };
    let (premise, conclusion) = (name(p)?, name(q)?);
    match (s, s2) {
        ("something", "it") => Some(Rule { scope: None, premise, conclusion }),
        (x, y) if (y == x || y == "it") && content(x) => Some(Rule { scope: Some(x.to_string()), premise, conclusion }),
        _ => None,
    }
}

fn parse_query(t: &[&str]) -> Option<Query> {
    use Pred::*;
    let holds = |subj: Term, pred: Pred| Some(Query::Holds { subj, pred });
    match t {
        ["is", s, a, c] if article(a) => holds(term(s)?, Isa(name(c)?)),
        ["can", s, v] => holds(term(s)?, Can(name(v)?)),
        ["is", s, "can", v] => holds(term(s)?, Can(name(v)?)),
        ["is", s, "in", y] => holds(term(s)?, In(name(y)?)),
        ["is", s, cmp, "than", o] => {
            let (dim, greater) = comparative(cmp)?;
            let (s, o) = (term(s)?, name(o)?);
            if greater {
                holds(s, Greater { dim: dim.into(), other: o })
            } else {
                let Term::Name(x) = s else { return None };
                holds(Term::Name(o), Greater { dim: dim.into(), other: x })
            }
        }
        ["did", s, "before", y] => holds(term(s)?, Before(name(y)?)),
        ["did", s, "after", y] => {
            let Term::Name(x) = term(s)? else { return None };
            holds(Term::Name(name(y)?), Before(x))
        }
        ["how", "many", thing, "does", s, h] if *h == "have" || *h == "own" => Some(Query::Count { subj: term(s)?, thing: singular(thing)? }),
        ["how", "many", thing, "has", s, "got"] => Some(Query::Count { subj: term(s)?, thing: singular(thing)? }),
        [w, "is", cmp, a, "or", b] if *w == "who" || *w == "which" => {
            let (dim, higher) = comparative(cmp)?;
            Some(Query::Which { dim: dim.into(), higher, a: name(a)?, b: name(b)? })
        }
        ["is", s, p] => holds(term(s)?, Prop(name(p)?)),
        _ => None,
    }
}

/// Parse one utterance. A trailing "?" (or a leading question word) makes it
/// a question; a declarative question ("x is a y?") asks whether it holds.
pub fn parse(text: &str) -> Option<Utterance> {
    let t = normalize(tokens(text));
    let t: Vec<&str> = t.iter().map(String::as_str).collect();
    let question = text.trim_end().ends_with('?')
        || matches!(t.first(), Some(&("is" | "can" | "did" | "does" | "how" | "who" | "which")));
    if question {
        if let Some(q) = parse_query(&t) {
            return Some(Utterance::Query(q));
        }
        return parse_fact(&t).and_then(|f| match f.pred {
            Pred::Count { .. } | Pred::More { .. } => None,
            pred if !f.negated => Some(Utterance::Query(Query::Holds { subj: f.subj, pred })),
            _ => None,
        });
    }
    if t.first() == Some(&"if") {
        return parse_rule(&t).map(Utterance::Rule);
    }
    parse_fact(&t).map(Utterance::Fact)
}

/// Recurrent discourse state: `c <- decay * c + subject` after each statement.
/// "it" resolves to the codebook entry closest to `c`, so the most recent
/// subject dominates. Session state only; never persisted.
#[derive(Debug, Clone)]
pub struct SemanticState {
    pub context: Hv,
    pub decay: f32,
    /// Ablation switch: when false the state stays zero and "it" never resolves.
    pub enabled: bool,
}

impl SemanticState {
    pub fn new(d: usize, decay: f32) -> Self {
        SemanticState { context: Hv::zeros(d), decay, enabled: true }
    }

    pub fn observe_subject(&mut self, subject: &Hv) {
        if !self.enabled {
            return;
        }
        self.context.scale(self.decay);
        self.context.add_assign(subject);
    }

    /// Index and cosine of the codebook row closest to the context.
    pub fn resolve<'a>(&self, candidates: impl Iterator<Item = (usize, &'a Hv)>, min_cos: f32) -> Option<(usize, f32)> {
        if self.context.is_zero() {
            return None;
        }
        let mut best: Option<(usize, f32)> = None;
        for (i, h) in candidates {
            let c = cosine(&self.context, h);
            if c >= min_cos && best.map(|b| c > b.1).unwrap_or(true) {
                best = Some((i, c));
            }
        }
        best
    }

    pub fn reset(&mut self) {
        self.context = Hv::zeros(self.context.d);
    }
}

#[cfg(test)]
mod tests {
    use super::Pred::*;
    use super::*;

    fn n(s: &str) -> Term {
        Term::Name(s.into())
    }

    fn f(text: &str) -> Fact {
        match parse(text) {
            Some(Utterance::Fact(f)) => f,
            other => panic!("{text}: {other:?}"),
        }
    }

    fn q(text: &str) -> Query {
        match parse(text) {
            Some(Utterance::Query(q)) => q,
            other => panic!("{text}: {other:?}"),
        }
    }

    #[test]
    fn statement_paraphrases_map_to_the_same_structure() {
        let isa = Fact { subj: n("bird"), pred: Isa("animal".into()), negated: false };
        for t in ["Every bird is an animal.", "each bird is an animal", "All birds are animals.", "birds are animals"] {
            assert_eq!(f(t), isa, "{t}");
        }
        let neg = Fact { subj: n("fish"), pred: Isa("mammal".into()), negated: true };
        assert_eq!(f("No fish is a mammal."), neg);
        assert_eq!(f("no fishs are mammals"), neg);
        let gt = Fact { subj: n("tom"), pred: Greater { dim: "height".into(), other: "ann".into() }, negated: false };
        assert_eq!(f("Tom is taller than Ann."), gt);
        assert_eq!(f("Ann is shorter than Tom."), gt);
        let before = Fact { subj: n("rain"), pred: Before("flood".into()), negated: false };
        for t in ["rain happened before flood", "Rain came before flood.", "rain preceded flood", "flood happened after rain", "flood followed rain"] {
            assert_eq!(f(t), before, "{t}");
        }
        let cannot = Fact { subj: n("pingu"), pred: Can("fly".into()), negated: true };
        for t in ["pingu cannot fly", "Pingu can not fly.", "pingu is unable to fly"] {
            assert_eq!(f(t), cannot, "{t}");
        }
        assert_eq!(f("Birds can fly").subj, n("bird"));
        assert_eq!(f("it is in the box"), Fact { subj: Term::Pronoun, pred: In("box".into()), negated: false });
        assert_eq!(f("tom lives in kiv").pred, In("kiv".into()));
        assert_eq!(f("tom has 3 apples").pred, Count { thing: "apple".into(), n: 3 });
        assert_eq!(f("tom has three apples").pred, Count { thing: "apple".into(), n: 3 });
        assert_eq!(f("ann has 2 fewer apples than tom").pred, More { thing: "apple".into(), delta: -2, other: "tom".into() });
        assert_eq!(f("It is wet").subj, Term::Pronoun);
        assert!(f("tom is not wet").negated);
    }

    #[test]
    fn questions_and_rules() {
        assert_eq!(q("Is tom a cat?"), Query::Holds { subj: n("tom"), pred: Isa("cat".into()) });
        assert_eq!(q("tom is a cat, right?"), Query::Holds { subj: n("tom"), pred: Isa("cat".into()) });
        assert_eq!(q("is it true that tom is a cat?"), Query::Holds { subj: n("tom"), pred: Isa("cat".into()) });
        assert_eq!(q("Is Ann shorter than Tom?"), Query::Holds { subj: n("tom"), pred: Greater { dim: "height".into(), other: "ann".into() } });
        assert_eq!(q("did flood happen after rain?"), Query::Holds { subj: n("rain"), pred: Before("flood".into()) });
        assert_eq!(q("can it fly?"), Query::Holds { subj: Term::Pronoun, pred: Can("fly".into()) });
        assert_eq!(q("is tom inside kiv?"), Query::Holds { subj: n("tom"), pred: In("kiv".into()) });
        assert_eq!(q("is tom able to fly?"), Query::Holds { subj: n("tom"), pred: Can("fly".into()) });
        assert_eq!(q("How many apples does Tom have?"), Query::Count { subj: n("tom"), thing: "apple".into() });
        assert_eq!(q("who is younger, tom or ann?"), Query::Which { dim: "age".into(), higher: false, a: "tom".into(), b: "ann".into() });
        assert_eq!(q("is tom wet?"), Query::Holds { subj: n("tom"), pred: Prop("wet".into()) });
        let rule = Rule { scope: None, premise: "wet".into(), conclusion: "cold".into() };
        for t in ["If something is wet then it is cold.", "if a thing is wet, it is cold", "if anything is wet then it is cold"] {
            assert_eq!(parse(t), Some(Utterance::Rule(rule.clone())), "{t}");
        }
        assert_eq!(parse("if tom is wet then tom is cold"), Some(Utterance::Rule(Rule { scope: Some("tom".into()), ..rule })));
        assert_eq!(parse("the cat sat on the mat"), None);
        assert_eq!(parse("is is a a?"), None);
    }
}
