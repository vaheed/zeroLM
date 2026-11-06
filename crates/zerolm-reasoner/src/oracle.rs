//! Symbolic reference reasoner: the test oracle.
//!
//! It parses with the same bounded grammar but reasons over strings with
//! ordinary graph search, implementing the documented semantics of every
//! operator. It is used only by tests and the benchmark to label questions
//! and to choose counterfactual premises. The vector reasoner never calls it.

use crate::Verdict;
use std::collections::{BTreeMap, BTreeSet};
use zerolm_engine::semantic_state::{parse, Pred, Query, Rule, Term, Utterance};

/// A canonical stored statement with a resolved subject.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stmt {
    Fact { subj: String, pred: Pred, negated: bool },
    Rule(Rule),
}

#[derive(Debug, Clone, Default)]
pub struct Oracle {
    /// Statement -> index of its first occurrence in the input.
    pub stmts: BTreeMap<Stmt, usize>,
    last_subject: Option<String>,
    pub max_depth: usize,
    n: usize,
}

/// Verdict plus the statement indices of one proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Labeled {
    pub verdict: Verdict,
    pub premises: Vec<usize>,
    /// True when the answer needs more than one direct statement.
    pub derived: bool,
}

type Reach = BTreeMap<String, Vec<usize>>;

impl Oracle {
    pub fn new(max_depth: usize) -> Oracle {
        Oracle { max_depth, ..Default::default() }
    }

    /// Observe a statement line; returns false if it was not stored.
    pub fn observe(&mut self, text: &str) -> bool {
        let idx = self.n;
        self.n += 1;
        match parse(text) {
            Some(Utterance::Fact(f)) => {
                let subj = match f.subj {
                    Term::Name(n) => n,
                    Term::Pronoun => match &self.last_subject {
                        Some(s) => s.clone(),
                        None => return false,
                    },
                };
                self.last_subject = Some(subj.clone());
                self.stmts.entry(Stmt::Fact { subj, pred: f.pred, negated: f.negated }).or_insert(idx);
                true
            }
            Some(Utterance::Rule(r)) => {
                self.stmts.entry(Stmt::Rule(r)).or_insert(idx);
                true
            }
            _ => false,
        }
    }

    /// Subject of the most recent fact ("it").
    pub fn last_subject(&self) -> Option<&str> {
        self.last_subject.as_deref()
    }

    fn has(&self, subj: &str, pred: &Pred, negated: bool) -> Option<usize> {
        self.stmts.get(&Stmt::Fact { subj: subj.into(), pred: pred.clone(), negated }).copied()
    }

    /// Positive edges subj --kind--> target for one predicate family.
    fn edges(&self, from: &str, negated: bool, target_of: &dyn Fn(&Pred) -> Option<String>) -> Vec<(String, usize)> {
        self.stmts
            .iter()
            .filter_map(|(s, &i)| match s {
                Stmt::Fact { subj, pred, negated: n } if subj == from && *n == negated => target_of(pred).map(|t| (t, i)),
                _ => None,
            })
            .collect()
    }

    fn reach(&self, from: &str, target_of: &dyn Fn(&Pred) -> Option<String>) -> Reach {
        let mut seen = Reach::new();
        let mut visited = BTreeSet::from([from.to_string()]);
        let mut frontier = vec![(from.to_string(), Vec::new())];
        for _ in 0..self.max_depth {
            let mut next = Vec::new();
            for (a, path) in &frontier {
                for (t, i) in self.edges(a, false, target_of) {
                    if visited.insert(t.clone()) {
                        let mut p: Vec<usize> = path.clone();
                        p.push(i);
                        seen.insert(t.clone(), p.clone());
                        next.push((t, p));
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        seen
    }

    fn resolve(&self, t: &Term) -> Option<String> {
        match t {
            Term::Name(n) => Some(n.clone()),
            Term::Pronoun => self.last_subject.clone(),
        }
    }

    fn label(verdict: Verdict, mut premises: Vec<usize>, derived: bool) -> Labeled {
        premises.sort_unstable();
        premises.dedup();
        Labeled { verdict, premises, derived }
    }

    fn decide(pos: Option<(Vec<usize>, bool)>, neg: Option<(Vec<usize>, bool)>) -> Labeled {
        match (pos, neg) {
            (Some((p, dp)), Some((n, dn))) => Self::label(Verdict::Contradiction, [p, n].concat(), dp || dn),
            (Some((p, d)), None) => Self::label(Verdict::Yes, p, d),
            (None, Some((n, d))) => Self::label(Verdict::No, n, d),
            (None, None) => Self::label(Verdict::Unknown, Vec::new(), false),
        }
    }

    pub fn answer(&self, q: &Query) -> Labeled {
        let unknown = Self::label(Verdict::Unknown, Vec::new(), false);
        match q {
            Query::Holds { subj, pred } => match self.resolve(subj) {
                Some(x) => self.holds(&x, pred),
                None => unknown,
            },
            Query::Count { subj, thing } => match self.resolve(subj) {
                Some(x) => {
                    let all = self.counts(&x, thing, 0, &mut BTreeSet::new());
                    let values: BTreeSet<i64> = all.iter().map(|(n, _, _)| *n).collect();
                    match values.len() {
                        0 => unknown,
                        1 => {
                            let (n, p, d) = all.into_iter().min_by_key(|(_, p, _)| p.len()).expect("one");
                            Self::label(Verdict::Number(n), p, d)
                        }
                        _ => Self::label(Verdict::Contradiction, all.iter().flat_map(|(_, p, _)| p.clone()).collect(), true),
                    }
                }
                None => unknown,
            },
            Query::Which { dim, higher, a, b } => {
                let l = self.holds(a, &Pred::Greater { dim: dim.clone(), other: b.clone() });
                let pick = |first: bool| Verdict::Choice(if first { a.clone() } else { b.clone() });
                let verdict = match l.verdict {
                    Verdict::Yes => pick(*higher),
                    Verdict::No => pick(!*higher),
                    v => v,
                };
                Labeled { verdict, ..l }
            }
        }
    }

    fn holds(&self, x: &str, pred: &Pred) -> Labeled {
        let isa = |p: &Pred| match p {
            Pred::Isa(c) => Some(c.clone()),
            _ => None,
        };
        match pred {
            Pred::Isa(y) => {
                let anc = self.reach(x, &isa);
                let pos = anc.get(y).map(|p| (p.clone(), p.len() > 1));
                let mut sources: Vec<(String, Vec<usize>)> = vec![(x.to_string(), Vec::new())];
                sources.extend(anc.iter().map(|(k, v)| (k.clone(), v.clone())));
                let mut targets = self.reach(y, &isa);
                targets.insert(y.clone(), Vec::new());
                let mut neg: Option<(Vec<usize>, bool)> = None;
                for (a, pa) in &sources {
                    for (z, i) in self.edges(a, true, &isa) {
                        if let Some(pz) = targets.get(&z) {
                            let p = [pa.clone(), vec![i], pz.clone()].concat();
                            if neg.as_ref().map(|n| p.len() < n.0.len()).unwrap_or(true) {
                                let d = p.len() > 1;
                                neg = Some((p, d));
                            }
                        }
                    }
                }
                Self::decide(pos, neg)
            }
            Pred::Can(v) => {
                let can = |p: &Pred| match p {
                    Pred::Can(a) => Some(a.clone()),
                    _ => None,
                };
                let mut frontier = vec![(x.to_string(), Vec::new())];
                let mut visited = BTreeSet::from([x.to_string()]);
                for level in 0..=self.max_depth {
                    let (mut pos, mut neg) = (None, None);
                    for (a, pa) in &frontier {
                        let found = |negated: bool| {
                            self.edges(a, negated, &can).into_iter().find(|(t, _)| t == v).map(|(_, i)| ([pa.clone(), vec![i]].concat(), level > 0))
                        };
                        pos = pos.or_else(|| found(false));
                        neg = neg.or_else(|| found(true));
                    }
                    if pos.is_some() || neg.is_some() {
                        return Self::decide(pos, neg);
                    }
                    let mut next = Vec::new();
                    for (a, pa) in &frontier {
                        for (t, i) in self.edges(a, false, &isa) {
                            if visited.insert(t.clone()) {
                                next.push((t, [pa.clone(), vec![i]].concat()));
                            }
                        }
                    }
                    if next.is_empty() {
                        break;
                    }
                    frontier = next;
                }
                Self::decide(None, None)
            }
            Pred::In(y) => {
                let inn = |p: &Pred| match p {
                    Pred::In(a) => Some(a.clone()),
                    _ => None,
                };
                let pos = self.reach(x, &inn).get(y).map(|p| (p.clone(), p.len() > 1));
                let neg = self.has(x, pred, true).map(|i| (vec![i], false));
                Self::decide(pos, neg)
            }
            Pred::Prop(p) => {
                let pos = self.derive_prop(x, p, 0, &mut BTreeSet::new());
                let neg = self.has(x, pred, true).map(|i| (vec![i], false));
                Self::decide(pos.map(|v| (v.0, v.1)), neg)
            }
            Pred::Greater { dim, other } => {
                let gt = |p: &Pred| match p {
                    Pred::Greater { dim: d, other } if d == dim => Some(other.clone()),
                    _ => None,
                };
                self.order(x, other, &gt)
            }
            Pred::Before(other) => {
                let before = |p: &Pred| match p {
                    Pred::Before(o) => Some(o.clone()),
                    _ => None,
                };
                self.order(x, other, &before)
            }
            Pred::Count { .. } | Pred::More { .. } => Self::decide(None, None),
        }
    }

    fn order(&self, a: &str, b: &str, target_of: &dyn Fn(&Pred) -> Option<String>) -> Labeled {
        let pos = self.reach(a, target_of).get(b).map(|p| (p.clone(), p.len() > 1));
        let neg = self.reach(b, target_of).get(a).map(|p| (p.clone(), true));
        Self::decide(pos, neg)
    }

    fn derive_prop(&self, x: &str, p: &str, depth: usize, visited: &mut BTreeSet<String>) -> Option<(Vec<usize>, bool)> {
        if let Some(i) = self.has(x, &Pred::Prop(p.into()), false) {
            return Some((vec![i], depth > 0));
        }
        if depth >= self.max_depth || !visited.insert(p.to_string()) {
            return None;
        }
        for (s, &i) in &self.stmts {
            let Stmt::Rule(r) = s else { continue };
            if r.conclusion != p || r.scope.as_deref().map(|sc| sc != x).unwrap_or(false) {
                continue;
            }
            if let Some((sub, _)) = self.derive_prop(x, &r.premise, depth + 1, visited) {
                return Some(([vec![i], sub].concat(), true));
            }
        }
        None
    }

    fn counts(&self, x: &str, thing: &str, depth: usize, visited: &mut BTreeSet<String>) -> Vec<(i64, Vec<usize>, bool)> {
        let mut out: Vec<(i64, Vec<usize>, bool)> = self
            .stmts
            .iter()
            .filter_map(|(s, &i)| match s {
                Stmt::Fact { subj, pred: Pred::Count { thing: t, n }, negated: false } if subj == x && t == thing => Some((*n, vec![i], false)),
                _ => None,
            })
            .collect();
        if depth >= self.max_depth || !visited.insert(x.to_string()) {
            return out;
        }
        let more: Vec<(String, i64, usize)> = self
            .stmts
            .iter()
            .filter_map(|(s, &i)| match s {
                Stmt::Fact { subj, pred: Pred::More { thing: t, delta, other }, negated: false } if subj == x && t == thing => {
                    Some((other.clone(), *delta, i))
                }
                _ => None,
            })
            .collect();
        for (other, delta, i) in more {
            for (c, p, _) in self.counts(&other, thing, depth + 1, visited) {
                out.push((c + delta, [vec![i], p].concat(), true));
            }
        }
        visited.remove(x);
        out
    }
}
