//! Controlled inference operators over the relational tensor memory.
//!
//! Every step is a vector operation: a probe (thresholded attention over
//! slots), unbinding a role, cleanup against the codebook, or binding number
//! vectors (addition). The control flow of each operator (which probe to issue
//! next, the depth limit, how polarities combine) is programmed and documented
//! here; nothing is learned. Derived answers are never written back to memory.

use crate::{field, Answer, Reasoner, Verdict, OBJ, QUANT, REL, SCOPE, SUBJ};
use std::collections::{BTreeMap, BTreeSet};
use zerolm_engine::binding::{bind, cosine, Hv};
use zerolm_engine::semantic_state::{Pred, Query};

/// A derivation: premise slots and the operators that produced it.
#[derive(Debug, Clone, Default)]
struct Proof {
    premises: Vec<usize>,
    ops: Vec<&'static str>,
}

impl Proof {
    fn of(premises: Vec<usize>, op: &'static str) -> Proof {
        Proof { premises, ops: vec![op] }
    }

    fn with(mut self, op: &'static str) -> Proof {
        self.ops.push(op);
        self
    }

    fn join(mut self, other: Proof) -> Proof {
        self.premises.extend(other.premises);
        self.ops.extend(other.ops);
        self
    }
}

/// Reached symbol -> premise slots of the shortest path to it.
type Reach = BTreeMap<usize, Vec<usize>>;

impl Reasoner {
    /// (target symbol, slot) for every stored edge `from --rel--> target` with `polarity`.
    fn edges(&self, from: usize, rel: &Hv, polarity: f32) -> Vec<(usize, usize)> {
        self.matches(&[(SUBJ, self.code(from)), (REL, rel.clone())])
            .into_iter()
            .filter(|&s| self.slot_meta(s, field::POLARITY) == polarity)
            .filter_map(|s| self.cleanup(&self.unbind_slot(s, OBJ)).map(|t| (t, s)))
            .collect()
    }

    /// Breadth-first reach over positive `rel` edges, up to `depth` steps.
    fn reach(&self, from: usize, rel: &Hv, depth: usize) -> Reach {
        let mut seen: Reach = BTreeMap::new();
        let mut frontier = vec![(from, Vec::new())];
        let mut visited = BTreeSet::from([from]);
        for _ in 0..depth {
            let mut next = Vec::new();
            for (a, path) in &frontier {
                for (t, s) in self.edges(*a, rel, 1.0) {
                    if visited.insert(t) {
                        let mut p: Vec<usize> = path.clone();
                        p.push(s);
                        seen.insert(t, p.clone());
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

    fn depth(&self, enabled: bool) -> usize {
        if enabled {
            self.cfg.max_depth
        } else {
            1
        }
    }

    fn finish(&self, verdict: Verdict, proof: Option<Proof>) -> Answer {
        let proof = proof.unwrap_or_default();
        let mut premises = proof.premises;
        premises.sort_unstable();
        premises.dedup();
        let mut ops = proof.ops;
        ops.sort_unstable();
        ops.dedup();
        let confidence = premises.iter().map(|&s| self.slot_meta(s, field::CONFIDENCE)).product();
        // A single stored statement (asserted or negated) is a retrieval, not an inference.
        let derived = ops.iter().any(|o| !matches!(*o, "direct" | "negation"));
        Answer { verdict, derived, premises, confidence, operators: ops }
    }

    fn decide(&self, pos: Option<Proof>, neg: Option<Proof>) -> Answer {
        match (pos, neg) {
            (Some(p), Some(n)) => self.finish(Verdict::Contradiction, Some(p.join(n))),
            (Some(p), None) => self.finish(Verdict::Yes, Some(p)),
            (None, Some(n)) => self.finish(Verdict::No, Some(n)),
            (None, None) => self.finish(Verdict::Unknown, None),
        }
    }

    fn unknown(&self) -> Answer {
        self.finish(Verdict::Unknown, None)
    }

    pub fn answer(&self, q: &Query) -> Answer {
        match q {
            Query::Holds { subj, pred } => {
                let Some(x) = self.resolve(subj).and_then(|(v, _)| self.find_symbol(&v)) else { return self.unknown() };
                self.holds(x, pred)
            }
            Query::Count { subj, thing } => {
                let Some(x) = self.resolve(subj).and_then(|(v, _)| self.find_symbol(&v)) else { return self.unknown() };
                self.count_answer(x, thing)
            }
            Query::Which { dim, higher, a, b } => {
                let (Some(ia), Some(_)) = (self.known(a), self.known(b)) else { return self.unknown() };
                let ans = self.holds(ia, &Pred::Greater { dim: dim.clone(), other: b.clone() });
                let pick = |first: bool| Verdict::Choice(if first { a.clone() } else { b.clone() });
                let verdict = match ans.verdict {
                    Verdict::Yes => pick(*higher),
                    Verdict::No => pick(!*higher),
                    other => other,
                };
                Answer { verdict, ..ans }
            }
        }
    }

    fn holds(&self, x: usize, pred: &Pred) -> Answer {
        let ops = self.ops;
        match pred {
            Pred::Isa(c) => {
                let Some(y) = self.known(c) else { return self.unknown() };
                let isa = self.rel("isa").clone();
                let anc = self.reach(x, &isa, self.depth(ops.class_inclusion));
                let tag = |p: &Vec<usize>| if p.len() > 1 { "class_inclusion" } else { "direct" };
                let pos = anc.get(&y).map(|p| Proof::of(p.clone(), tag(p)));
                let neg = if ops.negation {
                    let mut sources: Vec<(usize, Vec<usize>)> = vec![(x, Vec::new())];
                    let mut targets: Reach = BTreeMap::from([(y, Vec::new())]);
                    if ops.class_inclusion {
                        sources.extend(anc.iter().map(|(k, v)| (*k, v.clone())));
                        targets.extend(self.reach(y, &isa, self.cfg.max_depth));
                    }
                    let mut best: Option<Proof> = None;
                    for (a, pa) in &sources {
                        for (z, s) in self.edges(*a, &isa, -1.0) {
                            if let Some(pz) = targets.get(&z) {
                                let mut prem = pa.clone();
                                prem.push(s);
                                prem.extend(pz);
                                let mut p = Proof::of(prem, "negation");
                                if !pa.is_empty() || !pz.is_empty() {
                                    p = p.with("class_inclusion");
                                }
                                if best.as_ref().map(|b| p.premises.len() < b.premises.len()).unwrap_or(true) {
                                    best = Some(p);
                                }
                            }
                        }
                    }
                    best
                } else {
                    None
                };
                self.decide(pos, neg)
            }
            Pred::Can(v) => {
                let Some(v) = self.known(v) else { return self.unknown() };
                let (can, isa) = (self.rel("can").clone(), self.rel("isa").clone());
                let mut frontier: Vec<(usize, Vec<usize>)> = vec![(x, Vec::new())];
                let mut visited = BTreeSet::from([x]);
                for level in 0..=self.cfg.max_depth {
                    let (mut pos, mut neg): (Option<Proof>, Option<Proof>) = (None, None);
                    for (a, pa) in &frontier {
                        let found = |pol: f32| {
                            self.edges(*a, &can, pol).into_iter().find(|(t, _)| *t == v).map(|(_, s)| {
                                let mut prem = pa.clone();
                                prem.push(s);
                                Proof::of(prem, if level > 0 { "attribute_inheritance" } else { "direct" })
                            })
                        };
                        pos = pos.or_else(|| found(1.0));
                        if ops.negation {
                            neg = neg.or_else(|| found(-1.0).map(|p| p.with("negation")));
                        }
                    }
                    if pos.is_some() || neg.is_some() {
                        return self.decide(pos, neg);
                    }
                    if !ops.attribute_inheritance {
                        break;
                    }
                    let mut next = Vec::new();
                    for (a, pa) in &frontier {
                        for (t, s) in self.edges(*a, &isa, 1.0) {
                            if visited.insert(t) {
                                let mut p = pa.clone();
                                p.push(s);
                                next.push((t, p));
                            }
                        }
                    }
                    if next.is_empty() {
                        break;
                    }
                    frontier = next;
                }
                self.unknown()
            }
            Pred::In(y) => {
                let Some(y) = self.known(y) else { return self.unknown() };
                let rel = self.rel("in").clone();
                let tag = |p: &Vec<usize>| if p.len() > 1 { "transitivity" } else { "direct" };
                let pos = self.reach(x, &rel, self.depth(ops.transitivity)).get(&y).map(|p| Proof::of(p.clone(), tag(p)));
                let neg = if ops.negation {
                    self.edges(x, &rel, -1.0).into_iter().find(|(t, _)| *t == y).map(|(_, s)| Proof::of(vec![s], "negation"))
                } else {
                    None
                };
                self.decide(pos, neg)
            }
            Pred::Prop(p) => {
                let Some(p) = self.known(p) else { return self.unknown() };
                let pos = self.derive_prop(x, p, 0, &mut BTreeSet::new());
                let neg = if ops.negation {
                    let rel = self.rel("prop").clone();
                    self.edges(x, &rel, -1.0).into_iter().find(|(t, _)| *t == p).map(|(_, s)| Proof::of(vec![s], "negation"))
                } else {
                    None
                };
                self.decide(pos, neg)
            }
            Pred::Greater { dim, other } => {
                let rel = bind(self.rel("gt"), &self.space().symbol(dim));
                self.order(x, other, &rel, ops.comparison, "comparison")
            }
            Pred::Before(other) => {
                let rel = self.rel("before").clone();
                self.order(x, other, &rel, ops.temporal_ordering, "temporal_ordering")
            }
            Pred::Count { .. } | Pred::More { .. } => self.unknown(),
        }
    }

    /// Strict order: yes if a reaches b, no if b reaches a (antisymmetry).
    /// Disabled: only a direct a -> b edge counts.
    fn order(&self, a: usize, other: &str, rel: &Hv, enabled: bool, op: &'static str) -> Answer {
        let Some(b) = self.known(other) else { return self.unknown() };
        let tag = |p: &Vec<usize>| if p.len() > 1 { op } else { "direct" };
        let pos = self.reach(a, rel, self.depth(enabled)).get(&b).map(|p| Proof::of(p.clone(), tag(p)));
        let neg = if enabled { self.reach(b, rel, self.cfg.max_depth).get(&a).map(|p| Proof::of(p.clone(), op)) } else { None };
        self.decide(pos, neg)
    }

    /// Positive derivation of property `p` for `x`: a direct fact, or (with
    /// conditionals) a rule "if <scope> is q then it is p" whose premise holds.
    fn derive_prop(&self, x: usize, p: usize, depth: usize, visited: &mut BTreeSet<usize>) -> Option<Proof> {
        let prop = self.rel("prop").clone();
        if let Some((_, s)) = self.edges(x, &prop, 1.0).into_iter().find(|(t, _)| *t == p) {
            return Some(Proof::of(vec![s], if depth > 0 { "conditionals" } else { "direct" }));
        }
        if !self.ops.conditionals || depth >= self.cfg.max_depth || !visited.insert(p) {
            return None;
        }
        let something = self.rel("something").clone();
        let rules = self.matches(&[(REL, self.rel("implies").clone()), (OBJ, self.code(p))]);
        for r in rules {
            let scope = self.unbind_slot(r, SCOPE);
            let in_scope = cosine(&scope, &something) >= self.cfg.cleanup_min || cosine(&scope, &self.code(x)) >= self.cfg.cleanup_min;
            if !in_scope {
                continue;
            }
            let Some(q) = self.cleanup(&self.unbind_slot(r, SUBJ)) else { continue };
            if let Some(sub) = self.derive_prop(x, q, depth + 1, visited) {
                return Some(Proof::of(vec![r], "conditionals").join(sub));
            }
        }
        None
    }

    /// Every count of `thing` derivable for `x`, with its proof.
    fn counts(&self, x: usize, thing: &str, depth: usize, visited: &mut BTreeSet<usize>) -> Vec<(i64, Proof)> {
        let sym = self.space().symbol(thing);
        let r = self.cfg.number_range;
        let mut out: Vec<(i64, Proof)> = self
            .matches(&[(SUBJ, self.code(x)), (REL, bind(self.rel("has"), &sym))])
            .into_iter()
            .filter(|&s| self.slot_meta(s, field::POLARITY) == 1.0)
            .map(|s| (self.space().decode_number(&self.unbind_slot(s, QUANT), -r..=r).0, Proof::of(vec![s], "direct")))
            .collect();
        if !self.ops.arithmetic || depth >= self.cfg.max_depth || !visited.insert(x) {
            return out;
        }
        for s in self.matches(&[(SUBJ, self.code(x)), (REL, bind(self.rel("more"), &sym))]) {
            let Some(other) = self.cleanup(&self.unbind_slot(s, OBJ)) else { continue };
            let delta = self.space().decode_number(&self.unbind_slot(s, QUANT), -r..=r).0;
            for (c, p) in self.counts(other, thing, depth + 1, visited) {
                // Addition is binding of fractional-power number vectors.
                let sum = bind(&self.space().number(c), &self.space().number(delta));
                let (n, _) = self.space().decode_number(&sum, -r..=r);
                out.push((n, Proof::of(vec![s], "arithmetic").join(p)));
            }
        }
        visited.remove(&x);
        out
    }

    fn count_answer(&self, x: usize, thing: &str) -> Answer {
        let all = self.counts(x, thing, 0, &mut BTreeSet::new());
        let values: BTreeSet<i64> = all.iter().map(|(n, _)| *n).collect();
        match values.len() {
            0 => self.unknown(),
            1 => {
                let best = all.into_iter().min_by_key(|(_, p)| p.premises.len()).map(|(_, p)| p);
                self.finish(Verdict::Number(*values.iter().next().expect("one")), best)
            }
            _ => {
                let joined = all.into_iter().fold(Proof::default(), |acc, (_, p)| acc.join(p));
                self.finish(Verdict::Contradiction, Some(joined))
            }
        }
    }
}
