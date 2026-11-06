//! Lookup baseline: answers only what was stated verbatim (after parsing and
//! pronoun resolution, which it gets for free). It cannot chain, inherit,
//! compute, or apply rules. Evaluation only.

use crate::oracle::{Oracle, Stmt};
use crate::Verdict;
use zerolm_engine::semantic_state::{Pred, Query, Term};

pub struct Lookup<'a> {
    pub oracle: &'a Oracle,
}

impl Lookup<'_> {
    fn stated(&self, subj: &str, pred: &Pred, negated: bool) -> bool {
        self.oracle.stmts.contains_key(&Stmt::Fact { subj: subj.into(), pred: pred.clone(), negated })
    }

    pub fn answer(&self, q: &Query, last_subject: Option<&str>) -> Verdict {
        let name = |t: &Term| match t {
            Term::Name(n) => Some(n.clone()),
            Term::Pronoun => last_subject.map(str::to_string),
        };
        match q {
            Query::Holds { subj, pred } => {
                let Some(x) = name(subj) else { return Verdict::Unknown };
                match (self.stated(&x, pred, false), self.stated(&x, pred, true)) {
                    (true, true) => Verdict::Contradiction,
                    (true, false) => Verdict::Yes,
                    (false, true) => Verdict::No,
                    (false, false) => Verdict::Unknown,
                }
            }
            Query::Count { subj, thing } => {
                let Some(x) = name(subj) else { return Verdict::Unknown };
                let found: Vec<i64> = self
                    .oracle
                    .stmts
                    .keys()
                    .filter_map(|s| match s {
                        Stmt::Fact { subj, pred: Pred::Count { thing: t, n }, negated: false } if *subj == x && t == thing => Some(*n),
                        _ => None,
                    })
                    .collect();
                match found.as_slice() {
                    [n] => Verdict::Number(*n),
                    [] => Verdict::Unknown,
                    _ => Verdict::Contradiction,
                }
            }
            Query::Which { dim, higher, a, b } => {
                let gt = |x: &str, y: &str| self.stated(x, &Pred::Greater { dim: dim.clone(), other: y.into() }, false);
                let pick = |first: bool| Verdict::Choice(if first { a.clone() } else { b.clone() });
                match (gt(a, b), gt(b, a)) {
                    (true, false) => pick(*higher),
                    (false, true) => pick(!*higher),
                    (true, true) => Verdict::Contradiction,
                    _ => Verdict::Unknown,
                }
            }
        }
    }
}
