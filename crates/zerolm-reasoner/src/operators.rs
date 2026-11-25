//! Inference operators as a fixed sequence of compiled circuits per question
//! form (`docs/neural-isa.md`): C6 `relation-reach` (transitive chains),
//! `relation-join` (negation), `relation-levels` (attribute inheritance),
//! `relation-rules` (conditionals), `relation-counts` (arithmetic), and
//! `relation-verdict` (how positive and negative findings combine). The host
//! picks the circuits from the parsed question's form, passes outputs on
//! unchanged, and labels the result (which operators a proof used, from the
//! circuits' proof lengths and levels). Nothing is learned, and derived
//! answers are never written back to memory.

use crate::{field, Answer, Reasoner, Verdict};
use crate::frame::{Frame, FORM_COUNT, FORM_WHICH};
use zerolm_engine::binding::Hv;
use zerolm_engine::isa::{Buf, Outputs};

/// Columns of `relation-verdict`'s one-hot output.
const VERDICTS: [Kind; 4] = [Kind::Yes, Kind::No, Kind::Contradiction, Kind::Unknown];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Yes,
    No,
    Contradiction,
    Unknown,
}

/// One side of a yes/no question: whether a proof was found, the proof as a
/// slot mask, and the operators it used.
struct Finding {
    found: f32,
    proof: Buf,
    ops: Vec<&'static str>,
}

fn total(b: &Buf) -> f32 {
    b.data.iter().sum()
}

impl Reasoner {
    fn none(&self) -> Finding {
        Finding { found: 0.0, proof: Buf::zeros(1, self.n_slots), ops: Vec::new() }
    }

    /// A C6 result as a finding: `op` when the chain is longer than one edge
    /// (or `always`), else a direct retrieval.
    fn chain(o: &Outputs, op: &'static str, always: bool) -> Finding {
        let proof = o.get("proof").clone();
        let tag = if always || total(&proof) > 1.0 { op } else { "direct" };
        Finding { found: o.scalar("hit"), proof, ops: vec![tag] }
    }

    fn finish(&self, verdict: Verdict, mask: &[f32], mut ops: Vec<&'static str>) -> Answer {
        let premises: Vec<usize> = mask.iter().enumerate().filter(|(_, &v)| v > 0.0).map(|(s, _)| s).collect();
        ops.sort_unstable();
        ops.dedup();
        let confidence = premises.iter().map(|&s| self.slot_meta(s, field::CONFIDENCE)).product();
        // A single stored statement (asserted or negated) is a retrieval, not an inference.
        let derived = ops.iter().any(|o| !matches!(*o, "direct" | "negation"));
        Answer { verdict, derived, premises, confidence, operators: ops }
    }

    fn unknown(&self) -> Answer {
        self.finish(Verdict::Unknown, &[], Vec::new())
    }

    /// Combine findings with `relation-verdict`.
    fn decide(&self, pos: Finding, neg: Finding) -> Answer {
        let o = self.run_verdict(pos.found, &pos.proof, neg.found, &neg.proof);
        let col = o.get("verdict").data.iter().position(|&v| v == 1.0).expect("one-hot verdict");
        let kind = VERDICTS[col];
        let mut ops = Vec::new();
        if matches!(kind, Kind::Yes | Kind::Contradiction) {
            ops.extend(pos.ops);
        }
        if matches!(kind, Kind::No | Kind::Contradiction) {
            ops.extend(neg.ops);
        }
        let verdict = match kind {
            Kind::Yes => Verdict::Yes,
            Kind::No => Verdict::No,
            Kind::Contradiction => Verdict::Contradiction,
            Kind::Unknown => Verdict::Unknown,
        };
        self.finish(verdict, &o.get("premises").data, ops)
    }

    /// Answer a bound question: the form code picks the circuit sequence;
    /// every vector comes from `frame-bind`.
    pub(crate) fn answer_frame(&self, f: &Frame) -> Answer {
        if self.n_slots == 0 || self.n_sym == 0 {
            return self.unknown();
        }
        let subject = if f.pronoun { self.resolve_pronoun().0 } else { f.subject.clone() };
        match f.form {
            FORM_COUNT => self.count_answer(&subject, f),
            FORM_WHICH => {
                let ans = self.holds(&f.name_a, f);
                let (a, b) = &f.choices;
                let pick = |first: bool| Verdict::Choice(if first { a.clone() } else { b.clone() });
                let verdict = match ans.verdict {
                    Verdict::Yes => pick(f.higher),
                    Verdict::No => pick(!f.higher),
                    other => other,
                };
                Answer { verdict, ..ans }
            }
            _ => self.holds(&subject, f),
        }
    }

    /// Yes/no question: the predicate code picks the circuits.
    fn holds(&self, x: &Hv, f: &Frame) -> Answer {
        let y = &f.object;
        match f.pred {
            1 => {
                let isa = self.rel("isa").clone();
                let from_x = self.run_reach(x, y, &isa, "class_inclusion");
                let from_y = self.run_reach(y, y, &isa, "class_inclusion");
                let pos = Self::chain(&from_x, "class_inclusion", false);
                let j = self.run_join(x, y, &isa, Some(&from_x), Some(&from_y));
                let mut ops = vec!["negation"];
                if j.scalar("len") > 1.0 {
                    ops.push("class_inclusion");
                }
                self.decide(pos, Finding { found: j.scalar("found"), proof: j.get("proof").clone(), ops })
            }
            2 => {
                let o = self.run_levels(x, y, self.rel("can"), self.rel("isa"));
                let tag = if o.scalar("level") > 0.0 { "attribute_inheritance" } else { "direct" };
                let pos = Finding { found: o.scalar("pos_found"), proof: o.get("pos_proof").clone(), ops: vec![tag] };
                let neg = Finding { found: o.scalar("neg_found"), proof: o.get("neg_proof").clone(), ops: vec![tag, "negation"] };
                self.decide(pos, neg)
            }
            3 => {
                let rel = self.rel("in").clone();
                let pos = Self::chain(&self.run_reach(x, y, &rel, "transitivity"), "transitivity", false);
                self.decide(pos, self.direct_negation(x, y, &rel))
            }
            4 => {
                let rel = self.rel("prop").clone();
                let o = self.run_rules(x, y, &rel, self.rel("implies"), self.rel("something"));
                let tag = if o.scalar("level") > 0.0 { "conditionals" } else { "direct" };
                let pos = Finding { found: o.scalar("found"), proof: o.get("proof").clone(), ops: vec![tag] };
                self.decide(pos, self.direct_negation(x, y, &rel))
            }
            5 => self.order(x, y, &f.rel, self.ops.comparison, "comparison"),
            6 => self.order(x, y, &self.rel("before").clone(), self.ops.temporal_ordering, "temporal_ordering"),
            _ => self.unknown(),
        }
    }

    /// A stored negative edge x -rel-> y (`relation-join` with no ancestors).
    fn direct_negation(&self, x: &Hv, y: &Hv, rel: &Hv) -> Finding {
        let j = self.run_join(x, y, rel, None, None);
        Finding { found: j.scalar("found"), proof: j.get("proof").clone(), ops: vec!["negation"] }
    }

    /// Strict order: yes if a reaches b, no if b reaches a (antisymmetry).
    /// With the operator disabled (an evaluation ablation), only a direct
    /// a -> b edge counts and the reverse direction is not tried.
    fn order(&self, a: &Hv, b: &Hv, rel: &Hv, enabled: bool, op: &'static str) -> Answer {
        let pos = Self::chain(&self.run_reach(a, b, rel, op), op, false);
        let neg = if enabled { Self::chain(&self.run_reach(b, a, rel, op), op, true) } else { self.none() };
        self.decide(pos, neg)
    }

    fn count_answer(&self, x: &Hv, f: &Frame) -> Answer {
        let o = self.run_counts(x, &f.rel, &f.rel_more);
        let ops = if o.scalar("arith") > 0.0 { vec!["arithmetic", "direct"] } else { vec!["direct"] };
        let premises = &o.get("premises").data;
        if o.scalar("found") == 0.0 {
            self.unknown()
        } else if o.scalar("contradiction") == 1.0 {
            self.finish(Verdict::Contradiction, premises, ops)
        } else {
            self.finish(Verdict::Number(o.scalar("value") as i64), premises, ops)
        }
    }
}
