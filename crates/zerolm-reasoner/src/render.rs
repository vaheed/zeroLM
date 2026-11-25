//! Diagnostic view of a recognized frame as the reference `Utterance` enum.
//!
//! Evaluation and display only (`zerolm reason --verbose`, parity tests): the
//! model path (`observe`, `ask`) binds frames with the `frame-bind` circuit
//! and never calls this module, so it is not listed in the host audit.

use crate::parser_seed::FEATURES;
use crate::Reasoner;
use zerolm_engine::isa::Buf;
use zerolm_engine::semantic_state::{Fact, Pred, Query, Rule, Term, Utterance};

impl Reasoner {
    /// The recognized clause as a reference `Utterance` (diagnostics only).
    pub fn parse(&self, text: &str) -> Option<Utterance> {
        let r = self.recognize(text)?;
        self.render_frame(&r.frame.data, &r.words, &r.features)
    }

    fn render_frame(&self, f: &[f32], words: &[String], features: &Buf) -> Option<Utterance> {
        let word = |i: usize, singular: bool| {
            let length = if singular { features.data[i * FEATURES + 14] as usize } else { words[i].len() };
            words[i][..length].to_string()
        };
        let subject = || if f[4] != 0.0 { Term::Pronoun } else { Term::Name(word(f[2] as usize, f[3] != 0.0)) };
        let dimension = || self.grammar.dimensions[f[7] as usize - 1].clone();
        let object = || word(f[5] as usize, f[6] != 0.0);
        let thing = || word(f[15] as usize, true);
        match f[0] as usize {
            0 => None,
            2 => Some(Utterance::Rule(Rule { scope: (f[11] != 0.0).then(|| word(f[2] as usize, false)), premise: word(f[12] as usize, false), conclusion: word(f[13] as usize, false) })),
            4 => Some(Utterance::Query(Query::Count { subj: subject(), thing: thing() })),
            5 => Some(Utterance::Query(Query::Which { dim: dimension(), higher: f[14] != 0.0, a: word(f[2] as usize, false), b: object() })),
            form @ (1 | 3) => {
                let pred = match f[1] as usize {
                    1 => Pred::Isa(object()), 2 => Pred::Can(object()), 3 => Pred::In(object()), 4 => Pred::Prop(object()),
                    5 => Pred::Greater { dim: dimension(), other: object() }, 6 => Pred::Before(object()),
                    7 => Pred::Count { thing: thing(), n: f[8] as i64 },
                    8 => Pred::More { thing: thing(), delta: f[8] as i64, other: object() },
                    _ => unreachable!("validated template predicate"),
                };
                Some(if form == 1 { Utterance::Fact(Fact { subj: subject(), pred, negated: f[9] != 0.0 }) } else { Utterance::Query(Query::Holds { subj: subject(), pred }) })
            }
            _ => unreachable!("validated template form"),
        }
    }
}
