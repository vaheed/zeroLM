//! Host routing for `frame-bind`. The recognized numeric frame and the atom
//! vectors of the clause's tokens (each from `atom-encode` over the token's
//! bytes) go into the circuit; its outputs are the role fillers, the codebook
//! entries, and the control row. The host copies rows the circuit's masks
//! select and uses the form code to pick the next fixed circuit sequence.

use crate::circuits::with_consts;
use crate::parser_seed::{FEATURES, TOKENS};
use crate::Reasoner;
use zerolm_engine::binding::Hv;
use zerolm_engine::isa::Buf;

/// Form codes emitted by the template table (frame field 0).
pub(crate) const FORM_FACT: usize = 1;
pub(crate) const FORM_RULE: usize = 2;
pub(crate) const FORM_HOLDS: usize = 3;
pub(crate) const FORM_COUNT: usize = 4;
pub(crate) const FORM_WHICH: usize = 5;

/// The role-filler structure of one recognized utterance.
pub(crate) struct Frame {
    pub form: usize,
    /// Effective predicate code (a count question is 7, a which-question 5).
    pub pred: usize,
    /// Metadata kind of a stored statement.
    pub kind: f32,
    pub polarity: f32,
    pub pronoun: bool,
    pub higher: bool,
    /// Filled roles with their fillers (subject first).
    pub parts: Vec<(usize, Hv)>,
    /// Vectors to enter into the symbol codebook, in order.
    pub interns: Vec<Hv>,
    pub subject: Hv,
    pub name_a: Hv,
    pub object: Hv,
    pub rel: Hv,
    pub rel_more: Hv,
    /// Bytes of the two names of a which-question, for the reply.
    pub choices: (String, String),
}

impl Reasoner {
    /// Recognize and bind one clause; `None` outside the bounded language.
    pub(crate) fn parse_frame(&self, text: &str) -> Option<Frame> {
        let r = self.recognize(text)?;
        let f = &r.frame.data;
        if f[0] == 0.0 {
            return None;
        }
        let d2 = 2 * self.cfg.dim;
        let mut atoms = vec![0.0f32; 2 * TOKENS * d2];
        for (i, word) in r.words.iter().enumerate() {
            let singular = r.features.data[i * FEATURES + 14] as usize;
            atoms[i * d2..(i + 1) * d2].copy_from_slice(&self.symbol(word).data);
            atoms[(TOKENS + i) * d2..(TOKENS + i + 1) * d2].copy_from_slice(&self.symbol(&word[..singular]).data);
        }
        let rels: Vec<f32> = self.rels.iter().flat_map(|h| h.data.iter().copied()).collect();
        let c = &self.circuits.frame;
        let inputs = vec![
            ("frame", r.frame.clone()),
            ("words", Buf::new(2 * TOKENS, d2, atoms)),
            ("dimensions", Buf::new(self.dim_atoms.rows, d2, self.dim_atoms.data.clone())),
            ("relations", Buf::new(self.rels.len(), d2, rels)),
            ("quant", Buf::row(self.numbers.encode(f[8] as i64).data)),
        ];
        let o = self.exec(c, inputs, &with_consts(c, vec![]));
        let row = |name: &str| Hv::from_data(self.cfg.dim, &o.get(name).data);
        let control = &o.get("control").data;
        let mask = &o.get("mask").data;
        let parts_buf = o.get("parts");
        let parts = (0..mask.len()).filter(|&j| mask[j] == 1.0).map(|j| (j, Hv::from_data(self.cfg.dim, &parts_buf.data[j * d2..(j + 1) * d2]))).collect();
        let imask = &o.get("intern_mask").data;
        let ibuf = o.get("interns");
        let interns = (0..imask.len()).filter(|&j| imask[j] == 1.0).map(|j| Hv::from_data(self.cfg.dim, &ibuf.data[j * d2..(j + 1) * d2])).collect();
        // Names for the reply: the clause's own bytes at the circuit's indices.
        let pick = |i: f32, singular: f32| {
            let i = i as usize;
            let n = if singular != 0.0 { r.features.data[i * FEATURES + 14] as usize } else { r.words[i].len() };
            r.words[i][..n].to_string()
        };
        let choices = if control[0] as usize == FORM_WHICH { (pick(f[2], 0.0), pick(f[5], f[6])) } else { Default::default() };
        Some(Frame {
            form: control[0] as usize,
            pred: control[1] as usize,
            kind: control[2],
            polarity: control[3],
            pronoun: control[4] != 0.0,
            higher: control[5] != 0.0,
            parts,
            interns,
            subject: row("subject"),
            name_a: row("name_a"),
            object: row("object"),
            rel: row("rel"),
            rel_more: row("rel_more"),
            choices,
        })
    }
}
