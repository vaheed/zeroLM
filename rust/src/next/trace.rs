//! Forward-pass trace: every intermediate vector of a parse, with units named, as JSON.

use super::decoder::Decoded;
use crate::jsonl::escape;

pub struct StepTrace {
    pub token: String,
    pub known: bool,
    pub x: Vec<(String, f64)>,
    pub r: Vec<(String, f64)>,
    pub hidden_a: Vec<(String, f64)>,
    pub adv: f64,
    pub rev: f64,
    pub hidden_b: Vec<(String, f64)>,
    pub write: Vec<(String, f64)>,
    pub pointer: Option<usize>,
    pub max_time: f64,
    pub error_added: f64,
    pub row_after: Vec<(String, f64)>,
}

impl StepTrace {
    pub fn new(token: String, known: bool) -> StepTrace {
        StepTrace { token, known, x: vec![], r: vec![], hidden_a: vec![], adv: 0.0, rev: 0.0, hidden_b: vec![],
            write: vec![], pointer: None, max_time: 0.0, error_added: 0.0, row_after: vec![] }
    }
}

pub struct Trace {
    pub steps: Vec<StepTrace>,
}

pub fn num(v: f64) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

pub fn obj(v: &[(String, f64)]) -> String {
    let items: Vec<String> = v.iter().map(|(k, x)| format!("\"{}\":{}", escape(k), num(*x))).collect();
    format!("{{{}}}", items.join(","))
}

impl Trace {
    pub fn to_json(&self, tokens: &[String], d: &Decoded) -> String {
        let steps: Vec<String> = self
            .steps
            .iter()
            .map(|s| {
                format!(
                    "{{\"token\":\"{}\",\"in_lexicon\":{},\"embedding\":{},\"unbound_row\":{},\"boundary_units\":{},\"advance\":{},\"reverse\":{},\"write_units\":{},\"write_vector\":{},\"pointer_row\":{},\"row_after_write\":{},\"max_time\":{},\"error_added\":{}}}",
                    escape(&s.token), s.known, obj(&s.x), obj(&s.r), obj(&s.hidden_a), num(s.adv), num(s.rev),
                    obj(&s.hidden_b), obj(&s.write),
                    s.pointer.map_or("null".to_string(), |p| p.to_string()), obj(&s.row_after), num(s.max_time), num(s.error_added)
                )
            })
            .collect();
        let rows: Vec<String> = d
            .rows
            .iter()
            .filter(|r| r.exists > 0.0 || !r.errors.is_empty())
            .map(|r| {
                format!(
                    "{{\"row\":{},\"exists\":{},\"time\":{},\"rank\":{},\"fields\":{},\"validity_errors\":{}}}",
                    r.index, num(r.exists), num(r.time), num(r.rank), obj(&r.fields), obj(&r.errors)
                )
            })
            .collect();
        let slots: Vec<String> = d
            .readouts
            .iter()
            .enumerate()
            .map(|(i, rec)| {
                let items: Vec<String> = rec
                    .iter()
                    .map(|(p, logits, c)| {
                        let ls: Vec<String> = logits.iter().map(|v| num(*v)).collect();
                        format!("{{\"readout\":\"{p}\",\"logits\":[{}],\"argmax\":\"{c}\"}}", ls.join(","))
                    })
                    .collect();
                format!("{{\"slot\":{i},\"readouts\":[{}]}}", items.join(","))
            })
            .collect();
        let toks: Vec<String> = tokens.iter().map(|t| format!("\"{}\"", escape(t))).collect();
        let ops: Vec<String> = d.ops.iter().map(|o| o.to_json()).collect();
        format!(
            "{{\"tokens\":[{}],\"steps\":[{}],\"rows\":[{}],\"error_units\":{},\"error_total\":{},\"accepted\":{},\"output_slots\":[{}],\"ops\":[{}]}}",
            toks.join(","), steps.join(","), rows.join(","), obj(&d.error_units), num(d.error_total), d.accepted,
            slots.join(","), ops.join(",")
        )
    }
}
