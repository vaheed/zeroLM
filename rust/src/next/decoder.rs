//! Read the finished tape: row validity (a ReLU layer shared by all rows), ordering (rank by
//! time with a threshold comparator, permutation by a triangle function), and linear readouts.
//! All thresholds and class biases are weights; no grammar rule is coded here.

use super::network::{dense_relu, dot, relu, Net};

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub path: String,
    pub label: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    pub fields: Vec<Field>,
}

impl Op {
    pub fn get(&self, path: &str) -> Option<&str> {
        self.fields.iter().find(|f| f.path == path).map(|f| f.label.as_str())
    }

    /// Structured JSON: dotted paths become nested objects.
    pub fn to_json(&self) -> String {
        let mut parts: Vec<(String, Vec<(String, String)>)> = Vec::new(); // (group, members); group "" = top level
        for f in &self.fields {
            let val = if f.kind == "str" { format!("\"{}\"", f.label) } else { f.label.clone() };
            let (group, key) = match f.path.split_once('.') {
                Some((g, k)) => (g.to_string(), k.to_string()),
                None => (String::new(), f.path.clone()),
            };
            if group.is_empty() {
                parts.push((format!("\u{0}{key}"), vec![(key, val)]));
            } else if let Some(p) = parts.iter_mut().find(|p| p.0 == group) {
                p.1.push((key, val));
            } else {
                parts.push((group, vec![(key, val)]));
            }
        }
        let items: Vec<String> = parts
            .into_iter()
            .map(|(g, m)| {
                if let Some(k) = g.strip_prefix('\u{0}') {
                    format!("\"{k}\":{}", m[0].1)
                } else {
                    let inner: Vec<String> = m.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
                    format!("\"{g}\":{{{}}}", inner.join(","))
                }
            })
            .collect();
        format!("{{{}}}", items.join(","))
    }
}

pub struct DecodeSpec {
    pub path: String,
    pub kind: String,
    pub classes: Vec<String>,
}

pub struct RowInfo {
    pub index: usize,
    pub exists: f64,
    pub time: f64,
    pub rank: f64,
    pub fields: Vec<(String, f64)>,
    pub errors: Vec<(String, f64)>,
}

pub struct Decoded {
    pub accepted: bool,
    pub error_total: f64,
    pub error_units: Vec<(String, f64)>,
    pub rows: Vec<RowInfo>,
    /// per output slot: (readout path, logits per class, chosen class)
    pub readouts: Vec<Vec<(String, Vec<f64>, String)>>,
    pub ops: Vec<Op>,
}

fn clamp01(x: f64) -> f64 {
    relu(x) - relu(x - 1.0)
}

pub fn decode(net: &Net, specs: &[DecodeSpec], tape: &[f64], err_sticky: f64, n: usize, k: usize) -> Decoded {
    let fields = net.list("fields");
    let (ex_w, tf_w, tsel) = (net.t("ex_w"), net.t("tf_w"), net.t("time_sel"));
    let (vw, vb, vo) = (net.t("V_w"), net.t("V_b"), net.t("V_o"));
    let vnames = net.list("V_units");
    let mut error_units: Vec<(String, f64)> = Vec::new();
    let mut ex = vec![0.0; n];
    let mut time = vec![0.0; n];
    let mut rows = Vec::new();
    let mut verr_total = 0.0;
    for i in 0..n {
        let row = &tape[i * k..(i + 1) * k];
        ex[i] = clamp01(dot(ex_w.row(0), row));
        let tf = clamp01(dot(tf_w.row(0), row));
        time[i] = dot(tsel.row(0), row);
        let mut z = row.to_vec();
        z.push(ex[i]);
        z.push(tf);
        let h = dense_relu(vw, vb, &z);
        let mut errs = Vec::new();
        for (j, hv) in h.iter().enumerate() {
            let contrib = vo.row(0)[j] * hv;
            if contrib != 0.0 {
                errs.push((vnames[j].clone(), contrib));
                verr_total += contrib;
            }
        }
        let fl = (0..k).filter(|&c| row[c] != 0.0).map(|c| (fields[c].clone(), row[c])).collect();
        rows.push(RowInfo { index: i, exists: ex[i], time: time[i], rank: 0.0, fields: fl, errors: errs });
    }
    let empty = relu(dot(net.t("empty_w").row(0), &ex) + net.t("empty_b").data[0]);
    if empty != 0.0 {
        error_units.push(("empty_input".into(), empty));
    }
    for r in &rows {
        for (name, v) in &r.errors {
            error_units.push((format!("row{}:{name}", r.index), *v));
        }
    }
    if err_sticky != 0.0 {
        error_units.push(("recurrent_error_mass".into(), err_sticky));
    }
    let error_total = err_sticky + verr_total + empty;
    let accepted = error_total < net.t("reject_threshold").data[0];

    // rank_i = sum_j ex_j * H(slope * (t_i - t_j + eps_ij)),  H(d) = relu(d) - relu(d - 1)
    let slope = net.t("rank_slope").data[0];
    let eps = net.t("rank_eps");
    let mut rank = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            let d = slope * (time[i] - time[j] + eps.row(i)[j]);
            rank[i] += ex[j] * (relu(d) - relu(d - 1.0));
        }
        rows[i].rank = rank[i];
    }
    let count = ex.iter().sum::<f64>().round() as usize;
    let mut ops = Vec::new();
    let mut readouts = Vec::new();
    if accepted {
        for slot in 0..count.min(n) {
            // perm[slot][i] = ex_i * tri(rank_i - slot), tri(z) = relu(z+1) - 2 relu(z) + relu(z-1)
            let mut vec_k = vec![0.0; k];
            for i in 0..n {
                let z = rank[i] - slot as f64;
                let w = ex[i] * (relu(z + 1.0) - 2.0 * relu(z) + relu(z - 1.0));
                if w != 0.0 {
                    for (c, v) in vec_k.iter_mut().enumerate() {
                        *v += w * tape[i * k + c];
                    }
                }
            }
            let mut fields_out = Vec::new();
            let mut rec = Vec::new();
            for (di, spec) in specs.iter().enumerate() {
                let w = net.t(&format!("dec.{di}.w"));
                let b = net.t(&format!("dec.{di}.b"));
                let logits: Vec<f64> = w.matvec(&vec_k).iter().zip(&b.data).map(|(a, c)| a + c).collect();
                let mut best = 0;
                for (ci, l) in logits.iter().enumerate() {
                    if *l > logits[best] {
                        best = ci;
                    }
                }
                let label = spec.classes[best].clone();
                rec.push((spec.path.clone(), logits, label.clone()));
                if label != "none" {
                    fields_out.push(Field { path: spec.path.clone(), label, kind: spec.kind.clone() });
                }
            }
            readouts.push(rec);
            ops.push(Op { fields: fields_out });
        }
    }
    Decoded { accepted, error_total, error_units, rows, readouts, ops }
}
