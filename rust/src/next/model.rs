//! The ZeroLM Next forward pass.
//!
//! One recurrent step (token x, state = tape T, pointer p, max-time m, sticky error e):
//!   r        = p^T T                                   unbind the current row
//!   h_A      = relu(A_w [x; r] + A_b)                  row-boundary detector
//!   (adv,rev)= A_o h_A
//!   r'       = r (.) (1 - adv)                         a new row starts empty
//!   p'       = p + adv (S p - p)                       pointer advances one row
//!   h_B      = relu(B_w [x; r'; r; adv; rev; m] + B_b) role-gated writes, timing, error flags
//!   u        = B_o h_B
//!   T       += p' (x) u                                bind the write to the current row
//!   m       += relu(Gm [u; m])                         running maximum of row time
//!   e       += B_e h_B + relu(Pe p' + 1)
//! Everything above is arithmetic on weights from the file; the Rust code never inspects a token.

use super::binding::{advance, bind, unbind};
use super::decoder::{decode, DecodeSpec, Decoded};
pub use super::decoder::{Field, Op};
use super::encoder::{tokenize, Encoder, MAX_TOKENS};
use super::network::{dense_relu, dot, relu, Net};
use super::trace::{StepTrace, Trace};

#[derive(Debug, Clone, Default)]
pub struct Ablation {
    /// Reset the state before every token (no recurrence at all).
    pub reset_each_token: bool,
    /// Layers see r = 0 and m = 0: the recurrent state is written but never read back.
    pub state_read_off: bool,
    /// Replace the one-hot pointer by a uniform vector (writes smear across all rows).
    pub no_binding: bool,
    /// Weight-space surgeries, applied once at load.
    pub surgeries: Vec<Surgery>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Surgery {
    Zero(String),
    Permute(String, u64),
    Shuffle(u64),
    Random(u64),
    ShiftToIdentity,
    NoTime,
}

impl Ablation {
    pub fn parse(specs: &[String]) -> Result<Ablation, String> {
        let mut a = Ablation::default();
        for s in specs {
            let mut it = s.splitn(3, ':');
            let kind = it.next().unwrap_or("");
            let arg = it.next();
            let seed = |v: Option<&str>| v.and_then(|x| x.parse::<u64>().ok()).unwrap_or(1);
            match kind {
                "recurrence" => a.reset_each_token = true,
                "state_read" => a.state_read_off = true,
                "binding" => a.no_binding = true,
                "pointer_shift" => a.surgeries.push(Surgery::ShiftToIdentity),
                "time_order" => a.surgeries.push(Surgery::NoTime),
                "zero" => a.surgeries.push(Surgery::Zero(arg.ok_or("zero:TENSOR")?.to_string())),
                "permute" => a.surgeries.push(Surgery::Permute(arg.ok_or("permute:TENSOR[:SEED]")?.to_string(), seed(it.next()))),
                "shuffle" => a.surgeries.push(Surgery::Shuffle(seed(arg))),
                "random" => a.surgeries.push(Surgery::Random(seed(arg))),
                other => return Err(format!("unknown ablation {other}")),
            }
            a.labels.push(s.clone());
        }
        Ok(a)
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit().max(1e-12), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn apply_surgery(net: &mut Net, s: &Surgery) -> Result<(), String> {
    let mut names: Vec<String> = net.tensors.keys().cloned().collect();
    names.sort();
    match s {
        Surgery::Zero(n) => {
            net.tensors.get_mut(n).ok_or(format!("no tensor {n}"))?.data.iter_mut().for_each(|v| *v = 0.0);
        }
        Surgery::Permute(n, seed) => {
            let t = net.tensors.get_mut(n).ok_or(format!("no tensor {n}"))?;
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            let mut order: Vec<usize> = (0..t.rows).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i + 1));
            }
            let old = t.data.clone();
            for (new_i, &old_i) in order.iter().enumerate() {
                t.data[new_i * t.cols..(new_i + 1) * t.cols].copy_from_slice(&old[old_i * t.cols..(old_i + 1) * t.cols]);
            }
        }
        Surgery::Shuffle(seed) => {
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            for n in &names {
                let t = net.tensors.get_mut(n).unwrap();
                for i in (1..t.data.len()).rev() {
                    t.data.swap(i, rng.below(i + 1));
                }
            }
        }
        Surgery::Random(seed) => {
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            for n in &names {
                let t = net.tensors.get_mut(n).unwrap();
                let rms = (t.data.iter().map(|v| v * v).sum::<f64>() / t.data.len().max(1) as f64).sqrt();
                for v in t.data.iter_mut() {
                    *v = rng.gauss() * rms;
                }
            }
        }
        Surgery::ShiftToIdentity => {
            let t = net.tensors.get_mut("S").ok_or("no tensor S")?;
            for i in 0..t.rows {
                for j in 0..t.cols {
                    t.data[i * t.cols + j] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }
        Surgery::NoTime => {
            let units = net.list("B_units").to_vec();
            let k = net.dim("K");
            let tidx = net.list("fields").iter().position(|f| f == "TIME").ok_or("no TIME field")?;
            let t = net.tensors.get_mut("B_o").ok_or("no tensor B_o")?;
            for (j, u) in units.iter().enumerate() {
                if u.starts_with("time_") {
                    t.data[tidx * t.cols + j] = 0.0;
                }
            }
            let _ = k;
        }
    }
    Ok(())
}

pub struct State {
    pub tape: Vec<f64>,
    pub p: Vec<f64>,
    pub m: f64,
    pub err: f64,
}

pub struct Parse {
    pub tokens: Vec<String>,
    pub decoded: Decoded,
    pub trace: Option<Trace>,
}

pub struct NextModel {
    pub net: Net,
    pub ablation: Ablation,
    enc: Encoder,
    specs: Vec<DecodeSpec>,
    pub n: usize,
    pub k: usize,
    pub d: usize,
}

const REQUIRED: &[(&str, usize)] = &[("E", 0), ("A_w", 0), ("A_b", 0), ("A_o", 0), ("B_w", 0), ("B_b", 0), ("B_o", 0), ("B_e", 0),
    ("S", 0), ("T0", 0), ("p0", 0), ("m0", 0), ("Gm_w", 0), ("Gm_b", 0), ("Pe_w", 0), ("Pe_b", 0), ("V_w", 0), ("V_b", 0),
    ("V_o", 0), ("ex_w", 0), ("tf_w", 0), ("time_sel", 0), ("empty_w", 0), ("empty_b", 0), ("rank_eps", 0), ("rank_slope", 0),
    ("reject_threshold", 0)];

impl NextModel {
    pub fn load(path: &str, ablation: Ablation) -> Result<NextModel, String> {
        NextModel::from_net(Net::load(path)?, ablation)
    }

    pub fn from_net(mut net: Net, ablation: Ablation) -> Result<NextModel, String> {
        for (name, _) in REQUIRED {
            if !net.tensors.contains_key(*name) {
                return Err(format!("weight file lacks tensor {name}"));
            }
        }
        for l in ["vocab", "features", "fields", "A_units", "B_units", "V_units", "decode_paths", "decode_kinds"] {
            if !net.lists.contains_key(l) {
                return Err(format!("weight file lacks list {l}"));
            }
        }
        let (d, k, n) = (net.dim("D"), net.dim("K"), net.dim("N"));
        let shape = |name: &str, r: usize, c: usize| -> Result<(), String> {
            let t = net.t(name);
            if t.rows != r || t.cols != c {
                return Err(format!("tensor {name} is {}x{}, expected {r}x{c}", t.rows, t.cols));
            }
            Ok(())
        };
        let (ha, hb, hv) = (net.list("A_units").len(), net.list("B_units").len(), net.list("V_units").len());
        shape("E", net.list("vocab").len(), d)?;
        shape("A_w", ha, d + k)?;
        shape("A_b", 1, ha)?;
        shape("A_o", 2, ha)?;
        shape("B_w", hb, d + 2 * k + 3)?;
        shape("B_b", 1, hb)?;
        shape("B_o", k, hb)?;
        shape("B_e", 1, hb)?;
        shape("S", n, n)?;
        shape("T0", n, k)?;
        shape("p0", 1, n)?;
        shape("Gm_w", 1, k + 1)?;
        shape("Pe_w", 1, n)?;
        shape("V_w", hv, k + 2)?;
        shape("V_o", 1, hv)?;
        shape("ex_w", 1, k)?;
        shape("tf_w", 1, k)?;
        shape("time_sel", 1, k)?;
        shape("rank_eps", n, n)?;
        shape("empty_w", 1, n)?;
        let paths = net.list("decode_paths").to_vec();
        let kinds = net.list("decode_kinds").to_vec();
        let mut specs = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            let classes = net.lists.get(&format!("dec.{i}.classes")).ok_or(format!("no classes for decode {i}"))?.clone();
            shape(&format!("dec.{i}.w"), classes.len(), k)?;
            shape(&format!("dec.{i}.b"), 1, classes.len())?;
            specs.push(DecodeSpec { path: p.clone(), kind: kinds[i].clone(), classes });
        }
        for s in &ablation.surgeries {
            apply_surgery(&mut net, s)?;
        }
        let enc = Encoder::new(&net)?;
        Ok(NextModel { net, ablation, enc, specs, n, k, d })
    }

    fn init_state(&self) -> State {
        State {
            tape: self.net.t("T0").data.clone(),
            p: self.net.t("p0").data.clone(),
            m: self.net.t("m0").data[0],
            err: 0.0,
        }
    }

    pub fn embed(&self, tok: &str) -> &[f64] {
        self.net.t("E").row(self.enc.id(tok))
    }

    pub fn step(&self, st: &mut State, x: &[f64], rec: Option<&mut StepTrace>) {
        let net = &self.net;
        let (n, k) = (self.n, self.k);
        let uniform = vec![1.0 / n as f64; n];
        let p_eff: &[f64] = if self.ablation.no_binding { &uniform } else { &st.p };
        let mut r = unbind(&st.tape, p_eff, k);
        let mut m_in = st.m;
        if self.ablation.state_read_off {
            r.iter_mut().for_each(|v| *v = 0.0);
            m_in = 0.0;
        }
        // row-boundary layer
        let mut za = x.to_vec();
        za.extend(&r);
        let ha = dense_relu(net.t("A_w"), net.t("A_b"), &za);
        let ao = net.t("A_o").matvec(&ha);
        let (adv, rev) = (ao[0], ao[1]);
        let rp: Vec<f64> = r.iter().map(|v| v * (1.0 - adv)).collect();
        let p_new = if self.ablation.no_binding { uniform.clone() } else { advance(&st.p, net.t("S"), adv) };
        // write layer
        let mut zb = x.to_vec();
        zb.extend(&rp);
        zb.extend(&r);
        zb.extend([adv, rev, m_in]);
        let hb = dense_relu(net.t("B_w"), net.t("B_b"), &zb);
        let u = net.t("B_o").matvec(&hb);
        let err_inc = dot(net.t("B_e").row(0), &hb);
        bind(&mut st.tape, &p_new, &u, k);
        let mut gz = u.clone();
        gz.push(st.m);
        st.m += relu(dot(net.t("Gm_w").row(0), &gz) + net.t("Gm_b").data[0]);
        let ovf = relu(dot(net.t("Pe_w").row(0), &p_new) + net.t("Pe_b").data[0]);
        st.err += err_inc + ovf;
        st.p = p_new;
        if let Some(t) = rec {
            let (f, a, b) = (net.list("fields"), net.list("A_units"), net.list("B_units"));
            let named = |names: &[String], v: &[f64]| -> Vec<(String, f64)> {
                v.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, x)| (names[i].clone(), *x)).collect()
            };
            t.x = named(net.list("features"), x);
            t.r = named(f, &r);
            t.hidden_a = named(a, &ha);
            t.adv = adv;
            t.rev = rev;
            t.hidden_b = named(b, &hb);
            t.write = named(f, &u);
            t.pointer = st.p.iter().position(|v| *v > 0.5);
            t.max_time = st.m;
            t.error_added = err_inc + ovf;
            if let Some(i) = t.pointer {
                t.row_after = named(f, &st.tape[i * k..(i + 1) * k]);
            }
        }
    }

    pub fn parse(&self, text: &str, want_trace: bool) -> Parse {
        let tokens = tokenize(text);
        let mut st = self.init_state();
        let mut steps = Vec::new();
        let too_long = tokens.len() > MAX_TOKENS;
        if !too_long {
            for tok in &tokens {
                if self.ablation.reset_each_token {
                    st = self.init_state();
                }
                let x = self.embed(tok).to_vec();
                if want_trace {
                    let mut rec = StepTrace::new(tok.clone(), self.enc.known(tok));
                    self.step(&mut st, &x, Some(&mut rec));
                    steps.push(rec);
                } else {
                    self.step(&mut st, &x, None);
                }
            }
        } else {
            st.err = 1.0e6; // input longer than the model's declared maximum
        }
        let decoded = decode(&self.net, &self.specs, &st.tape, st.err, self.n, self.k);
        let trace = if want_trace { Some(Trace { steps }) } else { None };
        Parse { tokens, decoded, trace }
    }
}
