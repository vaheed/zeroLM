//! Reference interpreter for the neural ISA (constitution invariant 11).
//!
//! Written separately from `zerolm_engine::isa` and deliberately minimal:
//! it parses a program bundle from raw bytes with `serde_json::Value`, keeps
//! tensors in a name map, and runs each operation with the plain loops given
//! in `docs/NEURAL_ISA.md`. It shares no code with the engine and contains no
//! language-specific logic. If it reproduces the engine's outputs bit for bit
//! from the exported bundle, the behavior lives in the program and weights.

// Plain index loops mirror the spec text in docs/NEURAL_ISA.md.
#![allow(clippy::needless_range_loop)]

use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct T {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f32>,
}

impl T {
    fn at(&self, i: usize, j: usize) -> f32 {
        self.d[i * self.c + j]
    }
}

type Env = BTreeMap<String, T>;

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Parse a bundle into its program JSON and tensors. The trailing SHA-256 is
/// not re-verified here (the engine checks it); a truncated bundle fails.
pub fn load(bytes: &[u8]) -> Result<(Value, Env), String> {
    if bytes.len() < 44 || &bytes[..8] != b"ZLMPROG1" {
        return Err("not a program bundle".into());
    }
    let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let header: Value = serde_json::from_slice(bytes.get(12..12 + hl).ok_or("short header")?).map_err(|e| e.to_string())?;
    let mut at = 12 + hl;
    let mut env = Env::new();
    for t in header["tensors"].as_array().ok_or("no tensor list")? {
        let (r, c) = (t["rows"].as_u64().ok_or("rows")? as usize, t["cols"].as_u64().ok_or("cols")? as usize);
        let raw = bytes.get(at..at + 4 * r * c).ok_or("short tensor data")?;
        let d = raw.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        at += 4 * r * c;
        env.insert(t["name"].as_str().ok_or("name")?.to_string(), T { r, c, d });
    }
    if at + 32 != bytes.len() {
        return Err("bundle length mismatch".into());
    }
    Ok((header["program"].clone(), env))
}

/// Run a bundle with the given inputs (and initial state tensors, which must
/// also be inputs by name). Returns the declared outputs.
pub fn run(bundle: &[u8], inputs: &[(&str, T)]) -> Result<BTreeMap<String, T>, String> {
    let (prog, mut env) = load(bundle)?;
    for (n, t) in inputs {
        env.insert(n.to_string(), t.clone());
    }
    exec(prog["body"].as_array().ok_or("no body")?, &mut env)?;
    let mut out = BTreeMap::new();
    for o in prog["outputs"].as_array().ok_or("no outputs")? {
        let n = o.as_str().ok_or("output name")?;
        out.insert(n.to_string(), env.get(n).ok_or(format!("missing output {n}"))?.clone());
    }
    for s in prog["state"].as_array().into_iter().flatten() {
        let n = s["name"].as_str().ok_or("state name")?;
        out.insert(format!("state:{n}"), env[n].clone());
    }
    Ok(out)
}

fn get<'a>(env: &'a Env, i: &Value, key: &str) -> Result<&'a T, String> {
    let n = i[key].as_str().ok_or(format!("operand {key}"))?;
    env.get(n).ok_or(format!("unbound {n}"))
}

fn put(env: &mut Env, i: &Value, key: &str, t: T) {
    if let Some(n) = i[key].as_str() {
        env.insert(n.to_string(), t);
    }
}

fn dotp(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}

fn exec(body: &[Value], env: &mut Env) -> Result<(), String> {
    for i in body {
        let op = i["op"].as_str().ok_or("op")?;
        match op {
            "step" => {
                let n = i["n"].as_u64().ok_or("n")?;
                for _ in 0..n {
                    exec(i["body"].as_array().ok_or("step body")?, env)?;
                }
            }
            "matvec" => {
                let (w, x) = (get(env, i, "w")?, get(env, i, "x")?);
                let out = if i["transpose"].as_bool().unwrap_or(false) {
                    let mut o = vec![0.0f32; w.c];
                    for r in 0..w.r {
                        for c in 0..w.c {
                            o[c] += w.at(r, c) * x.d[r];
                        }
                    }
                    o
                } else {
                    (0..w.r).map(|r| dotp(&w.d[r * w.c..(r + 1) * w.c], &x.d)).collect()
                };
                put(env, i, "out", T { r: 1, c: out.len(), d: out });
            }
            "matmul" => {
                let (a, b) = (get(env, i, "a")?, get(env, i, "b")?);
                let mut d = Vec::new();
                for x in 0..a.r {
                    for y in 0..b.r {
                        d.push(dotp(&a.d[x * a.c..(x + 1) * a.c], &b.d[y * b.c..(y + 1) * b.c]));
                    }
                }
                let t = T { r: a.r, c: b.r, d };
                put(env, i, "out", t);
            }
            "gate" => {
                let f = i["f"].as_str().ok_or("f")?;
                let a = get(env, i, "a")?.clone();
                let t = if i.get("b").is_some() {
                    let b = get(env, i, "b")?;
                    let (r, c) = (a.r.max(b.r), a.c.max(b.c));
                    let pick = |t: &T, x: usize, y: usize| t.at(if t.r == 1 { 0 } else { x }, if t.c == 1 { 0 } else { y });
                    let mut d = Vec::new();
                    for x in 0..r {
                        for y in 0..c {
                            let (p, q) = (pick(&a, x, y), pick(b, x, y));
                            d.push(match f {
                                "add" => p + q,
                                "sub" => p - q,
                                "mul" => p * q,
                                "max" => p.max(q),
                                "min" => p.min(q),
                                "ge" => {
                                    if p >= q {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }
                                _ => return Err(format!("binary gate {f}")),
                            });
                        }
                    }
                    T { r, c, d }
                } else {
                    let mut d = Vec::new();
                    for &x in &a.d {
                        d.push(match f {
                            "sigmoid" => 1.0 / (1.0 + (-x).exp()),
                            "tanh" => x.tanh(),
                            "relu" => x.max(0.0),
                            "exp" => x.exp(),
                            "neg" => -x,
                            "copy" => x,
                            _ => return Err(format!("unary gate {f}")),
                        });
                    }
                    T { r: a.r, c: a.c, d }
                };
                put(env, i, "out", t);
            }
            "reduce" => {
                let x = get(env, i, "x")?;
                let f = i["f"].as_str().ok_or("f")?.to_string();
                let red = |v: Vec<f32>| -> f32 {
                    let mut acc = match f.as_str() {
                        "sum" => 0.0,
                        "max" => f32::NEG_INFINITY,
                        _ => f32::INFINITY,
                    };
                    for e in v {
                        acc = match f.as_str() {
                            "sum" => acc + e,
                            "max" => acc.max(e),
                            _ => acc.min(e),
                        };
                    }
                    acc
                };
                let t = match i["axis"].as_str().ok_or("axis")? {
                    "rows" => T { r: x.r, c: 1, d: (0..x.r).map(|r| red((0..x.c).map(|c| x.at(r, c)).collect())).collect() },
                    "cols" => T { r: 1, c: x.c, d: (0..x.c).map(|c| red((0..x.r).map(|r| x.at(r, c)).collect())).collect() },
                    _ => T { r: 1, c: 1, d: vec![red(x.d.clone())] },
                };
                put(env, i, "out", t);
            }
            "softmax" => {
                let x = get(env, i, "x")?;
                let mut d = Vec::new();
                for r in 0..x.r {
                    let row = &x.d[r * x.c..(r + 1) * x.c];
                    let mut m = f32::NEG_INFINITY;
                    for &v in row {
                        m = m.max(v);
                    }
                    let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
                    let mut s = 0.0f32;
                    for v in &e {
                        s += v;
                    }
                    d.extend(e.iter().map(|v| v / s));
                }
                let t = T { r: x.r, c: x.c, d };
                put(env, i, "out", t);
            }
            "norm" => {
                let (x, eps) = (get(env, i, "x")?, get(env, i, "eps")?.d[0]);
                let rms = i["kind"].as_str() == Some("rms");
                let mut d = Vec::new();
                for r in 0..x.r {
                    let row = &x.d[r * x.c..(r + 1) * x.c];
                    let ss = dotp(row, row);
                    let den = if rms { (ss / x.c as f32 + eps).sqrt() } else { (ss + eps).sqrt() };
                    d.extend(row.iter().map(|v| if den == 0.0 { 0.0 } else { v / den }));
                }
                let t = T { r: x.r, c: x.c, d };
                put(env, i, "out", t);
            }
            "bind" => {
                let (a, b) = (get(env, i, "a")?, get(env, i, "b")?);
                let s = if i["conj"].as_bool().unwrap_or(false) { -1.0 } else { 1.0 };
                let h = a.c / 2;
                let mut d = vec![0.0f32; a.d.len()];
                for r in 0..a.r {
                    let br = if b.r == 1 { 0 } else { r };
                    for k in 0..h {
                        let (p, q, u, v) = (a.at(r, k), a.at(r, h + k), b.at(br, k), s * b.at(br, h + k));
                        d[r * a.c + k] = p * u - q * v;
                        d[r * a.c + h + k] = p * v + q * u;
                    }
                }
                let t = T { r: a.r, c: a.c, d };
                put(env, i, "out", t);
            }
            "topk" => {
                let x = get(env, i, "x")?.clone();
                let k = i["k"].as_u64().ok_or("k")? as usize;
                let (mut idx, mut val, mut mask) = (Vec::new(), Vec::new(), vec![0.0f32; x.d.len()]);
                for r in 0..x.r {
                    for _ in 0..k {
                        let mut best = usize::MAX;
                        for c in 0..x.c {
                            if mask[r * x.c + c] == 0.0 && (best == usize::MAX || x.at(r, c) > x.at(r, best)) {
                                best = c;
                            }
                        }
                        mask[r * x.c + best] = 1.0;
                        idx.push(best as f32);
                        val.push(x.at(r, best));
                    }
                }
                put(env, i, "idx", T { r: x.r, c: k, d: idx });
                put(env, i, "val", T { r: x.r, c: k, d: val });
                put(env, i, "mask", T { r: x.r, c: x.c, d: mask });
            }
            "hash_rows" => {
                let (b, tab) = (get(env, i, "bytes")?, get(env, i, "table")?);
                let n = i["n"].as_u64().ok_or("n")? as usize;
                let mut d = vec![0.0f32; b.r * tab.c];
                for r in 0..b.r {
                    let row: Vec<u8> = (0..b.c).map(|c| b.at(r, c) as u8).collect();
                    for w in row.windows(n) {
                        let t = (fnv(w) % tab.r as u64) as usize;
                        for c in 0..tab.c {
                            d[r * tab.c + c] += tab.at(t, c);
                        }
                    }
                }
                let t = T { r: b.r, c: tab.c, d };
                put(env, i, "out", t);
            }
            "delta_write" => {
                let (k, v) = (get(env, i, "k")?.d.clone(), get(env, i, "v")?.d.clone());
                let (lr, decay, eps) = (get(env, i, "lr")?.d[0], get(env, i, "decay")?.d[0], get(env, i, "eps")?.d[0]);
                let name = i["m"].as_str().ok_or("m")?.to_string();
                let m = env.get_mut(&name).ok_or("unbound state")?;
                let scale = lr / (eps + dotp(&k, &k));
                let errs: Vec<f32> = (0..m.r).map(|r| v[r] - dotp(&m.d[r * m.c..(r + 1) * m.c], &k)).collect();
                for r in 0..m.r {
                    let g = scale * errs[r];
                    for c in 0..m.c {
                        m.d[r * m.c + c] = decay * m.d[r * m.c + c] + g * k[c];
                    }
                }
            }
            "reshape" => {
                let x = get(env, i, "x")?.clone();
                let (r, c) = (i["rows"].as_u64().ok_or("rows")? as usize, i["cols"].as_u64().ok_or("cols")? as usize);
                put(env, i, "out", T { r, c, d: x.d });
            }
            "scan" => {
                let x = get(env, i, "x")?;
                let ex = i["exclusive"].as_bool().unwrap_or(false);
                let mut d = Vec::new();
                for r in 0..x.r {
                    let mut p = 1.0f32;
                    for c in 0..x.c {
                        if ex {
                            d.push(p);
                            p *= x.at(r, c);
                        } else {
                            p *= x.at(r, c);
                            d.push(p);
                        }
                    }
                }
                let t = T { r: x.r, c: x.c, d };
                put(env, i, "out", t);
            }
            other => return Err(format!("unknown op {other}")),
        }
    }
    Ok(())
}
