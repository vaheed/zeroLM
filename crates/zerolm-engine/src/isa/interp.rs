//! The main ISA interpreter. Kernels are plain loops whose summation order is
//! the one specified in `docs/NEURAL_ISA.md`, so the reference interpreter can
//! reproduce every result bit for bit.

use super::{err, Axis, Buf, GateFn, Instr, IsaError, NormKind, Program, ReduceFn, View};
use std::collections::BTreeMap;
use zerolm_tensor::fnv1a;

/// Tensors bound to one run of a program.
#[derive(Debug, Default)]
pub struct Bindings<'a> {
    pub inputs: Vec<(String, Buf)>,
    pub weights: Vec<(String, View<'a>)>,
    pub state: Vec<(String, Buf)>,
}

impl<'a> Bindings<'a> {
    pub fn new() -> Self {
        Bindings::default()
    }

    pub fn input(mut self, name: &str, b: Buf) -> Self {
        self.inputs.push((name.into(), b));
        self
    }

    pub fn weight(mut self, name: &str, v: View<'a>) -> Self {
        self.weights.push((name.into(), v));
        self
    }

    pub fn state(mut self, name: &str, b: Buf) -> Self {
        self.state.push((name.into(), b));
        self
    }
}

/// Program outputs and the final state tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct Outputs {
    pub values: BTreeMap<String, Buf>,
    pub state: BTreeMap<String, Buf>,
}

impl Outputs {
    pub fn get(&self, name: &str) -> &Buf {
        self.values.get(name).unwrap_or_else(|| panic!("program has no output '{name}'"))
    }

    pub fn scalar(&self, name: &str) -> f32 {
        self.get(name).data[0]
    }
}

struct Env<'a> {
    weights: BTreeMap<String, View<'a>>,
    regs: BTreeMap<String, Buf>,
    state: BTreeMap<String, Buf>,
}

impl<'a> Env<'a> {
    fn get(&self, name: &str) -> Result<View<'_>, IsaError> {
        if let Some(b) = self.regs.get(name).or_else(|| self.state.get(name)) {
            return Ok(b.view());
        }
        match self.weights.get(name) {
            Some(v) => Ok(*v),
            None => err(format!("unbound tensor '{name}'")),
        }
    }

    fn set(&mut self, name: &str, b: Buf) {
        self.regs.insert(name.to_string(), b);
    }
}

/// Validate, bind, and run a program.
pub fn run(p: &Program, b: Bindings<'_>) -> Result<Outputs, IsaError> {
    p.validate()?;
    let mut env = Env { weights: BTreeMap::new(), regs: BTreeMap::new(), state: BTreeMap::new() };
    let check = |kind: &str, decls: &[super::Decl], got: &[(String, usize, usize)]| -> Result<(), IsaError> {
        if decls.len() != got.len() {
            return err(format!("{}: expected {} {kind} tensors, got {}", p.name, decls.len(), got.len()));
        }
        for d in decls {
            let Some((_, r, c)) = got.iter().find(|g| g.0 == d.name) else { return err(format!("{}: {kind} '{}' not bound", p.name, d.name)) };
            if d.cols.map(|dc| dc != *c).unwrap_or(false) || d.rows.map(|dr| dr != *r).unwrap_or(false) {
                return err(format!("{}: {kind} '{}' is {r}x{c}, declared {:?}x{:?}", p.name, d.name, d.rows, d.cols));
            }
        }
        Ok(())
    };
    check("input", &p.inputs, &b.inputs.iter().map(|(n, t)| (n.clone(), t.rows, t.cols)).collect::<Vec<_>>())?;
    check("weight", &p.weights, &b.weights.iter().map(|(n, t)| (n.clone(), t.rows, t.cols)).collect::<Vec<_>>())?;
    check("state", &p.state, &b.state.iter().map(|(n, t)| (n.clone(), t.rows, t.cols)).collect::<Vec<_>>())?;
    for (n, t) in b.inputs {
        env.regs.insert(n, t);
    }
    env.weights.extend(b.weights);
    env.state.extend(b.state);
    exec(&p.body, &mut env)?;
    let mut values = BTreeMap::new();
    for o in &p.outputs {
        values.insert(o.clone(), env.regs.get(o).cloned().expect("validated output"));
    }
    Ok(Outputs { values, state: env.state })
}

fn exec(body: &[Instr], env: &mut Env<'_>) -> Result<(), IsaError> {
    for ins in body {
        match ins {
            Instr::Step { n, body } => {
                for _ in 0..*n {
                    exec(body, env)?;
                }
            }
            Instr::DeltaWrite { m, k, v, lr, decay, eps } => {
                let (k, v) = (env.get(k)?.data.to_vec(), env.get(v)?.data.to_vec());
                let (lr, decay, eps) = (scalar(env.get(lr)?)?, scalar(env.get(decay)?)?, scalar(env.get(eps)?)?);
                let mt = env.state.get_mut(m).expect("validated state");
                if k.len() != mt.cols || v.len() != mt.rows {
                    return err(format!("delta_write: key {} / value {} do not fit {}x{}", k.len(), v.len(), mt.rows, mt.cols));
                }
                delta_write(mt, &k, &v, lr, decay, eps);
            }
            other => {
                for (name, out) in eval(other, env)? {
                    env.set(name, out);
                }
            }
        }
    }
    Ok(())
}

fn scalar(v: View<'_>) -> Result<f32, IsaError> {
    if v.data.len() != 1 {
        return err(format!("expected a 1x1 tensor, got {}x{}", v.rows, v.cols));
    }
    Ok(v.data[0])
}

fn eval<'i>(ins: &'i Instr, env: &Env<'_>) -> Result<Vec<(&'i str, Buf)>, IsaError> {
    Ok(match ins {
        Instr::Matvec { w, x, out, transpose } => vec![(out.as_str(), matvec(env.get(w)?, env.get(x)?, *transpose)?)],
        Instr::Matmul { a, b, out, ab } => {
            let r = if *ab { matmul_ab(env.get(a)?, env.get(b)?)? } else { matmul(env.get(a)?, env.get(b)?)? };
            vec![(out.as_str(), r)]
        }
        Instr::Gate { f, a, b, out } => {
            let a = env.get(a)?;
            let r = match (f.is_binary(), b) {
                (true, Some(b)) => binary(*f, a, env.get(b)?)?,
                (false, None) => unary(*f, a),
                _ => return err(format!("gate {f:?}: wrong operand count")),
            };
            vec![(out.as_str(), r)]
        }
        Instr::Reduce { f, x, axis, out } => vec![(out.as_str(), reduce(*f, env.get(x)?, *axis))],
        Instr::Softmax { x, out } => vec![(out.as_str(), softmax(env.get(x)?))],
        Instr::Norm { kind, x, eps, out } => vec![(out.as_str(), norm(*kind, env.get(x)?, scalar(env.get(eps)?)?))],
        Instr::Bind { a, b, out, conj } => vec![(out.as_str(), bind(env.get(a)?, env.get(b)?, *conj)?)],
        Instr::Topk { x, k, idx, val, mask } => {
            let (i, v, m) = topk(env.get(x)?, *k)?;
            let mut r = Vec::new();
            for (name, b) in [(idx, i), (val, v), (mask, m)] {
                if let Some(n) = name {
                    r.push((n.as_str(), b));
                }
            }
            r
        }
        Instr::HashRows { bytes, table, n, out } => vec![(out.as_str(), hash_rows(env.get(bytes)?, env.get(table)?, *n)?)],
        Instr::Reshape { x, rows, cols, out } => {
            let x = env.get(x)?;
            if rows * cols != x.data.len() {
                return err(format!("reshape {}x{} to {rows}x{cols}", x.rows, x.cols));
            }
            vec![(out.as_str(), Buf::new(*rows, *cols, x.data.to_vec()))]
        }
        Instr::Transpose { x, out } => {
            let x = env.get(x)?;
            let mut d = Vec::with_capacity(x.data.len());
            for c in 0..x.cols {
                for r in 0..x.rows {
                    d.push(x.at(r, c));
                }
            }
            vec![(out.as_str(), Buf::new(x.cols, x.rows, d))]
        }
        Instr::Scan { x, out, exclusive } => vec![(out.as_str(), scan(env.get(x)?, *exclusive))],
        Instr::Step { .. } | Instr::DeltaWrite { .. } => unreachable!("handled by exec"),
    })
}

fn matvec(w: View<'_>, x: View<'_>, transpose: bool) -> Result<Buf, IsaError> {
    let want = if transpose { w.rows } else { w.cols };
    if x.rows != 1 || x.cols != want {
        return err(format!("matvec: x is {}x{}, needs 1x{want} for a {}x{} matrix", x.rows, x.cols, w.rows, w.cols));
    }
    if !transpose {
        let out = (0..w.rows)
            .map(|r| {
                let row = &w.data[r * w.cols..(r + 1) * w.cols];
                let mut s = 0.0f32;
                for (a, b) in row.iter().zip(x.data) {
                    s += a * b;
                }
                s
            })
            .collect();
        return Ok(Buf::row(out));
    }
    // out[c] = sum over r (in order) of w[r][c] * x[r]. A row with x[r] == 0
    // adds +-0 to an accumulator that starts at +0 and so can never be -0;
    // skipping it is exact for finite weights (a bound-weight precondition,
    // docs/NEURAL_ISA.md). Attention reads and one-hot gathers are mostly zero.
    let mut out = vec![0.0f32; w.cols];
    for (r, &xr) in x.data.iter().enumerate() {
        if xr == 0.0 {
            continue;
        }
        let row = &w.data[r * w.cols..(r + 1) * w.cols];
        for (o, a) in out.iter_mut().zip(row) {
            *o += a * xr;
        }
    }
    Ok(Buf::row(out))
}

fn matmul(a: View<'_>, b: View<'_>) -> Result<Buf, IsaError> {
    if a.cols != b.cols {
        return err(format!("matmul: {}x{} by ({}x{})^T", a.rows, a.cols, b.rows, b.cols));
    }
    // out[i][j] = sum over k (in order) of a[i][k] * b[j][k]. Looping i, k, j
    // keeps that per-element order, so skipping a[i][k] == 0 is exact for
    // finite b (the same argument as the transposed matvec); provenance and
    // gather matrices are mostly zero.
    let (k, n) = (a.cols, b.rows);
    // b^T laid out contiguously (data movement only; arithmetic unchanged).
    let mut bt = vec![0.0f32; k * n];
    for j in 0..n {
        for kk in 0..k {
            bt[kk * n + j] = b.data[j * k + kk];
        }
    }
    let mut out = vec![0.0f32; a.rows * n];
    for i in 0..a.rows {
        let o = &mut out[i * n..(i + 1) * n];
        for (kk, &x) in a.data[i * k..(i + 1) * k].iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            for (oj, bv) in o.iter_mut().zip(&bt[kk * n..(kk + 1) * n]) {
                *oj += x * bv;
            }
        }
    }
    Ok(Buf::new(a.rows, b.rows, out))
}

/// out = A B: out[i][j] = sum over k (in order) of a[i][k] * b[k][j]. Rows of
/// b are contiguous, and zero a[i][k] are skipped (exact for finite b).
fn matmul_ab(a: View<'_>, b: View<'_>) -> Result<Buf, IsaError> {
    if a.cols != b.rows {
        return err(format!("matmul: {}x{} by {}x{}", a.rows, a.cols, b.rows, b.cols));
    }
    let n = b.cols;
    let mut out = vec![0.0f32; a.rows * n];
    for i in 0..a.rows {
        let o = &mut out[i * n..(i + 1) * n];
        for (kk, &x) in a.data[i * a.cols..(i + 1) * a.cols].iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            for (oj, bv) in o.iter_mut().zip(&b.data[kk * n..(kk + 1) * n]) {
                *oj += x * bv;
            }
        }
    }
    Ok(Buf::new(a.rows, n, out))
}

fn bdim(a: usize, b: usize, what: &str) -> Result<usize, IsaError> {
    match (a, b) {
        _ if a == b => Ok(a),
        (1, n) | (n, 1) => Ok(n),
        _ => err(format!("gate: cannot broadcast {what} {a} with {b}")),
    }
}

fn binary(f: GateFn, a: View<'_>, b: View<'_>) -> Result<Buf, IsaError> {
    let rows = bdim(a.rows, b.rows, "rows")?;
    let cols = bdim(a.cols, b.cols, "cols")?;
    let op = |x: f32, y: f32| match f {
        GateFn::Add => x + y,
        GateFn::Sub => x - y,
        GateFn::Mul => x * y,
        GateFn::Max => x.max(y),
        GateFn::Min => x.min(y),
        GateFn::Ge => (x >= y) as u8 as f32,
        _ => unreachable!("binary gate"),
    };
    // Fast paths (same per-element arithmetic, fewer index computations).
    if a.rows == rows && a.cols == cols {
        if b.rows == rows && b.cols == cols {
            return Ok(Buf::new(rows, cols, a.data.iter().zip(b.data).map(|(&x, &y)| op(x, y)).collect()));
        }
        if b.data.len() == 1 {
            let y = b.data[0];
            return Ok(Buf::new(rows, cols, a.data.iter().map(|&x| op(x, y)).collect()));
        }
        if b.rows == 1 {
            let mut out = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                out.extend(a.data[r * cols..(r + 1) * cols].iter().zip(b.data).map(|(&x, &y)| op(x, y)));
            }
            return Ok(Buf::new(rows, cols, out));
        }
        if b.cols == 1 {
            let mut out = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                let y = b.data[r];
                out.extend(a.data[r * cols..(r + 1) * cols].iter().map(|&x| op(x, y)));
            }
            return Ok(Buf::new(rows, cols, out));
        }
    }
    let at = |v: &View<'_>, r: usize, c: usize| v.at(if v.rows == 1 { 0 } else { r }, if v.cols == 1 { 0 } else { c });
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let (x, y) = (at(&a, r, c), at(&b, r, c));
            out.push(match f {
                GateFn::Add => x + y,
                GateFn::Sub => x - y,
                GateFn::Mul => x * y,
                GateFn::Max => x.max(y),
                GateFn::Min => x.min(y),
                GateFn::Ge => (x >= y) as u8 as f32,
                _ => unreachable!("binary gate"),
            });
        }
    }
    Ok(Buf::new(rows, cols, out))
}

fn unary(f: GateFn, a: View<'_>) -> Buf {
    let g = |x: f32| match f {
        GateFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        GateFn::Tanh => x.tanh(),
        GateFn::Relu => x.max(0.0),
        GateFn::Exp => x.exp(),
        GateFn::Neg => -x,
        GateFn::Copy => x,
        _ => unreachable!("unary gate"),
    };
    Buf::new(a.rows, a.cols, a.data.iter().map(|&x| g(x)).collect())
}

fn fold(f: ReduceFn, xs: impl Iterator<Item = f32>) -> f32 {
    match f {
        ReduceFn::Sum => xs.fold(0.0, |s, x| s + x),
        ReduceFn::Max => xs.fold(f32::NEG_INFINITY, f32::max),
        ReduceFn::Min => xs.fold(f32::INFINITY, f32::min),
    }
}

fn reduce(f: ReduceFn, x: View<'_>, axis: Axis) -> Buf {
    match axis {
        Axis::Rows => Buf::new(x.rows, 1, (0..x.rows).map(|r| fold(f, (0..x.cols).map(|c| x.at(r, c)))).collect()),
        Axis::Cols => Buf::new(1, x.cols, (0..x.cols).map(|c| fold(f, (0..x.rows).map(|r| x.at(r, c)))).collect()),
        Axis::All => Buf::scalar(fold(f, x.data.iter().copied())),
    }
}

fn softmax(x: View<'_>) -> Buf {
    let mut out = Vec::with_capacity(x.data.len());
    for r in 0..x.rows {
        let row = &x.data[r * x.cols..(r + 1) * x.cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let s = e.iter().fold(0.0f32, |s, v| s + v);
        out.extend(e.iter().map(|v| v / s));
    }
    Buf::new(x.rows, x.cols, out)
}

fn norm(kind: NormKind, x: View<'_>, eps: f32) -> Buf {
    let mut out = Vec::with_capacity(x.data.len());
    for r in 0..x.rows {
        let row = &x.data[r * x.cols..(r + 1) * x.cols];
        let ss = row.iter().fold(0.0f32, |s, v| s + v * v);
        let den = match kind {
            NormKind::L2 => (ss + eps).sqrt(),
            NormKind::Rms => (ss / x.cols as f32 + eps).sqrt(),
        };
        out.extend(row.iter().map(|v| if den == 0.0 { 0.0 } else { v / den }));
    }
    Buf::new(x.rows, x.cols, out)
}

fn bind(a: View<'_>, b: View<'_>, conj: bool) -> Result<Buf, IsaError> {
    if a.cols != b.cols || a.cols % 2 != 0 || !(b.rows == a.rows || b.rows == 1) {
        return err(format!("bind: {}x{} with {}x{}", a.rows, a.cols, b.rows, b.cols));
    }
    let d = a.cols / 2;
    let sign = if conj { -1.0 } else { 1.0 };
    let mut out = vec![0.0f32; a.data.len()];
    for r in 0..a.rows {
        let br = if b.rows == 1 { 0 } else { r };
        for k in 0..d {
            let (ar, ai) = (a.at(r, k), a.at(r, d + k));
            let (xr, xi) = (b.at(br, k), sign * b.at(br, d + k));
            out[r * a.cols + k] = ar * xr - ai * xi;
            out[r * a.cols + d + k] = ar * xi + ai * xr;
        }
    }
    Ok(Buf::new(a.rows, a.cols, out))
}

fn topk(x: View<'_>, k: usize) -> Result<(Buf, Buf, Buf), IsaError> {
    if k == 0 || k > x.cols {
        return err(format!("topk: k={k} for {} columns", x.cols));
    }
    let (mut idx, mut val, mut mask) = (Vec::new(), Vec::new(), vec![0.0f32; x.data.len()]);
    let mut taken = vec![false; x.cols];
    for r in 0..x.rows {
        let row = &x.data[r * x.cols..(r + 1) * x.cols];
        taken.iter_mut().for_each(|t| *t = false);
        for _ in 0..k {
            // First untaken column with the strictly largest value.
            let mut best: Option<(usize, f32)> = None;
            for (c, (&t, &v)) in taken.iter().zip(row).enumerate() {
                if !t && best.map(|b| v > b.1).unwrap_or(true) {
                    best = Some((c, v));
                }
            }
            let (c, v) = best.expect("k <= cols");
            taken[c] = true;
            idx.push(c as f32);
            val.push(v);
            mask[r * x.cols + c] = 1.0;
        }
    }
    Ok((Buf::new(x.rows, k, idx), Buf::new(x.rows, k, val), Buf::new(x.rows, x.cols, mask)))
}

fn hash_rows(bytes: View<'_>, table: View<'_>, n: usize) -> Result<Buf, IsaError> {
    if n == 0 || table.rows == 0 {
        return err("hash_rows: empty window or table");
    }
    let mut out = vec![0.0f32; bytes.rows * table.cols];
    for r in 0..bytes.rows {
        let mut row = Vec::with_capacity(bytes.cols);
        for c in 0..bytes.cols {
            let v = bytes.at(r, c);
            if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
                return err(format!("hash_rows: {v} is not a byte value"));
            }
            row.push(v as u8);
        }
        let o = &mut out[r * table.cols..(r + 1) * table.cols];
        for w in row.windows(n) {
            let t = (fnv1a(w) % table.rows as u64) as usize;
            for (a, b) in o.iter_mut().zip(&table.data[t * table.cols..(t + 1) * table.cols]) {
                *a += b;
            }
        }
    }
    Ok(Buf::new(bytes.rows, table.cols, out))
}

fn scan(x: View<'_>, exclusive: bool) -> Buf {
    let mut out = Vec::with_capacity(x.data.len());
    for r in 0..x.rows {
        let mut p = 1.0f32;
        for c in 0..x.cols {
            let v = x.at(r, c);
            if exclusive {
                out.push(p);
                p *= v;
            } else {
                p *= v;
                out.push(p);
            }
        }
    }
    Buf::new(x.rows, x.cols, out)
}

fn delta_write(m: &mut Buf, k: &[f32], v: &[f32], lr: f32, decay: f32, eps: f32) {
    let kk = k.iter().fold(0.0f32, |s, x| s + x * x);
    let scale = lr / (eps + kk);
    let cols = m.cols;
    let err: Vec<f32> = (0..m.rows)
        .map(|r| {
            let row = &m.data[r * cols..(r + 1) * cols];
            let mut s = 0.0f32;
            for (a, b) in row.iter().zip(k) {
                s += a * b;
            }
            v[r] - s
        })
        .collect();
    for (r, e) in err.iter().enumerate() {
        let g = scale * e;
        for (a, kc) in m.data[r * cols..(r + 1) * cols].iter_mut().zip(k) {
            *a = decay * *a + g * kc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::{Decl, Instr};

    fn s(x: &str) -> String {
        x.to_string()
    }

    #[test]
    fn validation_rejects_undefined_reads_and_writes_to_weights() {
        let mut p = Program {
            name: s("t"),
            inputs: vec![Decl::new("x", Some(1), 2)],
            weights: vec![Decl::new("w", Some(2), 2)],
            state: vec![],
            body: vec![Instr::Matvec { w: s("w"), x: s("y"), out: s("o"), transpose: false }],
            outputs: vec![s("o")],
        };
        assert!(p.validate().is_err());
        p.body = vec![Instr::Matvec { w: s("w"), x: s("x"), out: s("w"), transpose: false }];
        assert!(p.validate().is_err());
        p.body = vec![Instr::Matvec { w: s("w"), x: s("x"), out: s("o"), transpose: false }];
        p.validate().unwrap();
        let w = [1.0, 2.0, 3.0, 4.0];
        let out = run(&p, Bindings::new().input("x", Buf::row(vec![1.0, 1.0])).weight("w", View::new(2, 2, &w))).unwrap();
        assert_eq!(out.get("o").data, vec![3.0, 7.0]);
        assert!(run(&p, Bindings::new().input("x", Buf::row(vec![1.0, 1.0, 1.0])).weight("w", View::new(2, 2, &w))).is_err());
    }

    #[test]
    fn kernels_follow_the_specified_semantics() {
        let x = Buf::new(2, 3, vec![1.0, 3.0, 3.0, -1.0, 0.0, 2.0]);
        let (i, v, m) = topk(x.view(), 2).unwrap();
        assert_eq!(i.data, vec![1.0, 2.0, 2.0, 1.0]);
        assert_eq!(v.data, vec![3.0, 3.0, 2.0, 0.0]);
        assert_eq!(m.data, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(scan(Buf::row(vec![1.0, 1.0, 0.0, 1.0]).view(), false).data, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(scan(Buf::row(vec![1.0, 1.0, 0.0, 1.0]).view(), true).data, vec![1.0, 1.0, 1.0, 0.0]);
        let col = Buf::new(2, 1, vec![10.0, 20.0]);
        assert_eq!(binary(GateFn::Add, x.view(), col.view()).unwrap().data, vec![11.0, 13.0, 13.0, 19.0, 20.0, 22.0]);
        assert_eq!(binary(GateFn::Ge, x.view(), Buf::scalar(2.0).view()).unwrap().data, vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(reduce(ReduceFn::Max, x.view(), Axis::Rows).data, vec![3.0, 2.0]);
        assert_eq!(reduce(ReduceFn::Min, x.view(), Axis::Cols).data, vec![-1.0, 0.0, 2.0]);
        let sm = softmax(Buf::row(vec![0.0, 0.0]).view());
        assert_eq!(sm.data, vec![0.5, 0.5]);
        let mt = matvec(x.view(), Buf::row(vec![1.0, 2.0]).view(), true).unwrap();
        assert_eq!(mt.data, vec![-1.0, 3.0, 7.0]);
        let mm = matmul(x.view(), x.view()).unwrap();
        assert_eq!(mm.data, vec![19.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn delta_write_stores_an_association_and_bind_inverts() {
        let mut m = Buf::zeros(2, 3);
        delta_write(&mut m, &[1.0, 0.0, 1.0], &[2.0, -4.0], 1.0, 1.0, 0.0);
        let r = matvec(m.view(), Buf::row(vec![1.0, 0.0, 1.0]).view(), false).unwrap();
        assert_eq!(r.data, vec![2.0, -4.0]);
        let a = Buf::row(vec![0.6, 0.0, 0.8, 1.0]);
        let b = Buf::row(vec![0.0, 1.0, 1.0, 0.0]);
        let ab = bind(a.view(), b.view(), false).unwrap();
        let back = bind(ab.view(), b.view(), true).unwrap();
        assert_eq!(back.data, a.data);
    }

    #[test]
    fn hash_rows_matches_fnv_trigrams_and_rejects_non_bytes() {
        let table = Buf::new(4, 1, vec![1.0, 10.0, 100.0, 1000.0]);
        let bytes = Buf::row(b"#ab#".iter().map(|&b| b as f32).collect());
        let got = hash_rows(bytes.view(), table.view(), 3).unwrap();
        let want: f32 = [&b"#ab"[..], b"ab#"].iter().map(|w| table.data[(fnv1a(w) % 4) as usize]).sum();
        assert_eq!(got.data, vec![want]);
        assert!(hash_rows(Buf::row(vec![0.5]).view(), table.view(), 1).is_err());
    }

    #[test]
    fn step_repeats_with_persistent_registers() {
        let p = Program {
            name: s("count"),
            inputs: vec![],
            weights: vec![Decl::new("one", Some(1), 1)],
            state: vec![],
            body: vec![
                Instr::Gate { f: GateFn::Sub, a: s("one"), b: Some(s("one")), out: s("acc") },
                Instr::Step { n: 5, body: vec![Instr::Gate { f: GateFn::Add, a: s("acc"), b: Some(s("one")), out: s("acc") }] },
            ],
            outputs: vec![s("acc")],
        };
        let one = [1.0];
        let out = run(&p, Bindings::new().weight("one", View::new(1, 1, &one))).unwrap();
        assert_eq!(out.scalar("acc"), 5.0);
    }
}
