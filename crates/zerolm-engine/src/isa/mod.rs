//! The neural ISA (roadmap section 3.7): a fixed-shape program graph over a
//! small set of tensor operations. Every content-dependent decision of a
//! compiled capability is one of these operations on tensors; host code only
//! binds tensors, runs the program, and moves bytes in and out.
//!
//! Programs carry no floating-point immediates. Every constant (a threshold,
//! a temperature, a learning rate) is a weight tensor, so behavior is a
//! function of weights and inputs only, and a program serialized as JSON is
//! pure structure.
//!
//! Exact semantics (summation order, tie breaking, broadcasting) are specified
//! in `docs/neural-isa.md`. An independent reference interpreter in
//! `zerolm-eval` must reproduce every output bit for bit.

pub mod bundle;
mod interp;

pub use interp::{run, Bindings, Outputs};

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A dense row-major f32 tensor (rows x cols); vectors are 1 x n.
#[derive(Debug, Clone, PartialEq)]
pub struct Buf {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Buf {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "buffer shape mismatch");
        Buf { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Buf { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn row(data: Vec<f32>) -> Self {
        Buf { rows: 1, cols: data.len(), data }
    }

    pub fn scalar(x: f32) -> Self {
        Buf { rows: 1, cols: 1, data: vec![x] }
    }

    pub fn view(&self) -> View<'_> {
        View { rows: self.rows, cols: self.cols, data: &self.data }
    }
}

/// A borrowed tensor (weights are bound by reference, never copied).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View<'a> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [f32],
}

impl<'a> View<'a> {
    pub fn new(rows: usize, cols: usize, data: &'a [f32]) -> Self {
        assert_eq!(rows * cols, data.len(), "view shape mismatch");
        View { rows, cols, data }
    }

    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

/// A declared program tensor. `rows: None` means the caller fixes the row
/// count at run time (for example the number of words in a query); every
/// other shape inside the program follows from the declarations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decl {
    pub name: String,
    pub rows: Option<usize>,
    /// `None`: the caller fixes the column count at run time.
    pub cols: Option<usize>,
}

impl Decl {
    pub fn new(name: &str, rows: Option<usize>, cols: usize) -> Self {
        Decl { name: name.into(), rows, cols: Some(cols) }
    }

    /// A tensor whose rows and/or columns are fixed by the caller.
    pub fn dynamic(name: &str, rows: Option<usize>, cols: Option<usize>) -> Self {
        Decl { name: name.into(), rows, cols }
    }
}

/// Elementwise functions. Binary ones broadcast (see `docs/neural-isa.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateFn {
    Add,
    Sub,
    Mul,
    Max,
    Min,
    /// 1.0 where a >= b, else 0.0 (the hard limit of a sigmoid gate).
    Ge,
    Sigmoid,
    Tanh,
    Relu,
    Exp,
    Neg,
    Copy,
}

impl GateFn {
    pub fn is_binary(self) -> bool {
        matches!(self, GateFn::Add | GateFn::Sub | GateFn::Mul | GateFn::Max | GateFn::Min | GateFn::Ge)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduceFn {
    Sum,
    Max,
    Min,
}

/// `rows`: one value per row (R x 1); `cols`: one per column (1 x C); `all`: 1 x 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    Rows,
    Cols,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    L2,
    Rms,
}

/// One instruction. Operands name an input, weight, state tensor, or a
/// register written by an earlier instruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Instr {
    /// out = W x (1 x R), or with `transpose` out = W^T x (1 x C).
    Matvec {
        w: String,
        x: String,
        out: String,
        #[serde(default)]
        transpose: bool,
    },
    /// out = A B^T: (M x K) and (N x K) give M x N; with `ab`, out = A B:
    /// (M x K) and (K x N) give M x N.
    Matmul {
        a: String,
        b: String,
        out: String,
        #[serde(default, skip_serializing_if = "is_false")]
        ab: bool,
    },
    Gate {
        f: GateFn,
        a: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        b: Option<String>,
        out: String,
    },
    Reduce { f: ReduceFn, x: String, axis: Axis, out: String },
    /// Row-wise softmax.
    Softmax { x: String, out: String },
    /// Row-wise L2 (x / sqrt(sum x^2 + eps)) or RMS normalization; `eps` is a 1 x 1 tensor.
    Norm { kind: NormKind, x: String, eps: String, out: String },
    /// Complex phasor binding of rows laid out `[re; im]` (elementwise product,
    /// with the conjugate of `b` when `conj`).
    Bind {
        a: String,
        b: String,
        out: String,
        #[serde(default)]
        conj: bool,
    },
    /// Row-wise hard top-k (ties go to the lower index). Any of the outputs
    /// may be omitted: `idx` and `val` are R x k, `mask` is R x C with 1.0 at
    /// the selected positions.
    Topk {
        x: String,
        k: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        idx: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        val: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mask: Option<String>,
    },
    /// For each row of byte values (exact integers 0..=255), the sum of the
    /// `table` rows selected by FNV-1a of every window of `n` bytes.
    HashRows { bytes: String, table: String, n: usize, out: String },
    /// Normalized delta rule into the state tensor `m` (R x C) with key `k`
    /// (1 x C) and value `v` (1 x R); `lr`, `decay`, `eps` are 1 x 1 tensors.
    DeltaWrite { m: String, k: String, v: String, lr: String, decay: String, eps: String },
    Reshape { x: String, rows: usize, cols: usize, out: String },
    /// out[c][r] = x[r][c].
    Transpose { x: String, out: String },
    /// Row-wise prefix product (`exclusive`: out[0] = 1, out[j] = prod x[..j]).
    Scan {
        x: String,
        out: String,
        #[serde(default)]
        exclusive: bool,
    },
    /// Run `body` exactly `n` times; registers persist across iterations.
    Step { n: usize, body: Vec<Instr> },
}

impl Instr {
    /// Operation family, as listed in the roadmap ISA table.
    pub fn family(&self) -> &'static str {
        match self {
            Instr::Matvec { .. } | Instr::Matmul { .. } => "matvec",
            Instr::Gate { .. } | Instr::Reduce { .. } | Instr::Softmax { .. } | Instr::Scan { .. } | Instr::Reshape { .. } | Instr::Transpose { .. } => "gate",
            Instr::Norm { .. } => "norm",
            Instr::Bind { .. } => "bind",
            Instr::Topk { .. } => "topk",
            Instr::HashRows { .. } => "hash_rows",
            Instr::DeltaWrite { .. } => "delta_write",
            Instr::Step { .. } => "step",
        }
    }

    fn reads(&self) -> Vec<&str> {
        match self {
            Instr::Matvec { w, x, .. } => vec![w, x],
            Instr::Matmul { a, b, .. } => vec![a, b],
            Instr::Gate { a, b, .. } => std::iter::once(a.as_str()).chain(b.as_deref()).collect(),
            Instr::Reduce { x, .. } | Instr::Softmax { x, .. } | Instr::Reshape { x, .. } | Instr::Transpose { x, .. } | Instr::Scan { x, .. } | Instr::Topk { x, .. } => vec![x],
            Instr::Norm { x, eps, .. } => vec![x, eps],
            Instr::Bind { a, b, .. } => vec![a, b],
            Instr::HashRows { bytes, table, .. } => vec![bytes, table],
            Instr::DeltaWrite { m, k, v, lr, decay, eps } => vec![m, k, v, lr, decay, eps],
            Instr::Step { .. } => vec![],
        }
    }

    fn writes(&self) -> Vec<&str> {
        match self {
            Instr::Matvec { out, .. }
            | Instr::Matmul { out, .. }
            | Instr::Gate { out, .. }
            | Instr::Reduce { out, .. }
            | Instr::Softmax { out, .. }
            | Instr::Norm { out, .. }
            | Instr::Bind { out, .. }
            | Instr::HashRows { out, .. }
            | Instr::Reshape { out, .. }
            | Instr::Transpose { out, .. }
            | Instr::Scan { out, .. } => vec![out],
            Instr::Topk { idx, val, mask, .. } => [idx, val, mask].into_iter().flatten().map(String::as_str).collect(),
            Instr::DeltaWrite { .. } | Instr::Step { .. } => vec![],
        }
    }
}

/// A program: declarations, a body, and the registers it returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub name: String,
    /// Caller-supplied per-run tensors.
    pub inputs: Vec<Decl>,
    /// Read-only tensors (compiled `W_PROG` constants and bound model tensors).
    pub weights: Vec<Decl>,
    /// Mutable tensors written only by `delta_write`, returned after the run.
    #[serde(default)]
    pub state: Vec<Decl>,
    pub body: Vec<Instr>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsaError(pub String);

impl std::fmt::Display for IsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "isa: {}", self.0)
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T, IsaError> {
    Err(IsaError(msg.into()))
}

impl Program {
    /// Static check: unique declarations, every operand defined before use,
    /// no instruction writes a declared tensor, delta_write targets state,
    /// and every output is a register.
    pub fn validate(&self) -> Result<(), IsaError> {
        let mut declared = BTreeSet::new();
        for d in self.inputs.iter().chain(&self.weights).chain(&self.state) {
            if !declared.insert(d.name.as_str()) {
                return err(format!("{}: duplicate declaration {}", self.name, d.name));
            }
        }
        let state: BTreeSet<&str> = self.state.iter().map(|d| d.name.as_str()).collect();
        let mut regs = BTreeSet::new();
        fn walk<'a>(
            body: &'a [Instr],
            declared: &BTreeSet<&'a str>,
            state: &BTreeSet<&'a str>,
            regs: &mut BTreeSet<&'a str>,
            name: &str,
        ) -> Result<(), IsaError> {
            for ins in body {
                if let Instr::Step { n, body } = ins {
                    if *n == 0 {
                        return err(format!("{name}: step(0)"));
                    }
                    walk(body, declared, state, regs, name)?;
                    continue;
                }
                for r in ins.reads() {
                    if !declared.contains(r) && !regs.contains(r) {
                        return err(format!("{name}: '{r}' is read before it is defined"));
                    }
                }
                if let Instr::DeltaWrite { m, .. } = ins {
                    if !state.contains(m.as_str()) {
                        return err(format!("{name}: delta_write target '{m}' is not a state tensor"));
                    }
                }
                for w in ins.writes() {
                    if declared.contains(w) {
                        return err(format!("{name}: instruction writes declared tensor '{w}'"));
                    }
                    regs.insert(w);
                }
            }
            Ok(())
        }
        walk(&self.body, &declared, &state, &mut regs, &self.name)?;
        for o in &self.outputs {
            if !regs.contains(o.as_str()) {
                return err(format!("{}: output '{o}' is never written", self.name));
            }
        }
        Ok(())
    }

    /// Number of instructions, counting `step` bodies once.
    pub fn len(&self) -> usize {
        fn count(b: &[Instr]) -> usize {
            b.iter().map(|i| if let Instr::Step { body, .. } = i { 1 + count(body) } else { 1 }).sum()
        }
        count(&self.body)
    }

    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// Canonical JSON (field order fixed by the type definitions).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("program serializes")
    }
}
