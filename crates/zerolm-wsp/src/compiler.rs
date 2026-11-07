//! The circuit compiler: a typed circuit specification becomes a neural ISA
//! program plus its `W_PROG` constant tensors.
//!
//! A circuit encodes an algorithm (selection, attention, verification,
//! cleanup), never language data: no vocabulary, word list, fact, or corpus
//! statistic may appear in a spec or in its compiled tensors (constitution
//! invariant 11). Compilation is deterministic; the SHA-256 of the output is
//! pinned in `model/prog/pinned.json`.

use crate::circuits;
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, Decl, GateFn, Instr, IsaError, Program, ReduceFn, View};
use zerolm_tensor::sha256::Sha256;

/// Weight group of every compiled constant (`model/specs/weight-groups.json`).
pub const GROUP: &str = "w_prog";

/// A typed circuit specification (`model/prog/<id>.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub id: String,
    pub version: u32,
    /// Seed-knowledge entry that documents this circuit.
    pub seed_entry: String,
    pub description: String,
    pub circuit: Circuit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Circuit {
    AttentionRead(circuits::attention_read::Params),
    KeyEncode(circuits::key_encode::Params),
    RelationGraph(circuits::relational::Params),
    RelationReach(circuits::relational::Params),
}

/// One compiled constant. `block` names the circuit block it belongs to (the
/// unit of ablation); `adaptable` marks blocks local plasticity may change.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgTensor {
    /// Program-local name (also the program weight name).
    pub name: String,
    pub block: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    pub adaptable: bool,
}

impl ProgTensor {
    pub fn view(&self) -> View<'_> {
        View::new(self.rows, self.cols, &self.data)
    }
}

/// Compiler output.
#[derive(Debug, Clone, PartialEq)]
pub struct Compiled {
    pub id: String,
    pub program: Program,
    pub tensors: Vec<ProgTensor>,
    /// Program weights that the caller binds from the model (not compiled).
    pub external: Vec<String>,
}

impl Compiled {
    /// Checkpoint tensor name of a compiled constant.
    pub fn qualified(&self, local: &str) -> String {
        format!("prog.{}.{local}", self.id)
    }

    pub fn tensor(&self, name: &str) -> Option<&ProgTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn blocks(&self) -> Vec<String> {
        let mut b: Vec<String> = self.tensors.iter().map(|t| t.block.clone()).collect();
        b.dedup();
        b
    }

    /// SHA-256 over the program JSON and every constant (name, block, shape, f32 bits).
    pub fn sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"zerolm-wsp-compiled-v1\0");
        h.update(self.id.as_bytes());
        h.update(self.program.to_json().as_bytes());
        for t in &self.tensors {
            h.update(t.name.as_bytes());
            h.update(t.block.as_bytes());
            h.update(&(t.rows as u64).to_le_bytes());
            h.update(&(t.cols as u64).to_le_bytes());
            h.update(&[t.adaptable as u8]);
            h.update_f32(&t.data);
        }
        h.finish_hex()
    }

    /// Zero every constant of one block (a causal ablation; evaluation only).
    pub fn ablate_block(&mut self, block: &str) -> Result<(), IsaError> {
        let mut hit = false;
        for t in self.tensors.iter_mut().filter(|t| t.block == block) {
            t.data.iter_mut().for_each(|x| *x = 0.0);
            hit = true;
        }
        if hit {
            Ok(())
        } else {
            Err(IsaError(format!("{}: no block '{block}'", self.id)))
        }
    }
}

pub fn compile(spec: &Spec) -> Result<Compiled, IsaError> {
    let c = match &spec.circuit {
        Circuit::AttentionRead(p) => circuits::attention_read::compile(&spec.id, p)?,
        Circuit::KeyEncode(p) => circuits::key_encode::compile(&spec.id, p)?,
        Circuit::RelationGraph(p) => circuits::relational::compile_graph(&spec.id, p)?,
        Circuit::RelationReach(p) => circuits::relational::compile_reach(&spec.id, p)?,
    };
    c.program.validate()?;
    Ok(c)
}

pub fn parse_spec(json: &str) -> Result<Spec, String> {
    serde_json::from_str(json).map_err(|e| format!("circuit spec: {e}"))
}

/// Small helper for writing programs: tracks declarations, constants and the
/// current block.
pub struct Builder {
    name: String,
    inputs: Vec<Decl>,
    weights: Vec<Decl>,
    body: Vec<Instr>,
    tensors: Vec<ProgTensor>,
    external: Vec<String>,
    block: String,
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Builder {
            name: name.into(),
            inputs: Vec::new(),
            weights: Vec::new(),
            body: Vec::new(),
            tensors: Vec::new(),
            external: Vec::new(),
            block: "main".into(),
        }
    }

    pub fn block(&mut self, b: &str) {
        self.block = b.into();
    }

    pub fn input(&mut self, n: &str, rows: Option<usize>, cols: usize) {
        self.inputs.push(Decl::new(n, rows, cols));
    }

    /// A per-run input whose rows and/or columns the caller fixes.
    pub fn input_dyn(&mut self, n: &str, rows: Option<usize>, cols: Option<usize>) {
        self.inputs.push(Decl::dynamic(n, rows, cols));
    }

    /// A model tensor the caller binds at run time.
    pub fn external(&mut self, n: &str, rows: usize, cols: usize) {
        self.weights.push(Decl::new(n, Some(rows), cols));
        self.external.push(n.into());
    }

    /// A model tensor whose row count the caller fixes at run time (for
    /// example the used prefix of a slot table).
    pub fn external_dyn(&mut self, n: &str, rows: Option<usize>, cols: Option<usize>) {
        self.weights.push(Decl::dynamic(n, rows, cols));
        self.external.push(n.into());
    }

    /// A compiled `W_PROG` constant in the current block.
    pub fn constant(&mut self, n: &str, rows: usize, cols: usize, data: Vec<f32>) {
        assert_eq!(rows * cols, data.len(), "constant {n}");
        self.weights.push(Decl::new(n, Some(rows), cols));
        self.tensors.push(ProgTensor { name: n.into(), block: self.block.clone(), rows, cols, data, adaptable: false });
    }

    pub fn scalar(&mut self, n: &str, x: f32) {
        self.constant(n, 1, 1, vec![x]);
    }

    pub fn op(&mut self, i: Instr) {
        self.body.push(i);
    }

    pub fn gate(&mut self, f: GateFn, a: &str, b: &str, out: &str) {
        self.op(Instr::Gate { f, a: a.into(), b: Some(b.into()), out: out.into() });
    }

    pub fn unary(&mut self, f: GateFn, a: &str, out: &str) {
        self.op(Instr::Gate { f, a: a.into(), b: None, out: out.into() });
    }

    pub fn matvec(&mut self, w: &str, x: &str, out: &str, transpose: bool) {
        self.op(Instr::Matvec { w: w.into(), x: x.into(), out: out.into(), transpose });
    }

    pub fn matmul(&mut self, a: &str, b: &str, out: &str) {
        self.op(Instr::Matmul { a: a.into(), b: b.into(), out: out.into(), ab: false });
    }

    pub fn reduce(&mut self, f: ReduceFn, x: &str, axis: Axis, out: &str) {
        self.op(Instr::Reduce { f, x: x.into(), axis, out: out.into() });
    }

    /// out = a b (plain product).
    pub fn matmul_ab(&mut self, a: &str, b: &str, out: &str) {
        self.op(Instr::Matmul { a: a.into(), b: b.into(), out: out.into(), ab: true });
    }

    pub fn reshape(&mut self, x: &str, rows: usize, cols: usize, out: &str) {
        self.op(Instr::Reshape { x: x.into(), rows, cols, out: out.into() });
    }

    pub fn finish(self, outputs: &[&str]) -> Compiled {
        Compiled {
            id: self.name.clone(),
            program: Program {
                name: self.name,
                inputs: self.inputs,
                weights: self.weights,
                state: Vec::new(),
                body: self.body,
                outputs: outputs.iter().map(|s| s.to_string()).collect(),
            },
            tensors: self.tensors,
            external: self.external,
        }
    }
}
