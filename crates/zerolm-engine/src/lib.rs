//! Forward pass only. Weights are programmed/seeded, never trained here.
//! Session state is a fixed-size vector; nothing grows with stream length.

pub mod bootstrap;

use zerolm_nn::{build_with_budget, state, Cell, Kind, Trace};
use zerolm_tensor::sha256::Sha256;
use zerolm_tensor::{all_finite, log_sum_exp, Matrix};

pub const VOCAB: usize = 256;

#[derive(Debug, PartialEq)]
pub enum EngineError {
    NonFinite(&'static str),
}

pub struct Model {
    pub embed: Matrix,
    pub cell: Box<dyn Cell>,
    pub head: Matrix,
    pub head_bias: Vec<f32>,
}

impl Model {
    /// `budget` is the cell's parameter budget; embedding and head are extra.
    pub fn new(kind: Kind, seed: u64, embed_dim: usize, budget: usize) -> Self {
        let cell = build_with_budget(kind, embed_dim, budget, seed);
        let d = cell.hidden_dim();
        Model {
            embed: Matrix::uniform(VOCAB, embed_dim, seed, "embed", 1.0),
            head: Matrix::uniform(VOCAB, d, seed, "head", 0.1 / (d as f32).sqrt()),
            head_bias: vec![0.0; VOCAB],
            cell,
        }
    }

    pub fn param_count(&self) -> usize {
        self.embed.data.len() + self.cell.param_count() + self.head.data.len() + self.head_bias.len()
    }

    /// Size of a plain f32 checkpoint of all parameters.
    pub fn checkpoint_bytes(&self) -> usize {
        self.param_count() * 4
    }

    /// SHA-256 over every parameter (name, length, f32 bits) plus a header.
    /// Equal hashes mean byte-identical weights.
    pub fn weights_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"zerolm-weights-v1\0");
        h.update(self.cell.kind().name().as_bytes());
        for (name, data) in [("embed", &self.embed.data[..]), ("head", &self.head.data), ("head_bias", &self.head_bias)] {
            h.update(name.as_bytes());
            h.update(&(data.len() as u64).to_le_bytes());
            h.update_f32(data);
        }
        for (name, data) in self.cell.groups() {
            h.update(name.as_bytes());
            h.update(&(data.len() as u64).to_le_bytes());
            h.update_f32(data);
        }
        h.finish_hex()
    }

    pub fn init_state(&self) -> Vec<f32> {
        self.cell.init_state()
    }

    /// Next-byte logits from the current state (no state change).
    pub fn logits(&self, st: &[f32], out: &mut [f32]) {
        self.head.matvec(&st[..self.cell.hidden_dim()], out);
        for (o, b) in out.iter_mut().zip(&self.head_bias) {
            *o += b;
        }
    }

    /// Consume one byte.
    pub fn step(&self, st: &mut [f32], byte: u8, trace: Option<&mut dyn Trace>) -> Result<(), EngineError> {
        self.cell.step(st, self.embed.row(byte as usize), trace);
        if all_finite(st) {
            Ok(())
        } else {
            Err(EngineError::NonFinite("state"))
        }
    }

    /// Consume a chunk; equivalent to calling `step` per byte.
    pub fn feed(&self, st: &mut [f32], bytes: &[u8]) -> Result<(), EngineError> {
        bytes.iter().try_for_each(|&b| self.step(st, b, None))
    }

    /// Bits per byte of predicting each byte from the state before seeing it.
    pub fn bits_per_byte(&self, bytes: &[u8]) -> Result<f64, EngineError> {
        let mut st = self.init_state();
        let mut logits = vec![0.0; VOCAB];
        let mut bits = 0.0f64;
        for &b in bytes {
            self.logits(&st, &mut logits);
            if !all_finite(&logits) {
                return Err(EngineError::NonFinite("logits"));
            }
            let lp = logits[b as usize] - log_sum_exp(&logits);
            bits -= lp as f64 / std::f64::consts::LN_2;
            self.step(&mut st, b, None)?;
        }
        Ok(bits / bytes.len().max(1) as f64)
    }

    pub fn save_state(&self, st: &[f32]) -> Vec<u8> {
        state::encode(self.cell.kind().name(), st)
    }

    pub fn load_state(&self, bytes: &[u8]) -> Result<Vec<f32>, state::StateError> {
        state::decode(self.cell.kind().name(), self.cell.state_len(), bytes)
    }
}
