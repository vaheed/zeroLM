//! The compiled read circuit (`model/prog/memory-read.json`): how a store
//! answers a query. The host encodes the query text into its pooled key and
//! word vectors, binds the memory tensors, and runs the program; selection,
//! attention, verification, cleanup, and the accept decision are all ISA
//! operations (see `zerolm_wsp::circuits::attention_read`).

use crate::persistent_memory::{field, MemoryConfig};
use zerolm_engine::isa::{self, bundle, Bindings, Buf, IsaError, Outputs, View};
use zerolm_wsp::circuits::attention_read::Params;
use zerolm_wsp::{compile, Circuit, Compiled, Spec};

pub const ID: &str = "memory-read";

/// The read-circuit spec for a configuration (equal to the shipped spec for
/// the frozen default configuration).
pub fn spec(cfg: &MemoryConfig, code_rows: usize, code_dim: usize) -> Spec {
    Spec {
        id: ID.into(),
        version: 1,
        seed_entry: "memory-read-circuit".into(),
        description: String::new(),
        circuit: Circuit::AttentionRead(Params {
            slots: cfg.slots,
            key_dim: cfg.key_dim,
            max_key_words: cfg.max_key_words,
            positions: cfg.positions(),
            code_rows,
            code_dim,
            meta_cols: field::COUNT,
            status_col: field::STATUS,
            active_status: crate::persistent_memory::ACTIVE,
            key_words_col: field::KEY_WORDS,
            beta: cfg.beta,
            read_threshold: cfg.read_threshold,
            coverage_threshold: cfg.coverage_threshold,
            min_weight: crate::retrieval::MIN_WEIGHT,
        }),
    }
}

pub fn build(cfg: &MemoryConfig, code_rows: usize, code_dim: usize) -> Compiled {
    compile(&spec(cfg, code_rows, code_dim)).expect("valid memory configuration compiles")
}

/// The model tensors the circuit reads, by program weight name.
pub struct External<'a> {
    pub keys: View<'a>,
    pub key_words: View<'a>,
    pub values: View<'a>,
    pub meta: View<'a>,
    pub codes: View<'a>,
}

impl<'a> External<'a> {
    fn named(&self) -> [(&'static str, View<'a>); 5] {
        [("keys", self.keys), ("key_words", self.key_words), ("values", self.values), ("meta", self.meta), ("codes", self.codes)]
    }
}

/// Query inputs: pooled key (1 x key_dim) and word vectors (words x key_dim).
pub fn inputs(q: &[f32], words: &[Vec<f32>]) -> (Buf, Buf) {
    let d = q.len();
    (Buf::row(q.to_vec()), Buf::new(words.len(), d, words.concat()))
}

pub fn run(c: &Compiled, ext: &External<'_>, q: &[f32], words: &[Vec<f32>]) -> Result<Outputs, IsaError> {
    let (qb, wb) = inputs(q, words);
    let mut b = Bindings::new().input("q", qb).input("qw", wb);
    for (n, v) in ext.named() {
        b = b.weight(n, v);
    }
    for t in &c.tensors {
        b = b.weight(&t.name, t.view());
    }
    isa::run(&c.program, b)
}

/// Program plus every tensor it reads, for an independent interpreter.
pub fn export(c: &Compiled, ext: &External<'_>) -> Vec<u8> {
    let mut ws: Vec<(&str, View<'_>)> = ext.named().to_vec();
    ws.extend(c.tensors.iter().map(|t| (t.name.as_str(), t.view())));
    bundle::encode(&c.program, &ws)
}

/// Bytes of a row up to (not including) the first 0 value.
pub fn bytes(b: &Buf) -> Vec<u8> {
    b.data.iter().take_while(|&&x| x != 0.0).map(|&x| x as u8).collect()
}
