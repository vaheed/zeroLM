//! The compiled memory circuits: how a store encodes text and answers a
//! query. `memory-encode` (`model/prog/memory-encode.json`) turns raw text
//! bytes into the pooled key, word rows, and word mask
//! (`zerolm_wsp::circuits::key_encode`); `memory-read`
//! (`model/prog/memory-read.json`) takes those three tensors and does
//! selection, attention, verification, cleanup, and the accept decision
//! (`zerolm_wsp::circuits::attention_read`). The host copies bytes into the
//! input tensor, binds model tensors, runs the programs, and passes one
//! program's outputs to the other unchanged.

use crate::persistent_memory::{field, MemoryConfig};
use zerolm_engine::isa::{self, bundle, Bindings, Buf, IsaError, Outputs, View};
use zerolm_wsp::circuits::attention_read::Params;
use zerolm_wsp::circuits::key_encode;
use zerolm_wsp::{compile, Circuit, Compiled, Spec};

pub const ID: &str = "memory-read";
pub const ENCODE_ID: &str = "memory-encode";

/// Input width of the encoder: texts up to `TEXT_BYTES - 2` bytes.
pub const TEXT_BYTES: usize = 256;

/// The encoder spec for a configuration (equal to the shipped spec for the
/// frozen default configuration).
pub fn encode_spec(cfg: &MemoryConfig) -> Spec {
    Spec {
        id: ENCODE_ID.into(),
        version: 1,
        seed_entry: "memory-encode-circuit".into(),
        description: String::new(),
        circuit: Circuit::KeyEncode(key_encode::Params { text_bytes: TEXT_BYTES, buckets: cfg.buckets, key_dim: cfg.key_dim }),
    }
}

pub fn build_encoder(cfg: &MemoryConfig) -> Compiled {
    compile(&encode_spec(cfg)).expect("valid memory configuration compiles")
}

/// The encoder input: the text's bytes copied to positions 1..=len of a
/// zero row (data movement only; no byte is inspected). `None` if too long.
pub fn text_input(text: &str) -> Option<Buf> {
    let mut d = vec![0.0f32; TEXT_BYTES];
    if text.len() + 2 > TEXT_BYTES {
        return None;
    }
    for (slot, b) in d[1..].iter_mut().zip(text.bytes()) {
        *slot = b as f32;
    }
    Some(Buf::row(d))
}

fn encoder_bindings<'a>(c: &'a Compiled, table: View<'a>) -> Vec<(&'a str, View<'a>)> {
    let mut ws = vec![("table", table)];
    ws.extend(c.tensors.iter().map(|t| (t.name.as_str(), t.view())));
    ws
}

pub fn run_encoder(c: &Compiled, table: View<'_>, text: Buf) -> Result<Outputs, IsaError> {
    let mut b = Bindings::new().input("text", text);
    for (n, v) in encoder_bindings(c, table) {
        b = b.weight(n, v);
    }
    isa::run(&c.program, b)
}

/// Encoder program plus every tensor it reads, for an independent interpreter.
pub fn export_encoder(c: &Compiled, table: View<'_>) -> Vec<u8> {
    bundle::encode(&c.program, &encoder_bindings(c, table))
}

/// The read circuit's query inputs from the encoder's outputs, in order
/// (`q`, `qw`, `qmask`).
pub fn read_inputs(encoded: &Outputs) -> [Buf; 3] {
    [encoded.get("key").clone(), encoded.get("words").clone(), encoded.get("word_mask").clone()]
}

/// Read inputs for a single key vector used as its own only word.
pub fn key_inputs(k: &[f32]) -> [Buf; 3] {
    [Buf::row(k.to_vec()), Buf::row(k.to_vec()), Buf::scalar(1.0)]
}

/// Input names of the read circuit, matching [`read_inputs`].
pub const READ_INPUTS: [&str; 3] = ["q", "qw", "qmask"];

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

pub fn run(c: &Compiled, ext: &External<'_>, inputs: [Buf; 3]) -> Result<Outputs, IsaError> {
    let mut b = Bindings::new();
    for (n, v) in READ_INPUTS.into_iter().zip(inputs) {
        b = b.input(n, v);
    }
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
