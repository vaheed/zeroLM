//! `W_MEMORY`: persistent semantic slots and an episodic event ring.
//!
//! Semantic bank (what is believed): `memory.semantic.keys` (slots x key_dim,
//! pooled keys for the attention read), `memory.semantic.key_words` (slots x
//! max_key_words*key_dim, per-word vectors for late-interaction verification),
//! `memory.semantic.values` (slots x positions*embed_dim) and
//! `memory.semantic.meta` (slots x META_FIELDS: status, confidence, creation
//! and update time, source, provenance, supersession links, contradictions).
//! Episodic ring (what happened): `memory.episodic.keys` and
//! `memory.episodic.meta`, one row per memory operation.
//!
//! Time is a logical event clock (deterministic), not wall-clock time.
//! Integers are stored as f32 in 24-bit pieces, which f32 represents exactly.

use crate::checkpoint::{self, CheckpointError, Fault, LoadSource, Tensor};
use crate::encoder::{self, KeyEncoder};
use crate::fast_weights::{self, FastWeights};
use crate::read_circuit;
use crate::retrieval::{self, Decoded};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;
use zerolm_engine::bootstrap::default_model;
use zerolm_engine::Model;
use zerolm_tensor::sha256::Sha256;
use zerolm_tensor::{fnv1a, shuffle, Matrix};
use zerolm_wsp::Compiled;

pub const KEYS: &str = "memory.semantic.keys";
pub const KEY_WORDS: &str = "memory.semantic.key_words";
pub const VALUES: &str = "memory.semantic.values";
pub const META: &str = "memory.semantic.meta";
pub const EP_KEYS: &str = "memory.episodic.keys";
pub const EP_META: &str = "memory.episodic.meta";
pub const STATE: &str = "memory.state";
pub const EMBED: &str = "core.embed";
pub const CELL_PREFIX: &str = "core.cell.";

/// Semantic metadata columns.
pub mod field {
    pub const STATUS: usize = 0;
    pub const CONFIDENCE: usize = 1;
    pub const CREATED_LO: usize = 2;
    pub const CREATED_HI: usize = 3;
    pub const UPDATED_LO: usize = 4;
    pub const UPDATED_HI: usize = 5;
    pub const SOURCE: usize = 6;
    pub const PROV: usize = 7; // 7, 8, 9: 64-bit stream hash in 22-bit pieces
    pub const SUPERSEDES: usize = 10;
    pub const SUPERSEDED_BY: usize = 11;
    pub const CONTRADICTIONS: usize = 12;
    pub const KEY_WORDS: usize = 13;
    pub const COUNT: usize = 14;
}

/// Episodic metadata columns.
pub mod ep_field {
    pub const OP: usize = 0;
    pub const SEQ_LO: usize = 1;
    pub const SEQ_HI: usize = 2;
    pub const SLOT: usize = 3;
    pub const SOURCE: usize = 4;
    pub const PROV: usize = 5; // 5, 6, 7
    pub const VALID: usize = 8;
    pub const COUNT: usize = 9;
}

pub const EMPTY: f32 = 0.0;
pub const ACTIVE: f32 = 1.0;
pub const SUPERSEDED: f32 = 2.0;
pub const DELETED: f32 = 3.0;
pub const UNCONFIRMED: f32 = 0.5;
pub const CONFIRMED: f32 = 1.0;

/// Where a fact came from (`field::SOURCE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    UserStatement = 1,
    UserCorrection = 2,
}

impl Source {
    pub fn signal(self) -> &'static str {
        match self {
            Source::UserStatement => "user_statement",
            Source::UserCorrection => "user_correction",
        }
    }
}

/// Episodic operation codes (`ep_field::OP`).
pub const OPS: [&str; 8] = ["", "add", "unchanged", "contradiction", "supersede", "confirm", "forget", "compact"];

fn op_code(op: &str) -> f32 {
    OPS.iter().position(|o| *o == op).expect("known op") as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    /// Model seed: selects the frozen core (embedding, GRU, encoder).
    pub seed: u64,
    pub slots: usize,
    pub key_dim: usize,
    pub buckets: usize,
    /// Maximum value length in bytes (one more position holds the terminator).
    pub value_bytes: usize,
    /// Softmax sharpness of the attention read.
    pub beta: f32,
    /// Maximum words in a key (each gets a row in `memory.semantic.key_words`).
    pub max_key_words: usize,
    /// Minimum best-slot pooled cosine to answer.
    pub read_threshold: f32,
    /// Minimum late-interaction coverage of the best slot's key words by the
    /// query's words to answer; below either threshold the memory abstains.
    pub coverage_threshold: f32,
    /// Cosine at which a write addresses an existing slot (same fact key).
    pub same_key_threshold: f32,
    pub episodic_capacity: usize,
    pub fast_decay: f32,
    pub fast_lr: f32,
    /// Width of the frozen pattern-separation layer in front of W_FAST.
    pub fast_expansion: usize,
    /// Active units in the k-winners-take-all fast key.
    pub fast_active: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            seed: zerolm_engine::bootstrap::DEFAULT_SEED,
            slots: 2048,
            key_dim: 128,
            buckets: 4096,
            value_bytes: 47,
            beta: 32.0,
            max_key_words: 8,
            read_threshold: 0.3,
            coverage_threshold: 0.5,
            same_key_threshold: 0.95,
            episodic_capacity: 4096,
            fast_decay: 0.99,
            fast_lr: 1.0,
            fast_expansion: 1024,
            fast_active: 32,
        }
    }
}

impl MemoryConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.slots == 0 || self.slots >= 1 << 24 {
            return Err("slots must be in 1..2^24".into());
        }
        if self.key_dim == 0 || self.buckets == 0 || self.value_bytes == 0 || self.episodic_capacity == 0 || self.max_key_words == 0 {
            return Err("dimensions must be positive".into());
        }
        if !(self.beta > 0.0 && self.beta.is_finite()) {
            return Err("beta must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.read_threshold) || !(self.read_threshold..=1.0).contains(&self.same_key_threshold) {
            return Err("need 0 <= read_threshold <= same_key_threshold <= 1".into());
        }
        if !(0.0..=1.0).contains(&self.coverage_threshold) {
            return Err("coverage_threshold must be in [0,1]".into());
        }
        if !(0.0..=1.0).contains(&self.fast_decay) || self.fast_lr <= 0.0 {
            return Err("fast_decay in [0,1], fast_lr > 0".into());
        }
        if self.fast_active == 0 || self.fast_active > self.fast_expansion {
            return Err("need 0 < fast_active <= fast_expansion".into());
        }
        Ok(())
    }

    pub fn positions(&self) -> usize {
        self.value_bytes + 1
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryError {
    EmptyKey,
    KeyTooLong { max_words: usize },
    TextTooLong { max_bytes: usize },
    EmptyValue,
    NulInValue,
    ValueTooLong { max: usize },
    CapacityFull,
    Config(String),
    CoreMismatch(String),
    Format(String),
    Checkpoint(CheckpointError),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::EmptyKey => write!(f, "key has no word characters"),
            MemoryError::KeyTooLong { max_words } => write!(f, "key has more than {max_words} words"),
            MemoryError::TextTooLong { max_bytes } => write!(f, "key or query longer than {max_bytes} bytes"),
            MemoryError::EmptyValue => write!(f, "value is empty"),
            MemoryError::NulInValue => write!(f, "value contains a NUL byte"),
            MemoryError::ValueTooLong { max } => write!(f, "value longer than {max} bytes"),
            MemoryError::CapacityFull => write!(f, "all memory slots are in use (compact or forget first)"),
            MemoryError::Config(e) => write!(f, "config: {e}"),
            MemoryError::CoreMismatch(e) => write!(f, "frozen core tensor differs from its pinned initialization: {e}"),
            MemoryError::Format(e) => write!(f, "checkpoint content: {e}"),
            MemoryError::Checkpoint(e) => write!(f, "checkpoint: {e}"),
        }
    }
}

impl From<CheckpointError> for MemoryError {
    fn from(e: CheckpointError) -> Self {
        MemoryError::Checkpoint(e)
    }
}

/// Result of a memory operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    Stored { slot: usize },
    Unchanged { slot: usize },
    /// A different value is already believed; nothing was overwritten.
    Conflict { slot: usize },
    Superseded { old: usize, new: usize },
    Confirmed { slot: usize },
    Forgotten { slots: Vec<usize> },
    Unknown,
}

/// One logged update, with the fields `update-policy.json` requires.
/// `checkpoint_sha256` is filled in when the update becomes durable (the
/// next successful save); `None` means it has not been saved yet.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryEvent {
    pub seq: u64,
    pub op: &'static str,
    pub input_stream: String,
    pub signal: &'static str,
    pub rule: &'static str,
    pub changed_tensors: Vec<&'static str>,
    pub slot: Option<usize>,
    /// SHA-256 of the rows this event wrote.
    pub rows_sha256: String,
    pub checkpoint_sha256: Option<String>,
}

/// A query result. `value` is `None` when the memory abstains.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    pub value: Option<String>,
    /// What the read decoded to, even when abstaining (for PR sweeps).
    pub decoded: String,
    pub slot: Option<usize>,
    pub score: f32,
    pub runner_up: f32,
    /// Late-interaction coverage of the best slot's key words.
    pub coverage: f32,
    pub weight: f32,
    pub margin: f32,
    pub confidence: f32,
    pub contradictions: u32,
    pub created_seq: u64,
}

impl Answer {
    pub fn confirmed(&self) -> bool {
        self.confidence >= CONFIRMED
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Stats {
    pub slots: usize,
    pub active: usize,
    pub superseded: usize,
    pub deleted: usize,
    pub free: usize,
    pub episodes: u64,
    pub clock: u64,
}

/// One episodic record (oldest first in `history`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Episode {
    pub seq: u64,
    pub op: &'static str,
    pub slot: Option<usize>,
}

/// Evaluation-only damage to memory tensors (causal ablations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ablation {
    ZeroValues,
    ZeroKeys,
    /// Permute key rows among active slots (breaks key-value binding).
    ShuffleKeys(u64),
}

/// Information about how a store was loaded.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoadInfo {
    pub source: LoadSource,
    pub current_error: Option<String>,
    pub checkpoint_sha256: String,
}

pub struct MemoryStore {
    cfg: MemoryConfig,
    core: Model,
    encoder: KeyEncoder,
    keys: Matrix,
    key_words: Matrix,
    values: Matrix,
    meta: Matrix,
    ep_keys: Matrix,
    ep_meta: Matrix,
    fast: FastWeights,
    separator: Matrix,
    /// The compiled key encoder (`W_PROG`, `prog.memory-encode.*`).
    encode: Compiled,
    /// The compiled read circuit (`W_PROG`, `prog.memory-read.*`).
    read: Compiled,
    clock: u64,
    ep_count: u64,
    events: Vec<MemoryEvent>,
    unsaved_from: usize,
}

const P24: u64 = 1 << 24;

fn split48(x: u64) -> (f32, f32) {
    ((x % P24) as f32, ((x / P24) % P24) as f32)
}

fn join48(lo: f32, hi: f32) -> u64 {
    lo as u64 + hi as u64 * P24
}

fn prov(stream: &str) -> [f32; 3] {
    let h = fnv1a(stream.as_bytes());
    let m = (1u64 << 22) - 1;
    [(h & m) as f32, ((h >> 22) & m) as f32, (h >> 44) as f32]
}

fn slot_ref(x: f32) -> Option<usize> {
    if x < 0.0 {
        None
    } else {
        Some(x as usize)
    }
}

impl MemoryStore {
    /// A clean store: frozen core from the pinned seed, all memory zero.
    pub fn new(cfg: MemoryConfig) -> Result<Self, MemoryError> {
        cfg.validate().map_err(MemoryError::Config)?;
        let core = default_model(cfg.seed);
        let e = core.embed.cols;
        let mut meta = Matrix::zeros(cfg.slots, field::COUNT);
        for r in 0..cfg.slots {
            meta.data[r * field::COUNT + field::SUPERSEDES] = -1.0;
            meta.data[r * field::COUNT + field::SUPERSEDED_BY] = -1.0;
        }
        Ok(MemoryStore {
            encoder: KeyEncoder::new(cfg.seed, cfg.buckets, cfg.key_dim),
            keys: Matrix::zeros(cfg.slots, cfg.key_dim),
            key_words: Matrix::zeros(cfg.slots, cfg.max_key_words * cfg.key_dim),
            values: Matrix::zeros(cfg.slots, cfg.positions() * e),
            meta,
            ep_keys: Matrix::zeros(cfg.episodic_capacity, cfg.key_dim),
            ep_meta: Matrix::zeros(cfg.episodic_capacity, ep_field::COUNT),
            fast: FastWeights::new(cfg.positions() * e, cfg.fast_expansion, cfg.fast_decay, cfg.fast_lr),
            separator: Matrix::uniform(cfg.fast_expansion, cfg.key_dim, cfg.seed, fast_weights::SEPARATOR, 1.0),
            encode: read_circuit::build_encoder(&cfg),
            read: read_circuit::build(&cfg, core.embed.rows, e),
            core,
            cfg,
            clock: 0,
            ep_count: 0,
            events: Vec::new(),
            unsaved_from: 0,
        })
    }

    pub fn config(&self) -> &MemoryConfig {
        &self.cfg
    }

    pub fn events(&self) -> &[MemoryEvent] {
        &self.events
    }

    pub fn encoder(&self) -> &KeyEncoder {
        &self.encoder
    }

    /// Raw outputs of the compiled key encoder for a text.
    pub fn encode_outputs(&self, text: &str) -> Result<zerolm_engine::isa::Outputs, MemoryError> {
        let input = read_circuit::text_input(text).ok_or(MemoryError::TextTooLong { max_bytes: read_circuit::TEXT_BYTES - 2 })?;
        Ok(read_circuit::run_encoder(&self.encode, self.encoder.view(), input).expect("the compiled encoder runs on its own table"))
    }

    /// Encoder outputs of a text with at least one word.
    fn encoded(&self, text: &str) -> Result<zerolm_engine::isa::Outputs, MemoryError> {
        let o = self.encode_outputs(text)?;
        if o.scalar("n_words") == 0.0 {
            return Err(MemoryError::EmptyKey);
        }
        Ok(o)
    }

    /// Pooled key and the first `max_key_words` word rows (row-major; rows
    /// past the key's word count are zero) of a write key, and its word count.
    pub fn key(&self, text: &str) -> Result<(Vec<f32>, Vec<f32>, usize), MemoryError> {
        let o = self.encoded(text)?;
        let n = o.scalar("n_words") as usize;
        if n > self.cfg.max_key_words {
            return Err(MemoryError::KeyTooLong { max_words: self.cfg.max_key_words });
        }
        let words = o.get("words").data[..self.key_words.cols].to_vec();
        Ok((o.get("key").data.clone(), words, n))
    }

    fn pooled(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        Ok(self.encoded(text)?.get("key").data.clone())
    }

    fn m(&self, slot: usize, f: usize) -> f32 {
        self.meta.data[slot * field::COUNT + f]
    }

    fn set(&mut self, slot: usize, f: usize, v: f32) {
        self.meta.data[slot * field::COUNT + f] = v;
    }

    pub fn status(&self, slot: usize) -> f32 {
        self.m(slot, field::STATUS)
    }

    pub fn active_slots(&self) -> Vec<usize> {
        (0..self.cfg.slots).filter(|&i| self.status(i) == ACTIVE).collect()
    }

    pub fn stats(&self) -> Stats {
        let count = |s: f32| (0..self.cfg.slots).filter(|&i| self.status(i) == s).count();
        Stats {
            slots: self.cfg.slots,
            active: count(ACTIVE),
            superseded: count(SUPERSEDED),
            deleted: count(DELETED),
            free: count(EMPTY),
            episodes: self.ep_count,
            clock: self.clock,
        }
    }

    fn check_value(&self, value: &str) -> Result<(), MemoryError> {
        let b = value.as_bytes();
        if b.is_empty() {
            return Err(MemoryError::EmptyValue);
        }
        if b.contains(&0) {
            return Err(MemoryError::NulInValue);
        }
        if b.len() > self.cfg.value_bytes {
            return Err(MemoryError::ValueTooLong { max: self.cfg.value_bytes });
        }
        Ok(())
    }

    fn encode_value(&self, value: &str) -> Vec<f32> {
        retrieval::encode_value(value.as_bytes(), &self.core.embed, self.cfg.positions())
    }

    /// Decode exactly what one slot holds (used for write-time comparison).
    fn slot_value(&self, slot: usize) -> Vec<u8> {
        retrieval::decode(self.values.row(slot), &self.core.embed).bytes
    }

    /// Active slot whose key is the same fact key: the read circuit's best
    /// slot when its score reaches `same_key_threshold`.
    fn find_same(&self, k: &[f32]) -> Option<usize> {
        let a = self.answer(read_circuit::key_inputs(k));
        a.slot.filter(|_| a.score >= self.cfg.same_key_threshold)
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn rows_sha(&self, slot: Option<usize>, ep_row: Option<usize>) -> String {
        let mut h = Sha256::new();
        if let Some(s) = slot {
            for m in [&self.keys, &self.key_words, &self.values, &self.meta] {
                h.update_f32(m.row(s));
            }
        }
        if let Some(r) = ep_row {
            h.update_f32(self.ep_keys.row(r));
            h.update_f32(self.ep_meta.row(r));
        }
        h.finish_hex()
    }

    fn episode(&mut self, op: &str, seq: u64, key: Option<&[f32]>, slot: Option<usize>, source: Option<Source>, stream: &str) -> usize {
        let r = (self.ep_count % self.cfg.episodic_capacity as u64) as usize;
        self.ep_count += 1;
        let c = self.ep_keys.cols;
        match key {
            Some(k) => self.ep_keys.data[r * c..(r + 1) * c].copy_from_slice(k),
            None => self.ep_keys.data[r * c..(r + 1) * c].iter_mut().for_each(|x| *x = 0.0),
        }
        let (lo, hi) = split48(seq);
        let p = prov(stream);
        let row = &mut self.ep_meta.data[r * ep_field::COUNT..(r + 1) * ep_field::COUNT];
        row[ep_field::OP] = op_code(op);
        row[ep_field::SEQ_LO] = lo;
        row[ep_field::SEQ_HI] = hi;
        row[ep_field::SLOT] = slot.map(|s| s as f32).unwrap_or(-1.0);
        row[ep_field::SOURCE] = source.map(|s| s as u8 as f32).unwrap_or(0.0);
        row[ep_field::PROV..ep_field::PROV + 3].copy_from_slice(&p);
        row[ep_field::VALID] = 1.0;
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn log(&mut self, seq: u64, op: &'static str, stream: &str, signal: &'static str, rule: &'static str, changed: Vec<&'static str>, slot: Option<usize>, ep_row: Option<usize>) {
        let rows_sha256 = self.rows_sha(slot, ep_row);
        self.events.push(MemoryEvent {
            seq,
            op,
            input_stream: stream.to_string(),
            signal,
            rule,
            changed_tensors: changed,
            slot,
            rows_sha256,
            checkpoint_sha256: None,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn write_slot(&mut self, slot: usize, k: &[f32], words: &[f32], n_words: usize, value: &str, source: Source, stream: &str, seq: u64) {
        let v = self.encode_value(value);
        let kc = self.keys.cols;
        self.keys.data[slot * kc..(slot + 1) * kc].copy_from_slice(k);
        let wc = self.key_words.cols;
        self.key_words.data[slot * wc..(slot + 1) * wc].copy_from_slice(words);
        self.set(slot, field::KEY_WORDS, n_words as f32);
        let vc = self.values.cols;
        self.values.data[slot * vc..(slot + 1) * vc].copy_from_slice(&v);
        let (lo, hi) = split48(seq);
        let p = prov(stream);
        self.set(slot, field::STATUS, ACTIVE);
        self.set(slot, field::CONFIDENCE, UNCONFIRMED);
        self.set(slot, field::CREATED_LO, lo);
        self.set(slot, field::CREATED_HI, hi);
        self.set(slot, field::UPDATED_LO, lo);
        self.set(slot, field::UPDATED_HI, hi);
        self.set(slot, field::SOURCE, source as u8 as f32);
        for (i, x) in p.iter().enumerate() {
            self.set(slot, field::PROV + i, *x);
        }
        self.set(slot, field::SUPERSEDES, -1.0);
        self.set(slot, field::SUPERSEDED_BY, -1.0);
        self.set(slot, field::CONTRADICTIONS, 0.0);
    }

    fn touch(&mut self, slot: usize, seq: u64) {
        let (lo, hi) = split48(seq);
        self.set(slot, field::UPDATED_LO, lo);
        self.set(slot, field::UPDATED_HI, hi);
    }

    fn free_slot(&self) -> Option<usize> {
        (0..self.cfg.slots).find(|&i| self.status(i) == EMPTY)
    }

    /// `teach`: store a new fact; never overwrites a different belief.
    pub fn teach(&mut self, key: &str, value: &str, stream: &str) -> Result<Outcome, MemoryError> {
        let (k, words, n_words) = self.key(key)?;
        self.check_value(value)?;
        let src = Source::UserStatement;
        if let Some(s) = self.find_same(&k) {
            let seq = self.tick();
            if self.slot_value(s) == value.as_bytes() {
                let r = self.episode("unchanged", seq, Some(&k), Some(s), Some(src), stream);
                self.log(seq, "unchanged", stream, src.signal(), "memory_write", vec![EP_KEYS, EP_META], None, Some(r));
                return Ok(Outcome::Unchanged { slot: s });
            }
            let c = self.m(s, field::CONTRADICTIONS) + 1.0;
            self.set(s, field::CONTRADICTIONS, c);
            let r = self.episode("contradiction", seq, Some(&k), Some(s), Some(src), stream);
            self.log(seq, "contradiction", stream, src.signal(), "memory_write", vec![META, EP_KEYS, EP_META], Some(s), Some(r));
            return Ok(Outcome::Conflict { slot: s });
        }
        let slot = self.free_slot().ok_or(MemoryError::CapacityFull)?;
        let seq = self.tick();
        self.write_slot(slot, &k, &words, n_words, value, src, stream, seq);
        let r = self.episode("add", seq, Some(&k), Some(slot), Some(src), stream);
        self.log(seq, "add", stream, src.signal(), "memory_write", vec![KEYS, KEY_WORDS, VALUES, META, EP_KEYS, EP_META], Some(slot), Some(r));
        Ok(Outcome::Stored { slot })
    }

    /// `correct`: the new value supersedes the current belief (kept, marked
    /// superseded, until compaction); stores it if no belief exists.
    pub fn correct(&mut self, key: &str, value: &str, stream: &str) -> Result<Outcome, MemoryError> {
        let (k, words, n_words) = self.key(key)?;
        self.check_value(value)?;
        let src = Source::UserCorrection;
        let old = self.find_same(&k);
        if let Some(s) = old {
            if self.slot_value(s) == value.as_bytes() {
                let seq = self.tick();
                let r = self.episode("unchanged", seq, Some(&k), Some(s), Some(src), stream);
                self.log(seq, "unchanged", stream, src.signal(), "memory_write", vec![EP_KEYS, EP_META], None, Some(r));
                return Ok(Outcome::Unchanged { slot: s });
            }
        }
        let slot = self.free_slot().ok_or(MemoryError::CapacityFull)?;
        let seq = self.tick();
        self.write_slot(slot, &k, &words, n_words, value, src, stream, seq);
        match old {
            Some(s) => {
                self.set(s, field::STATUS, SUPERSEDED);
                self.set(s, field::SUPERSEDED_BY, slot as f32);
                self.touch(s, seq);
                self.set(slot, field::SUPERSEDES, s as f32);
                let r = self.episode("supersede", seq, Some(&k), Some(slot), Some(src), stream);
                self.log(seq, "supersede", stream, src.signal(), "memory_write", vec![KEYS, KEY_WORDS, VALUES, META, EP_KEYS, EP_META], Some(slot), Some(r));
                Ok(Outcome::Superseded { old: s, new: slot })
            }
            None => {
                let r = self.episode("add", seq, Some(&k), Some(slot), Some(src), stream);
                self.log(seq, "add", stream, src.signal(), "memory_write", vec![KEYS, KEY_WORDS, VALUES, META, EP_KEYS, EP_META], Some(slot), Some(r));
                Ok(Outcome::Stored { slot })
            }
        }
    }

    /// `confirm`: raise the stored belief's confidence to 1.
    pub fn confirm(&mut self, key: &str, stream: &str) -> Result<Outcome, MemoryError> {
        let k = self.pooled(key)?;
        let Some(s) = self.find_same(&k) else { return Ok(Outcome::Unknown) };
        let seq = self.tick();
        self.set(s, field::CONFIDENCE, CONFIRMED);
        self.touch(s, seq);
        let src = Source::UserCorrection;
        let r = self.episode("confirm", seq, Some(&k), Some(s), Some(src), stream);
        self.log(seq, "confirm", stream, src.signal(), "memory_write", vec![META, EP_KEYS, EP_META], Some(s), Some(r));
        Ok(Outcome::Confirmed { slot: s })
    }

    /// `forget`: delete the belief and every value it superseded. Key and
    /// value rows are zeroed (erased, not hidden); the episodic records that
    /// carry this key lose their key vector. Metadata keeps status=deleted.
    pub fn forget(&mut self, key: &str, stream: &str) -> Result<Outcome, MemoryError> {
        let k = self.pooled(key)?;
        let Some(s) = self.find_same(&k) else { return Ok(Outcome::Unknown) };
        let seq = self.tick();
        let mut chain = vec![s];
        let mut cur = s;
        while let Some(p) = slot_ref(self.m(cur, field::SUPERSEDES)) {
            if chain.contains(&p) {
                break;
            }
            chain.push(p);
            cur = p;
        }
        for &c in &chain {
            let (kc, vc, wc) = (self.keys.cols, self.values.cols, self.key_words.cols);
            self.keys.data[c * kc..(c + 1) * kc].iter_mut().for_each(|x| *x = 0.0);
            self.key_words.data[c * wc..(c + 1) * wc].iter_mut().for_each(|x| *x = 0.0);
            self.values.data[c * vc..(c + 1) * vc].iter_mut().for_each(|x| *x = 0.0);
            self.set(c, field::STATUS, DELETED);
            self.touch(c, seq);
        }
        for r in 0..self.cfg.episodic_capacity {
            let row = r * ep_field::COUNT;
            if self.ep_meta.data[row + ep_field::VALID] == 1.0 {
                if let Some(es) = slot_ref(self.ep_meta.data[row + ep_field::SLOT]) {
                    if chain.contains(&es) {
                        let c = self.ep_keys.cols;
                        self.ep_keys.data[r * c..(r + 1) * c].iter_mut().for_each(|x| *x = 0.0);
                    }
                }
            }
        }
        let src = Source::UserCorrection;
        let r = self.episode("forget", seq, None, Some(s), Some(src), stream);
        self.log(seq, "forget", stream, src.signal(), "memory_write", vec![KEYS, KEY_WORDS, VALUES, META, EP_KEYS, EP_META], Some(s), Some(r));
        chain.sort_unstable();
        Ok(Outcome::Forgotten { slots: chain })
    }

    /// Retrieval: the compiled encoder, then the compiled read circuit.
    pub fn query(&self, text: &str) -> Result<Answer, MemoryError> {
        Ok(self.answer(self.query_inputs(text)?))
    }

    /// Run the compiled read circuit. The host only formats its outputs.
    pub fn answer(&self, inputs: [zerolm_engine::isa::Buf; 3]) -> Answer {
        let o = self.read_outputs(inputs).expect("the compiled read circuit runs on its own store");
        let have = o.scalar("have") == 1.0;
        let meta = &o.get("best_meta").data;
        let value = read_circuit::bytes(o.get("value"));
        Answer {
            value: (!value.is_empty()).then(|| String::from_utf8_lossy(&value).into_owned()),
            decoded: String::from_utf8_lossy(&read_circuit::bytes(o.get("decoded"))).into_owned(),
            slot: have.then(|| o.scalar("best") as usize),
            score: o.scalar("score"),
            runner_up: o.scalar("runner_up"),
            coverage: o.scalar("coverage"),
            weight: o.scalar("weight"),
            margin: o.scalar("margin"),
            confidence: meta[field::CONFIDENCE],
            contradictions: meta[field::CONTRADICTIONS] as u32,
            created_seq: join48(meta[field::CREATED_LO], meta[field::CREATED_HI]),
        }
    }

    fn external(&self) -> read_circuit::External<'_> {
        fn v(m: &Matrix) -> zerolm_engine::isa::View<'_> {
            zerolm_engine::isa::View::new(m.rows, m.cols, &m.data)
        }
        read_circuit::External { keys: v(&self.keys), key_words: v(&self.key_words), values: v(&self.values), meta: v(&self.meta), codes: v(&self.core.embed) }
    }

    /// Raw outputs of the read circuit for its inputs (`q`, `qw`, `qmask`).
    pub fn read_outputs(&self, inputs: [zerolm_engine::isa::Buf; 3]) -> Result<zerolm_engine::isa::Outputs, zerolm_engine::isa::IsaError> {
        read_circuit::run(&self.read, &self.external(), inputs)
    }

    /// The read circuit's inputs for a query text: the encoder's outputs.
    pub fn query_inputs(&self, text: &str) -> Result<[zerolm_engine::isa::Buf; 3], MemoryError> {
        Ok(read_circuit::read_inputs(&self.encoded(text)?))
    }

    /// The read program and every tensor it reads, as a program bundle.
    pub fn read_bundle(&self) -> Vec<u8> {
        read_circuit::export(&self.read, &self.external())
    }

    /// The encoder program and every tensor it reads, as a program bundle.
    pub fn encode_bundle(&self) -> Vec<u8> {
        read_circuit::export_encoder(&self.encode, self.encoder.view())
    }

    pub fn read_circuit(&self) -> &Compiled {
        &self.read
    }

    pub fn encode_circuit(&self) -> &Compiled {
        &self.encode
    }

    /// Zero one block of the read circuit (causal ablation; evaluation only).
    pub fn ablate_circuit(&mut self, block: &str) -> Result<(), zerolm_engine::isa::IsaError> {
        self.read.ablate_block(block)
    }

    /// Zero one block of the key encoder circuit (causal ablation; evaluation only).
    pub fn ablate_encoder(&mut self, block: &str) -> Result<(), zerolm_engine::isa::IsaError> {
        self.encode.ablate_block(block)
    }

    /// Episodic records whose key is this fact key, oldest first.
    pub fn history(&self, text: &str) -> Result<Vec<Episode>, MemoryError> {
        let q = self.pooled(text)?;
        let mut out = Vec::new();
        for r in 0..self.cfg.episodic_capacity {
            let m = self.ep_meta.row(r);
            if m[ep_field::VALID] == 1.0 && zerolm_tensor::dot(self.ep_keys.row(r), &q) >= self.cfg.same_key_threshold {
                out.push(Episode {
                    seq: join48(m[ep_field::SEQ_LO], m[ep_field::SEQ_HI]),
                    op: OPS[m[ep_field::OP] as usize],
                    slot: slot_ref(m[ep_field::SLOT]),
                });
            }
        }
        out.sort_by_key(|e| e.seq);
        Ok(out)
    }

    /// Reclaim superseded and deleted slots. Active facts are moved to the
    /// lowest slots in order; supersession links and episodic slot references
    /// to reclaimed slots become -1. Returns the number of slots freed.
    pub fn compact(&mut self, stream: &str) -> usize {
        let active = self.active_slots();
        let freed = self.cfg.slots - active.len() - self.stats().free;
        let mut map = vec![-1.0f32; self.cfg.slots];
        for (new, &old) in active.iter().enumerate() {
            map[old] = new as f32;
        }
        let remap = |x: f32| if x < 0.0 { -1.0 } else { map[x as usize] };
        let (kc, vc, mc, wc) = (self.keys.cols, self.values.cols, field::COUNT, self.key_words.cols);
        let mut keys = Matrix::zeros(self.cfg.slots, kc);
        let mut key_words = Matrix::zeros(self.cfg.slots, wc);
        let mut values = Matrix::zeros(self.cfg.slots, vc);
        let mut meta = Matrix::zeros(self.cfg.slots, mc);
        for r in 0..self.cfg.slots {
            meta.data[r * mc + field::SUPERSEDES] = -1.0;
            meta.data[r * mc + field::SUPERSEDED_BY] = -1.0;
        }
        for (new, &old) in active.iter().enumerate() {
            keys.data[new * kc..(new + 1) * kc].copy_from_slice(self.keys.row(old));
            key_words.data[new * wc..(new + 1) * wc].copy_from_slice(self.key_words.row(old));
            values.data[new * vc..(new + 1) * vc].copy_from_slice(self.values.row(old));
            meta.data[new * mc..(new + 1) * mc].copy_from_slice(self.meta.row(old));
            meta.data[new * mc + field::SUPERSEDES] = remap(self.m(old, field::SUPERSEDES));
            meta.data[new * mc + field::SUPERSEDED_BY] = remap(self.m(old, field::SUPERSEDED_BY));
        }
        for r in 0..self.cfg.episodic_capacity {
            let i = r * ep_field::COUNT + ep_field::SLOT;
            self.ep_meta.data[i] = remap(self.ep_meta.data[i]);
        }
        self.keys = keys;
        self.key_words = key_words;
        self.values = values;
        self.meta = meta;
        let seq = self.tick();
        let r = self.episode("compact", seq, None, None, None, stream);
        self.log(seq, "compact", stream, "memory_maintenance", "compaction", vec![KEYS, KEY_WORDS, VALUES, META, EP_META], None, Some(r));
        freed
    }

    /// `W_FAST` write: a short-lived association (not a semantic belief).
    pub fn associate_fast(&mut self, key: &str, value: &str, stream: &str) -> Result<f32, MemoryError> {
        let k = self.pooled(key)?;
        self.check_value(value)?;
        let v = self.encode_value(value);
        let sk = fast_weights::separate(&self.separator, &k, self.cfg.fast_active);
        let n = self.fast.write(&sk, &v);
        let seq = self.tick();
        let mut h = Sha256::new();
        h.update_f32(&self.fast.m.data);
        self.events.push(MemoryEvent {
            seq,
            op: "fast_associate",
            input_stream: stream.into(),
            signal: Source::UserStatement.signal(),
            rule: "fast_delta",
            changed_tensors: vec![fast_weights::TENSOR],
            slot: None,
            rows_sha256: h.finish_hex(),
            checkpoint_sha256: None,
        });
        Ok(n)
    }

    pub fn recall_fast(&self, key: &str) -> Result<Decoded, MemoryError> {
        let k = self.pooled(key)?;
        let sk = fast_weights::separate(&self.separator, &k, self.cfg.fast_active);
        Ok(retrieval::decode(&self.fast.read(&sk), &self.core.embed))
    }

    /// Damage memory tensors for a causal ablation (evaluation only).
    pub fn ablate(&mut self, a: Ablation) {
        match a {
            Ablation::ZeroValues => self.values.data.iter_mut().for_each(|x| *x = 0.0),
            Ablation::ZeroKeys => {
                self.keys.data.iter_mut().for_each(|x| *x = 0.0);
                self.key_words.data.iter_mut().for_each(|x| *x = 0.0);
            }
            Ablation::ShuffleKeys(seed) => {
                let active = self.active_slots();
                let mut perm = active.clone();
                shuffle(&mut perm, seed);
                let rows: Vec<(Vec<f32>, Vec<f32>, f32)> = perm
                    .iter()
                    .map(|&s| (self.keys.row(s).to_vec(), self.key_words.row(s).to_vec(), self.m(s, field::KEY_WORDS)))
                    .collect();
                let (kc, wc) = (self.keys.cols, self.key_words.cols);
                for (&dst, (k, w, n)) in active.iter().zip(rows) {
                    self.keys.data[dst * kc..(dst + 1) * kc].copy_from_slice(&k);
                    self.key_words.data[dst * wc..(dst + 1) * wc].copy_from_slice(&w);
                    self.set(dst, field::KEY_WORDS, n);
                }
            }
        }
    }

    // ---- Checkpoint ----

    fn core_tensors(core: &Model, encoder: &KeyEncoder, separator: &Matrix, circuits: [&Compiled; 2]) -> Vec<Tensor> {
        let mut t = vec![Tensor::new(EMBED, "byte_embed", core.embed.rows, core.embed.cols, core.embed.data.clone())];
        for (name, data) in core.cell.groups() {
            t.push(Tensor::new(&format!("{CELL_PREFIX}{name}"), "recurrent_core", 1, data.len(), data.to_vec()));
        }
        let e = &encoder.table;
        t.push(Tensor::new(encoder::TENSOR, "memory_encoder", e.rows, e.cols, e.data.clone()));
        t.push(Tensor::new(fast_weights::SEPARATOR, "memory_encoder", separator.rows, separator.cols, separator.data.clone()));
        for prog in circuits {
            for c in &prog.tensors {
                t.push(Tensor::new(&prog.qualified(&c.name), zerolm_wsp::compiler::GROUP, c.rows, c.cols, c.data.clone()));
            }
        }
        t
    }

    /// Every tensor of the checkpoint, with its weight group.
    pub fn tensors(&self) -> Vec<Tensor> {
        let mut t = Self::core_tensors(&self.core, &self.encoder, &self.separator, [&self.encode, &self.read]);
        let mat = |name: &str, group: &str, m: &Matrix| Tensor::new(name, group, m.rows, m.cols, m.data.clone());
        t.push(mat(KEYS, "memory_keys", &self.keys));
        t.push(mat(KEY_WORDS, "memory_keys", &self.key_words));
        t.push(mat(VALUES, "memory_values", &self.values));
        t.push(mat(META, "memory_meta", &self.meta));
        t.push(mat(EP_KEYS, "episodic_memory", &self.ep_keys));
        t.push(mat(EP_META, "episodic_memory", &self.ep_meta));
        t.push(mat(fast_weights::TENSOR, "fast_weights", &self.fast.m));
        let (c0, c1) = split48(self.clock);
        let (e0, e1) = split48(self.ep_count);
        t.push(Tensor::new(STATE, "memory_meta", 1, 4, vec![c0, c1, e0, e1]));
        t
    }

    /// SHA-256 over the named tensors whose names start with `prefix`.
    pub fn digest(&self, prefix: &str) -> String {
        let mut h = Sha256::new();
        for t in self.tensors().iter().filter(|t| t.name.starts_with(prefix)) {
            h.update(t.sha256().as_bytes());
        }
        h.finish_hex()
    }

    pub fn checkpoint_bytes(&self) -> Vec<u8> {
        checkpoint::encode(&serde_json::json!({ "config": self.cfg }), &self.tensors())
    }

    /// Atomically save; marks every unsaved event with the checkpoint hash.
    pub fn save(&mut self, dir: &Path) -> Result<String, MemoryError> {
        self.save_with_fault(dir, Fault::None)
    }

    pub fn save_with_fault(&mut self, dir: &Path, fault: Fault) -> Result<String, MemoryError> {
        let sha = checkpoint::save_with_fault(dir, &self.checkpoint_bytes(), fault)?;
        for e in &mut self.events[self.unsaved_from..] {
            e.checkpoint_sha256 = Some(sha.clone());
        }
        self.unsaved_from = self.events.len();
        Ok(sha)
    }

    /// Load the newest valid checkpoint in `dir` (falling back to the previous one).
    pub fn load(dir: &Path) -> Result<(Self, LoadInfo), MemoryError> {
        let l = checkpoint::load_latest(dir)?;
        let store = Self::from_checkpoint(&l.meta, l.tensors)?;
        Ok((store, LoadInfo { source: l.source, current_error: l.current_error.map(|e| e.to_string()), checkpoint_sha256: l.sha256 }))
    }

    /// Load if a checkpoint exists, else start clean with `cfg`.
    pub fn open(dir: &Path, cfg: MemoryConfig) -> Result<(Self, Option<LoadInfo>), MemoryError> {
        match Self::load(dir) {
            Ok((s, info)) => Ok((s, Some(info))),
            Err(MemoryError::Checkpoint(CheckpointError::NotFound)) => Ok((Self::new(cfg)?, None)),
            Err(e) => Err(e),
        }
    }

    pub fn from_checkpoint(meta: &serde_json::Value, tensors: Vec<Tensor>) -> Result<Self, MemoryError> {
        let cfg: MemoryConfig = serde_json::from_value(meta.get("config").cloned().unwrap_or_default())
            .map_err(|e| MemoryError::Format(format!("config: {e}")))?;
        let mut s = Self::new(cfg)?;
        let expected = s.tensors();
        if tensors.len() != expected.len() {
            return Err(MemoryError::Format(format!("expected {} tensors, found {}", expected.len(), tensors.len())));
        }
        for (got, want) in tensors.into_iter().zip(expected) {
            if got.name != want.name || got.group != want.group || got.rows != want.rows || got.cols != want.cols {
                return Err(MemoryError::Format(format!("tensor '{}' does not match expected '{}' {}x{}", got.name, want.name, want.rows, want.cols)));
            }
            if got.name.starts_with("core.") || got.name.starts_with("prog.") {
                if got.data.iter().zip(&want.data).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    return Err(MemoryError::CoreMismatch(got.name));
                }
                continue;
            }
            if got.data.iter().any(|x| !x.is_finite()) {
                return Err(MemoryError::Format(format!("tensor '{}' has non-finite values", got.name)));
            }
            let target = match got.name.as_str() {
                KEYS => &mut s.keys.data,
                KEY_WORDS => &mut s.key_words.data,
                VALUES => &mut s.values.data,
                META => &mut s.meta.data,
                EP_KEYS => &mut s.ep_keys.data,
                EP_META => &mut s.ep_meta.data,
                fast_weights::TENSOR => &mut s.fast.m.data,
                STATE => {
                    s.clock = join48(got.data[0], got.data[1]);
                    s.ep_count = join48(got.data[2], got.data[3]);
                    continue;
                }
                other => return Err(MemoryError::Format(format!("unexpected tensor '{other}'"))),
            };
            *target = got.data;
        }
        Ok(s)
    }
}
