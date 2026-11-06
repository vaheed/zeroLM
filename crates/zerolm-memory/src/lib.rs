//! Phase 5: weight-resident associative memory.
//!
//! Facts are stored only as f32 tensors in the model checkpoint
//! (`memory.semantic.*`, `memory.episodic.*`, `memory.fast`) and are answered
//! only by the compiled read circuit in [`read_circuit`] (a neural ISA
//! program, `model/prog/memory-read.json`). There is no JSON fact
//! database, transcript replay, SQLite, or vector database. The core tensors
//! (`core.*`: byte embedding, GRU cell, key encoder) are frozen and verified
//! byte-identical on load.

pub mod checkpoint;
pub mod encoder;
pub mod fast_weights;
pub mod persistent_memory;
pub mod read_circuit;
pub mod retrieval;

pub use persistent_memory::{Ablation, Answer, MemoryConfig, MemoryError, MemoryEvent, MemoryStore, Outcome, Stats};
