//! ZeroLM Next: a compositional, manually weighted neural parser plus a separate world executor.
//!
//! Pipeline: text -> `encoder` (tokenize, embedding lookup) -> `network` (recurrent cell with
//! tensor-product binding, all semantics in `weights.zln`) -> `decoder` (validity, ordering,
//! linear readouts) -> `Op` list -> `world` (executor; never sees the text).
//!
//! The legacy v0.2 runtime in `crate::` is untouched and remains the baseline.

pub mod binding;
pub mod cli;
pub mod decoder;
pub mod encoder;
pub mod model;
pub mod network;
pub mod trace;
pub mod world;

pub use model::{Ablation, NextModel, Op, Parse};
