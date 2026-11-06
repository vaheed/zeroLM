//! Phase 4: online learning rules for the byte-level language model.
//! No offline training loop exists here: every update is driven by one
//! observed byte, in stream order, and is logged.

pub mod ngram;
pub mod online_predictor;
pub mod streams;

pub use online_predictor::{Constants, EvalMetrics, OnlinePredictor, Rule};
