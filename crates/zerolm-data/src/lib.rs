//! Real-text data pipeline (roadmap Phase 8, `real-text`).
//!
//! - [`manifest`]: the corpus manifest (`research/corpus/manifest.json`):
//!   sources with licenses, SHA-256 per shard, split fractions, exposure
//!   order, and evaluation sets. Validation rejects model-generated text and
//!   benchmark data as exposure sources.
//! - [`corpus`]: hash verification of local shards, whole-document splits
//!   (a document's role is a function of its source, index, and the split
//!   seed only), and the deterministic exposure byte stream.
//! - [`contamination`]: word n-gram overlap between exposure and every
//!   evaluation set.
//!
//! No data is downloaded here; shards are read from a local root directory.

pub mod contamination;
pub mod corpus;
pub mod evalsets;
pub mod manifest;
