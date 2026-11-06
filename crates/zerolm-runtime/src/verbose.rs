//! `--verbose` (or `ZEROLM_VERBOSE=1`): a human-readable view, on stderr, of
//! what happens inside each step: how a line was parsed, which tensors a write
//! changed, which operators and premises an answer used, and which checkpoint
//! resulted. It reports the same facts as `--trace` and never changes
//! behavior; stdout stays exactly the replies.

use std::fmt::Display;

pub struct View {
    on: bool,
}

impl View {
    pub fn new(flag: bool) -> View {
        let env = std::env::var("ZEROLM_VERBOSE").map(|v| !v.is_empty() && v != "0").unwrap_or(false);
        View { on: flag || env }
    }

    pub fn on(&self) -> bool {
        self.on
    }

    /// Header line for one input line.
    pub fn input(&self, seq: usize, line: &str) {
        if self.on {
            eprintln!("[{seq}] > {line}");
        }
    }

    /// One labelled detail under the current input line.
    pub fn note(&self, label: &str, value: impl Display) {
        if self.on {
            eprintln!("      {label:<11} {value}");
        }
    }
}

/// First 12 hex digits of a hash, enough to tell checkpoints apart by eye.
pub fn short(hash: &str) -> &str {
    &hash[..hash.len().min(12)]
}
