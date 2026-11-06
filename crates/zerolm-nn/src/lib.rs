//! Recurrent cell candidates for the architecture bake-off.
//!
//! All state lives in one fixed-size `Vec<f32>` per session (no KV cache
//! that grows with the stream). The first `hidden_dim` entries of the state
//! are the cell output. No backpropagation exists in this crate.

mod fastweight;
mod gru;
mod hybrid;
mod ssm;
pub mod state;

pub use fastweight::FastWeight;
pub use gru::Gru;
pub use hybrid::Hybrid;
pub use ssm::DiagSsm;

/// Receives named intermediate vectors during a step (inspection only).
pub trait Trace {
    fn record(&mut self, name: &'static str, v: &[f32]);
}

/// Trace that stores everything it sees.
#[derive(Default)]
pub struct VecTrace(pub Vec<(&'static str, Vec<f32>)>);

impl Trace for VecTrace {
    fn record(&mut self, name: &'static str, v: &[f32]) {
        self.0.push((name, v.to_vec()));
    }
}

pub(crate) fn rec(t: &mut Option<&mut dyn Trace>, name: &'static str, v: &[f32]) {
    if let Some(t) = t.as_deref_mut() {
        t.record(name, v);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Gru,
    Ssm,
    FastWeight,
    Hybrid,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Gru, Kind::Ssm, Kind::FastWeight, Kind::Hybrid];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Gru => "gru",
            Kind::Ssm => "ssm",
            Kind::FastWeight => "fast_weight",
            Kind::Hybrid => "hybrid",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.name() == s)
    }

    /// The hybrid is a comparison arm, not a selectable default (roadmap 3.4).
    pub fn eligible(self) -> bool {
        self != Kind::Hybrid
    }
}

pub trait Cell {
    fn kind(&self) -> Kind;
    fn input_dim(&self) -> usize;
    fn hidden_dim(&self) -> usize;
    fn param_count(&self) -> usize;
    fn state_len(&self) -> usize;
    fn init_state(&self) -> Vec<f32>;
    /// Advance `state` by one input vector.
    fn step(&self, state: &mut [f32], x: &[f32], trace: Option<&mut dyn Trace>);
    fn groups(&self) -> Vec<(&'static str, &[f32])>;
    fn groups_mut(&mut self) -> Vec<(&'static str, &mut [f32])>;
    /// Ablation: make the output independent of the previous state.
    fn sever_recurrence(&mut self);
    /// Ablation: shuffle the recurrent weights (keeps their distribution).
    fn shuffle_recurrence(&mut self, seed: u64);
    /// Number of named trace points emitted per step.
    fn trace_points(&self) -> usize;
}

/// Build a cell of hidden size `d`.
pub fn build(kind: Kind, input: usize, d: usize, seed: u64) -> Box<dyn Cell> {
    match kind {
        Kind::Gru => Box::new(Gru::new(input, d, seed)),
        Kind::Ssm => Box::new(DiagSsm::new(input, d, seed)),
        Kind::FastWeight => Box::new(FastWeight::new(input, d, seed)),
        Kind::Hybrid => Box::new(Hybrid::new(input, d, seed)),
    }
}

/// Closed-form parameter count (checked against real cells in tests).
pub fn param_count_for(kind: Kind, e: usize, d: usize) -> usize {
    let gru = 3 * d * e + 3 * d * d + 3 * d;
    match kind {
        Kind::Gru => gru,
        Kind::Ssm => 2 * d * e + d,
        Kind::FastWeight => {
            let dk = (d / 2).max(2);
            2 * d * e + 2 * d * d + 2 * dk * e + 2 * d
        }
        Kind::Hybrid => gru + 2 * d * d,
    }
}

/// Build the cell whose parameter count is closest to `budget`.
pub fn build_with_budget(kind: Kind, input: usize, budget: usize, seed: u64) -> Box<dyn Cell> {
    let best = (4..=4096usize)
        .min_by_key(|&d| param_count_for(kind, input, d).abs_diff(budget))
        .expect("non-empty range");
    build(kind, input, best, seed)
}

/// Linearly spaced values, used for multi-timescale gate biases.
pub(crate) fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![a; n];
    }
    (0..n).map(|i| a + (b - a) * i as f32 / (n - 1) as f32).collect()
}
