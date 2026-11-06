use crate::gru::Gru;
use crate::{rec, Cell, Kind, Trace};
use zerolm_tensor::{dot, shuffle, softmax_inplace, Matrix};

/// Window length of the local attention (fixed, so state stays bounded).
pub const WINDOW: usize = 8;

/// Comparison arm: GRU plus attention over the last `WINDOW` hidden states.
/// State layout: [h (d) | ring (WINDOW*d) | next_slot | filled].
pub struct Hybrid {
    d: usize,
    gru: Gru,
    wq: Matrix,
    wo: Matrix,
}

impl Hybrid {
    pub fn new(e: usize, d: usize, seed: u64) -> Self {
        Hybrid {
            d,
            gru: Gru::new(e, d, seed),
            wq: Matrix::fan_in(d, d, seed, "hy.wq"),
            wo: Matrix::fan_in(d, d, seed, "hy.wo"),
        }
    }
}

impl Cell for Hybrid {
    fn kind(&self) -> Kind {
        Kind::Hybrid
    }
    fn input_dim(&self) -> usize {
        self.gru.input_dim()
    }
    fn hidden_dim(&self) -> usize {
        self.d
    }
    fn param_count(&self) -> usize {
        self.gru.count() + 2 * self.d * self.d
    }
    fn state_len(&self) -> usize {
        self.d + WINDOW * self.d + 2
    }
    fn init_state(&self) -> Vec<f32> {
        vec![0.0; self.state_len()]
    }
    fn step(&self, state: &mut [f32], x: &[f32], mut trace: Option<&mut dyn Trace>) {
        let d = self.d;
        let tail = state.len() - 2;
        let slot = state[tail] as usize;
        let filled = state[tail + 1] as usize;
        let hp = state[..d].to_vec();
        let mut g = hp.clone();
        self.gru.update(&mut g, x, &mut trace);
        // local attention over the ring of previous hidden states
        let mut q = vec![0.0; d];
        self.wq.matvec(&hp, &mut q);
        let mut ctx = vec![0.0; d];
        if filled > 0 {
            let scale = 1.0 / (d as f32).sqrt();
            let mut scores: Vec<f32> =
                (0..filled).map(|j| dot(&q, &state[d + j * d..d + (j + 1) * d]) * scale).collect();
            softmax_inplace(&mut scores);
            for (j, s) in scores.iter().enumerate() {
                for i in 0..d {
                    ctx[i] += s * state[d + j * d + i];
                }
            }
            rec(&mut trace, "hy.attention", &scores);
        }
        let mut out = g;
        self.wo.matvec_acc(&ctx, &mut out);
        out.iter_mut().for_each(|v| *v = v.tanh());
        // push the previous state into the ring, then publish the new output
        state[d + slot * d..d + (slot + 1) * d].copy_from_slice(&hp);
        state[tail] = ((slot + 1) % WINDOW) as f32;
        state[tail + 1] = (filled + 1).min(WINDOW) as f32;
        state[..d].copy_from_slice(&out);
    }
    fn groups(&self) -> Vec<(&'static str, &[f32])> {
        let mut g = self.gru.groups_vec();
        g.push(("hy.wq", &self.wq.data));
        g.push(("hy.wo", &self.wo.data));
        g
    }
    fn groups_mut(&mut self) -> Vec<(&'static str, &mut [f32])> {
        let mut g = self.gru.groups_mut_vec();
        g.push(("hy.wq", &mut self.wq.data));
        g.push(("hy.wo", &mut self.wo.data));
        g
    }
    fn sever_recurrence(&mut self) {
        self.gru.sever();
        self.wo.data.fill(0.0);
    }
    fn shuffle_recurrence(&mut self, seed: u64) {
        self.gru.shuffle_rec(seed);
        shuffle(&mut self.wo.data, seed.wrapping_add(2));
    }
    fn trace_points(&self) -> usize {
        4
    }
}
