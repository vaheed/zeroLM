//! Arms G and H of the novelty design (`docs/experimental-design.md`), the bars that any ZeroLM-specific
//! claim must beat.
//!
//! * G: the online-trained GRU (`bptt`) mixed with Kneser-Ney 5 by an online mixture of experts.
//! * H: G plus the fixed-code match model of `headroom` (the PAQ-style hybrid), applied on top of the
//!   mixture with the v2 confidence rule (conditioned on the mixture's own probability of the predicted byte).
//!
//! Mixture: `p(b) = sum_i w_i p_i(b)`, `w = softmax(theta[ctx])`, where `ctx` quantizes the GRU's maximum
//! probability into 4 levels. After each byte is scored, `theta` takes one gradient step on `-ln p(b)`:
//! `d(-ln p)/d theta_j = -w_j (p_j(b) - p(b)) / p(b)`. The mixer has 8 numbers; it adapts on the held-out
//! stream prequentially (scored before updated), as the match confidences do. The GRU and the count model
//! are frozen on held-out text. Research code; the compiled counterpart is future work.

pub const CTX: usize = 4;
pub const EXPERTS: usize = 2;

pub struct Mixer {
    theta: [[f64; EXPERTS]; CTX],
    lr: f64,
}

pub fn ctx_of(max_prob: f32) -> usize {
    if max_prob < 0.3 { 0 } else if max_prob < 0.6 { 1 } else if max_prob < 0.85 { 2 } else { 3 }
}

impl Mixer {
    pub fn new(lr: f64) -> Self {
        Mixer { theta: [[0.0; EXPERTS]; CTX], lr }
    }

    pub fn weights(&self, ctx: usize) -> [f64; EXPERTS] {
        let t = &self.theta[ctx];
        let m = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = t.iter().map(|v| (v - m).exp()).collect();
        let z: f64 = e.iter().sum();
        let mut w = [0.0; EXPERTS];
        for i in 0..EXPERTS {
            w[i] = e[i] / z;
        }
        w
    }

    pub fn mix(&self, ctx: usize, ps: &[f64; EXPERTS]) -> f64 {
        let w = self.weights(ctx);
        w.iter().zip(ps).map(|(a, b)| a * b).sum()
    }

    /// One gradient step on `-ln p(b)` given the experts' probabilities of the byte that occurred.
    pub fn update(&mut self, ctx: usize, ps: &[f64; EXPERTS]) {
        let w = self.weights(ctx);
        let p = self.mix(ctx, ps).max(1e-12);
        for j in 0..EXPERTS {
            self.theta[ctx][j] += self.lr * w[j] * (ps[j] - p) / p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_experts_give_the_expert() {
        let m = Mixer::new(0.1);
        assert!((m.mix(2, &[0.3, 0.3]) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn mixture_moves_toward_the_better_expert_and_never_worse_than_the_worse() {
        let mut m = Mixer::new(0.2);
        for _ in 0..200 {
            m.update(1, &[0.6, 0.1]);
        }
        let w = m.weights(1);
        assert!(w[0] > 0.9, "{w:?}");
        let p = m.mix(1, &[0.6, 0.1]);
        assert!(p > 0.1 && p <= 0.6);
        // other contexts are untouched
        assert!((m.weights(0)[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let ps = [0.7, 0.05];
        let mut m = Mixer::new(1e-3);
        m.theta[0] = [0.3, -0.2];
        let loss = |m: &Mixer| -m.mix(0, &ps).ln();
        let base = loss(&m);
        let mut a = Mixer::new(1e-3);
        a.theta = m.theta;
        a.update(0, &ps);
        let step = (a.theta[0][0] - m.theta[0][0]) / 1e-3; // = -grad_0
        let mut b = Mixer::new(1e-3);
        b.theta = m.theta;
        b.theta[0][0] += 1e-6;
        let fd = (loss(&b) - base) / 1e-6;
        assert!((step + fd).abs() < 1e-4, "{step} {fd}");
    }
}
