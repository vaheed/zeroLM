use zerolm_eval::bptt::{Bptt, Config};
use zerolm_tensor::Rng;

fn randomized(seed: u64) -> Bptt {
    let mut m = Bptt::new(seed, Config::default());
    let mut r = Rng::from_tag(seed, "bptt-test");
    for p in m.params_mut_for_test().iter_mut() {
        *p += r.uniform(0.2);
    }
    m
}

#[test]
fn analytic_gradients_match_finite_differences_in_every_parameter_group() {
    let mut m = randomized(5);
    let text = b"the cat sat on the mat. the cat ran.";
    let (loss, g) = m.gradient_for_test(text);
    assert!(loss.is_finite() && loss > 0.0);
    let n = g.len();
    let mut checked = 0;
    let mut worst = 0.0f64;
    // 400 spread-out parameters, skipping ones with a negligible gradient.
    for k in 0..400 {
        let i = (k * 7919 + 13) % n;
        if g[i].abs() < 1e-4 {
            continue;
        }
        let eps = 2e-3f32;
        let old = m.params_mut_for_test()[i];
        m.params_mut_for_test()[i] = old + eps;
        let up = m.loss_for_test(text);
        m.params_mut_for_test()[i] = old - eps;
        let down = m.loss_for_test(text);
        m.params_mut_for_test()[i] = old;
        let numeric = (up - down) / (2.0 * eps as f64);
        let rel = (numeric - g[i] as f64).abs() / (g[i].abs() as f64).max(1e-3);
        worst = worst.max(rel);
        checked += 1;
    }
    assert!(checked > 100, "only {checked} parameters had a usable gradient");
    assert!(worst < 0.05, "worst relative gradient error {worst}");
}

#[test]
fn training_changes_weights_and_learns_a_repeating_text() {
    let mut m = Bptt::new(1, Config::default());
    let before = m.evaluate(&b"abcabcabcabcabcabcabcabc".repeat(4));
    let stream = b"abcabcabcabcabcabcabcabc".repeat(400);
    let online = m.observe_all(&stream);
    let after = m.evaluate(&b"abcabcabcabcabcabcabcabc".repeat(4));
    assert!(before > 7.5, "untrained readout is uniform: {before}");
    assert!(after < 0.5, "should learn the repeat: {after}");
    assert!(online < before && online.is_finite());
}
