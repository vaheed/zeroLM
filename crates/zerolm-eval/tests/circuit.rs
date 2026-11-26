use zerolm_eval::bptt::{Bptt, Circuit, CircuitInit, Config};
use zerolm_tensor::Rng;

fn cfg(c: Circuit) -> Config {
    Config { circuit: Some(c), lr: 0.003, ..Config::default() }
}

fn repeated_random(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng::new(seed);
    let block: Vec<u8> = (0..n).map(|_| (97 + r.below(20)) as u8).collect();
    let mut v = block.clone();
    v.extend(&block);
    v
}

#[test]
fn circuit_gradients_match_finite_differences() {
    let c = Circuit { beta: 2.0, s0: 1.0, gamma0: 1.5, window: 24, ..Circuit::compiled() };
    let mut m = Bptt::with_budget(5, cfg(c), 4000);
    let mut r = Rng::from_tag(5, "circuit-test");
    let range = m.circuit_param_range();
    for p in m.params_mut_for_test()[range.clone()].iter_mut() {
        *p += r.uniform(0.05);
    }
    let text = b"abcabcabdabcabcxyzabcabc";
    let (loss, g) = m.gradient_for_test(text);
    assert!(loss.is_finite());
    // rows of the bytes that occur, in U and V, then the scalars and w
    let mut idx: Vec<usize> = Vec::new();
    for b in text.iter().collect::<std::collections::BTreeSet<_>>() {
        for k in 0..16 {
            idx.push(range.start + *b as usize * 16 + k);
            idx.push(range.start + 4096 + *b as usize * 16 + k);
        }
    }
    idx.extend(range.start + 8192..range.end);
    let (mut checked, mut worst) = (0, 0.0f64);
    for &i in &idx {
        if g[i].abs() < 1e-3 {
            continue;
        }
        let eps = 2e-2f32;
        let old = m.params_mut_for_test()[i];
        m.params_mut_for_test()[i] = old + eps;
        let up = m.loss_for_test(text);
        m.params_mut_for_test()[i] = old - eps;
        let down = m.loss_for_test(text);
        m.params_mut_for_test()[i] = old;
        let numeric = (up - down) / (2.0 * eps as f64);
        let rel = (numeric - g[i] as f64).abs() / (g[i].abs() as f64).max(1e-3);
        if rel > 0.05 {
            eprintln!("idx {} (offset {}) analytic {} numeric {}", i, i - range.start, g[i], numeric);
        }
        worst = worst.max(rel);
        checked += 1;
    }
    assert!(checked > 25, "only {checked} circuit parameters had a usable gradient");
    assert!(worst < 0.05, "worst relative gradient error {worst}");
}

#[test]
fn compiled_circuit_copies_a_repeat_at_step_zero_and_a_random_one_does_not() {
    let text = repeated_random(200, 9);
    let half = &text[200..];
    let mut bits = Vec::new();
    for init in [CircuitInit::Compiled, CircuitInit::Random] {
        let m = Bptt::with_budget(1, cfg(Circuit { init, ..Circuit::compiled() }), 4000);
        let s = m.score(&text);
        bits.push(s[200..].iter().sum::<f32>() / half.len() as f32 - s[..200].iter().sum::<f32>() / 200.0);
    }
    // second copy is cheaper than the first under the compiled circuit, by a lot; not under random tables
    assert!(bits[0] < -2.0, "compiled: {}", bits[0]);
    assert!(bits[1] > -0.5, "random: {}", bits[1]);
}

#[test]
fn frozen_circuit_keeps_its_tables_and_learns_the_rest() {
    let text = repeated_random(200, 3);
    let mut m = Bptt::with_budget(1, cfg(Circuit { frozen: true, ..Circuit::compiled() }), 4000);
    let before = m.circuit_param_range();
    let snap: Vec<f32> = m.params_mut_for_test()[before.clone()].to_vec();
    m.observe_all(&text);
    let d = m.params_mut_for_test()[before.clone()].to_vec();
    // U, V, beta, s0, gamma0 are the first 2*256*16+3 entries
    let fixed = 2 * 256 * 16 + 3;
    assert_eq!(snap[..fixed], d[..fixed]);
    assert_ne!(snap[fixed..], d[fixed..], "the hidden-state gate w should have learned");
}

#[test]
fn no_circuit_means_no_extra_parameters() {
    let a = Bptt::with_budget(1, Config::default(), 4000);
    assert!(a.circuit_param_range().is_empty());
    let b = Bptt::with_budget(1, cfg(Circuit::compiled()), 4000);
    assert!(b.param_count() > a.param_count() + 8192);
}
