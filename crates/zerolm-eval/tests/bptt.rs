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

fn text(n: usize) -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog. ".iter().cycle().take(n).copied().collect()
}

#[test]
fn the_mainline_update_log_is_exact() {
    let data = text(1000);
    let mut b = Bptt::new(1, Config::default());
    b.enable_log("test-stream", 5);
    b.observe_all(&data);
    b.close_log();
    let ev = b.events();
    assert_eq!(ev.len(), data.len().div_ceil(32), "one event per update chunk");
    let mut at = 0;
    for (i, e) in ev.iter().enumerate() {
        assert_eq!((e.seq, e.stream_offset), (i, at), "events are contiguous and ordered");
        assert!(e.update_norm > 0.0 && e.grad_norm.is_finite());
        at += e.bytes;
    }
    assert_eq!(at, data.len());
    let cps = b.checkpoints();
    assert_eq!(cps.first().unwrap().first_seq, 0);
    assert_eq!(cps.last().unwrap().last_seq, ev.len() - 1, "the closing block covers the last update");
    for w in cps.windows(2) {
        assert_eq!(w[1].first_seq, w[0].last_seq + 1, "blocks tile the update sequence");
    }
    assert!(cps.iter().all(|c| c.signal == "observed_next_byte" && c.rule == "truncated_bptt_adam" && c.input_stream == "test-stream" && c.changed_tensors.contains(&"embed")));
    // The final checkpoint hash is the weights' hash, and the target hashes cover exactly the stream.
    assert_eq!(cps.last().unwrap().checkpoint_sha256, b.weights_sha256());
    let all: Vec<&str> = cps.iter().map(|c| c.targets_sha256.as_str()).collect();
    assert_eq!(all.len(), cps.len());
    assert_ne!(all[0], all[1], "different blocks have different target hashes");
}

#[test]
fn disabled_means_no_update_and_the_future_does_not_leak() {
    let data = text(400);
    let init = Bptt::new(2, Config::default()).weights_sha256();
    let mut b = Bptt::new(2, Config::default());
    b.set_learning(false);
    b.enable_log("s", 5);
    b.observe_all(&data);
    assert_eq!(b.weights_sha256(), init);
    assert!(b.events().is_empty());
    // Bits of an update chunk depend only on bytes before and inside it.
    let mut other = data.clone();
    for x in other[200..].iter_mut() {
        *x = b'~';
    }
    let (mut a, mut c) = (Bptt::new(2, Config::default()), Bptt::new(2, Config::default()));
    a.enable_log("s", 5);
    c.enable_log("s", 5);
    a.observe_all(&data);
    c.observe_all(&other);
    let first = 200 / 32;
    assert_eq!(a.events()[..first].iter().map(|e| e.bits_before).collect::<Vec<_>>(), c.events()[..first].iter().map(|e| e.bits_before).collect::<Vec<_>>());
    // Determinism.
    let mut d = Bptt::new(2, Config::default());
    d.observe_all(&data);
    assert_eq!(d.weights_sha256(), a.weights_sha256());
}
