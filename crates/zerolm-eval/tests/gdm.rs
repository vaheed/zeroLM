use zerolm_eval::bptt::{Bptt, Config, Gdm, GdmInit};
use zerolm_tensor::Rng;

fn cfg(g: Gdm) -> Config {
    Config { gdm: Some(g), lr: 0.003, ..Config::default() }
}

fn small(init: GdmInit) -> Gdm {
    Gdm { rank: 8, init, a0: 1.5, b0: 0.3, gamma0: 1.5, ..Gdm::compiled() }
}

fn repeated_random(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng::new(seed);
    let block: Vec<u8> = (0..n).map(|_| (97 + r.below(20)) as u8).collect();
    let mut v = block.clone();
    v.extend(&block);
    v
}

#[test]
fn gdm_gradients_match_finite_differences() {
    let mut m = Bptt::with_budget(5, cfg(small(GdmInit::Compiled)), 4000);
    let mut r = Rng::from_tag(5, "gdm-test");
    let range = m.gdm_param_range();
    for p in m.params_mut_for_test()[range.clone()].iter_mut() {
        *p += r.uniform(0.05);
    }
    let text = b"abcabcabdabcabcxyzabcabcqrsabc";
    let (loss, g) = m.gradient_for_test(text);
    assert!(loss.is_finite());
    let [gv, gd, sc, _wa, _wb] = m.gdm_offsets();
    let mut idx: Vec<usize> = Vec::new();
    for b in text.iter().collect::<std::collections::BTreeSet<_>>() {
        for k in 0..8 {
            idx.push(range.start + *b as usize * 8 + k); // U
            idx.push(gv + *b as usize * 8 + k);
            idx.push(gd + *b as usize * 8 + k);
        }
    }
    idx.extend(sc..range.end);
    let (mut checked, mut worst) = (0, 0.0f64);
    for &i in &idx {
        if g[i].abs() < 1e-4 {
            continue;
        }
        let h = 1e-2f32;
        let orig = m.params_mut_for_test()[i];
        m.params_mut_for_test()[i] = orig + h;
        let lp = m.loss_for_test(text);
        m.params_mut_for_test()[i] = orig - h;
        let lm = m.loss_for_test(text);
        m.params_mut_for_test()[i] = orig;
        let num = (lp - lm) / (2.0 * h as f64);
        let rel = (num - g[i] as f64).abs() / (num.abs().max(g[i].abs() as f64) + 1e-6);
        worst = worst.max(rel);
        checked += 1;
    }
    assert!(checked > 40, "only {checked} memory parameters had a usable gradient");
    assert!(worst < 0.05, "worst relative gradient error {worst}");
}

#[test]
fn memory_off_is_the_plain_gru_with_extra_dead_parameters() {
    let base = Config { lr: 0.003, ..Config::default() };
    let off = cfg(Gdm { off: true, ..Gdm::compiled() });
    let (mut a, mut b) = (Bptt::with_budget(3, base, 4000), Bptt::with_budget(3, off, 4000));
    let text = repeated_random(60, 1);
    let (ba, bb) = (a.observe_all(&text), b.observe_all(&text));
    assert_eq!(ba.to_bits(), bb.to_bits(), "memory-off arm must reproduce the plain GRU bit for bit");
    assert!(b.param_count() > a.param_count());
    assert_eq!(b.effective_param_count(), a.param_count());
    assert!(b.fast_state_len() > 0 && a.fast_state_len() == 0);
}

#[test]
fn compiled_memory_copies_a_repeat_at_step_zero_and_a_random_one_does_not() {
    let text = repeated_random(20, 2);
    let first = |init| {
        let m = Bptt::with_budget(4, cfg(Gdm { init, gamma0: 6.0, a0: 6.0, b0: 2.0, ..Gdm::compiled() }), 4000);
        let bits = m.score(&text);
        let (a, b) = (bits[..20].iter().sum::<f32>(), bits[20..].iter().sum::<f32>());
        (a, b)
    };
    let (ca, cb) = first(GdmInit::Compiled);
    let (ra, rb) = first(GdmInit::Random);
    assert!(cb < ca * 0.7, "compiled memory should make the second copy cheap (capacity is about `rank` associations): {ca} then {cb}");
    assert!(rb > ra * 0.9, "random tables should not: {ra} then {rb}");
}

#[test]
fn memory_state_is_carried_across_chunks_and_trained_weights_change() {
    let mut m = Bptt::with_budget(6, cfg(small(GdmInit::Compiled)), 4000);
    let before = m.weights_sha256();
    m.observe_all(&repeated_random(70, 3));
    assert_ne!(before, m.weights_sha256());
    assert!(m.fast_state_len() == 64);
}


#[test]
fn snapshot_restores_exact_execution_and_detects_corruption() {
    let c = cfg(small(GdmInit::Compiled));
    let text = repeated_random(80, 9);
    let more = repeated_random(50, 10);
    let mut a = Bptt::with_budget(7, c, 4000);
    a.observe_all(&text);
    let snap = a.snapshot();
    // continue the original
    let bits_a = a.observe_all(&more);
    // fresh model of the same genome, restored, continues identically
    let mut b = Bptt::with_budget(7, c, 4000);
    b.restore(&snap).unwrap();
    assert_eq!(b.snapshot(), snap, "restore then snapshot is the identity");
    let bits_b = b.observe_all(&more);
    assert_eq!(bits_a.to_bits(), bits_b.to_bits());
    assert_eq!(a.weights_sha256(), b.weights_sha256());
    assert_eq!(a.snapshot_sha256(), b.snapshot_sha256());
    // corruption and genome mismatch are rejected
    let mut bad = snap.clone();
    bad[200] ^= 1;
    assert!(Bptt::with_budget(7, c, 4000).restore(&bad).is_err());
    let mut other = Bptt::with_budget(7, cfg(small(GdmInit::Random)), 4000);
    assert!(other.restore(&snap).is_err(), "a different init is a different genome");
    assert_ne!(a.genome_sha256(), other.genome_sha256());
    // genome is deterministic
    assert_eq!(a.genome_sha256(), Bptt::with_budget(7, c, 4000).genome_sha256());
}

#[test]
fn score_from_matches_a_continuous_pass_and_changes_nothing() {
    let m = Bptt::with_budget(8, cfg(small(GdmInit::Compiled)), 4000);
    let text = repeated_random(30, 11);
    let all = m.score(&text);
    // score_from a fresh state equals score
    let part = m.score_from(&m.fresh_state(), &text);
    assert_eq!(all, part);
    let before = m.weights_sha256();
    let _ = m.score_from(&m.fresh_state(), &text);
    assert_eq!(before, m.weights_sha256());
}
