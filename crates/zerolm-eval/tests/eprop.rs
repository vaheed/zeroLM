use zerolm_eval::bptt::{Bptt, Config};
use zerolm_learning::online_predictor::Constants;
use zerolm_learning::plastic::*;
use zerolm_tensor::Rng;

fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng::new(seed);
    let words: [&[u8]; 6] = [b"the ", b"cat ", b"sat ", b"on ", b"a ", b"mat. "];
    let mut out = Vec::new();
    while out.len() < n {
        out.extend_from_slice(words[r.below(6)]);
    }
    out.truncate(n);
    out
}

fn model(plast: Plasticity) -> PlasticModel {
    PlasticModel::new(Core::from_seed(3), plast, Constants::default(), EpropConfig::default())
}

/// d h_i / d W_ij with h_{t-1} fixed, by central finite differences of the step function.
fn fd(core: &Core, hp: &[f32], prev: u8, idx: usize, unit: usize) -> f64 {
    let eps = 2e-3f32;
    let (mut a, mut b) = (core.clone(), core.clone());
    a.p[idx] += eps;
    b.p[idx] -= eps;
    ((a.step(hp, prev).h[unit] - b.step(hp, prev).h[unit]) as f64) / (2.0 * eps as f64)
}

fn unit_of(core: &Core, idx: usize) -> (usize, usize) {
    let t = (0..10).rev().find(|t| core.range(*t).start <= idx).unwrap();
    let off = idx - core.range(t).start;
    let cols = match t {
        1..=3 => core.e,
        4..=6 => core.d,
        _ => 1,
    };
    (t, off / cols)
}

#[test]
fn core_layout_matches_the_bptt_control() {
    let c = Core::from_seed(5);
    let b = Bptt::new(5, Config::default());
    assert_eq!(b.core(), c);
    let (e, d) = (c.e, c.d);
    assert_eq!(c.p.len(), 256 * e + 3 * d * e + 3 * d * d + 3 * d);
}

#[test]
fn direct_terms_match_finite_differences_for_every_gru_tensor() {
    let core = Core::from_seed(3);
    let mut m = model(Plasticity::Eprop);
    // Warm the state on some bytes so h_{t-1} is nonzero.
    let mut h = vec![0.0f32; core.d];
    for &y in &bytes(20, 1) {
        h = core.step(&h, y).h;
    }
    let s = core.step(&h, b'c');
    let direct = PlasticModel::direct_terms(&core, &s);
    let base = 256 * core.e;
    let mut rng = Rng::new(9);
    let mut checked = [0usize; 10];
    for n in 0..900 {
        // Ten samples of each tensor first, then random ones.
        let idx = if n < 90 {
            let r = core.range(1 + n % 9);
            r.start + rng.below(r.len())
        } else {
            base + rng.below(core.p.len() - base)
        };
        let (t, unit) = unit_of(&core, idx);
        let want = fd(&core, &h, b'c', idx, unit);
        let got = direct[idx - base] as f64;
        assert!((want - got).abs() < 2e-3 + 0.02 * want.abs(), "tensor {t} idx {idx}: finite difference {want}, direct {got}");
        checked[t] += 1;
    }
    assert!(checked[1..].iter().all(|c| *c > 0), "every GRU tensor sampled: {checked:?}");
    let _ = &mut m;
}

#[test]
fn traces_follow_the_stated_recursion() {
    let core = Core::from_seed(3);
    let mut m = PlasticModel::new(core.clone(), Plasticity::Frozen, Constants::default(), EpropConfig::default());
    // Direct evaluation per weight: eps_t = (1 - z_t) eps_{t-1} + d h_t/dW by finite differences.
    let idxs: Vec<usize> = {
        let mut r = Rng::new(4);
        (0..12).map(|_| 256 * core.e + r.below(core.p.len() - 256 * core.e)).collect()
    };
    let mut eps = vec![0.0f64; idxs.len()];
    let mut h = vec![0.0f32; core.d];
    let mut prev = START_BYTE;
    let mut trace = m.traces().to_vec();
    for &y in &bytes(40, 2) {
        let s = core.step(&h, prev);
        PlasticModel::advance_traces(&mut trace, &core, &s);
        for (k, &idx) in idxs.iter().enumerate() {
            let (_, unit) = unit_of(&core, idx);
            eps[k] = (1.0 - s.z[unit] as f64) * eps[k] + fd(&core, &h, prev, idx, unit);
            let got = trace[idx - 256 * core.e] as f64;
            assert!((got - eps[k]).abs() < 5e-3 + 0.03 * eps[k].abs(), "idx {idx}: trace {got}, recursion {}", eps[k]);
        }
        h = s.h;
        prev = y;
    }
    let _ = &mut m;
}

#[test]
fn eprop_direction_aligns_with_exact_gradients_and_is_reported() {
    // Exact gradient of the loss by BPTT on the same core and the same nonzero readout.
    let mut b = Bptt::new(3, Config::default());
    let mut r = Rng::new(11);
    let d = b.core().d;
    let ro_start = b.core().p.len();
    for w in b.params_mut_for_test()[ro_start..].iter_mut() {
        *w = r.uniform(0.3);
    }
    let data = bytes(48, 6);
    let (_, g) = b.gradient_for_test(&data);
    let (rw, rb) = b.readout();
    let mut m = model(Plasticity::Frozen);
    m.core = b.core();
    m.readout.w = rw.to_vec();
    m.readout.b = rb.to_vec();
    // Feedback = transpose of the readout's hidden columns (symmetric feedback): the best case.
    let f = d + m.core.e;
    let mut sym = vec![0.0f32; d * 256];
    for k in 0..256 {
        for i in 0..d {
            sym[i * 256 + k] = m.readout.w[k * f + i];
        }
    }
    let base = 256 * m.core.e;
    let cos = |a: &[f32], b: &[f32]| {
        let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            ab += (*x as f64) * (*y as f64);
            aa += (*x as f64).powi(2);
            bb += (*y as f64).powi(2);
        }
        ab / (aa.sqrt() * bb.sqrt() + 1e-30)
    };
    // The BPTT gradient is of the mean loss (descent direction -g); e-prop ascends the log-likelihood.
    let neg_g: Vec<f32> = g[base..base + (m.core.p.len() - base)].iter().map(|x| -x).collect();
    let c_sym = cos(&m.eprop_direction(&data, &sym), &neg_g);
    let c_rand = cos(&m.eprop_direction(&data, m.feedback_matrix()), &neg_g);
    println!("cosine with exact gradient: symmetric feedback {c_sym:.3}, random feedback {c_rand:.3}");
    assert!(c_sym > 0.1, "diagonal approximation should keep some alignment, got {c_sym}");
}

#[test]
fn updates_are_deterministic_and_disabled_means_no_update() {
    let data = bytes(400, 7);
    let run = |plast| {
        let mut m = model(plast);
        m.observe_all(&data).unwrap();
        (m.core.p.clone(), m.readout.w.clone())
    };
    assert_eq!(run(Plasticity::Eprop), run(Plasticity::Eprop));
    let init = Core::from_seed(3);
    let (c0, r0) = run(Plasticity::Frozen);
    assert_eq!(c0, init.p);
    assert!(r0.iter().all(|w| *w == 0.0));
    let (c1, r1) = run(Plasticity::ReadoutOnly);
    assert_eq!(c1, init.p, "readout-only arm must not touch the core");
    assert!(r1.iter().any(|w| *w != 0.0));
    let (c2, _) = run(Plasticity::Eprop);
    assert_ne!(c2, init.p);
    // set_learning(false) on an e-prop model changes nothing.
    let mut m = model(Plasticity::Eprop);
    m.set_learning(false);
    m.observe_all(&data).unwrap();
    assert_eq!(m.core.p, init.p);
}

#[test]
fn no_future_leak_and_the_core_update_is_bounded() {
    let data = bytes(300, 8);
    let mut a = model(Plasticity::Eprop);
    let mut b = model(Plasticity::Eprop);
    let mut other = data.clone();
    for x in other[200..].iter_mut() {
        *x = b'z';
    }
    let (mut bits_a, mut bits_b) = (Vec::new(), Vec::new());
    for i in 0..300 {
        bits_a.push(a.observe(data[i]).unwrap());
        bits_b.push(b.observe(other[i]).unwrap());
    }
    // Bits for byte i depend only on bytes before i (and the byte itself as the target): identical until the streams differ.
    assert_eq!(bits_a[..200], bits_b[..200]);
    let mut m = model(Plasticity::Eprop);
    m.log_updates = true;
    m.observe_all(&data).unwrap();
    assert_eq!(m.log.len(), 300);
    assert!(m.log.iter().all(|u| u.core_update_norm <= m.cfg.max_core_update + 1e-6));
}

#[test]
fn perturbation_has_the_same_per_tensor_norm_as_the_change() {
    let init = Core::from_seed(3);
    let mut m = model(Plasticity::Eprop);
    m.observe_all(&bytes(500, 9)).unwrap();
    let norms = m.core.change_norms(&init);
    assert!(norms.iter().any(|n| *n > 0.0));
    let pert = init.perturbed(&norms, 123);
    let got = pert.change_norms(&init);
    for (a, b) in norms.iter().zip(&got) {
        assert!((a - b).abs() < 1e-4 * (1.0 + a), "{a} vs {b}");
    }
    assert_ne!(pert.p, m.core.p);
}
