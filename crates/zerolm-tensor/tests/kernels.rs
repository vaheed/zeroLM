//! Kernels vs an independent naive f64 reference on seeded inputs.
use zerolm_tensor::*;

fn matvec_ref(m: &Matrix, x: &[f32]) -> Vec<f64> {
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m.data[r * m.cols + c] as f64 * x[c] as f64).sum())
        .collect()
}

fn vec_of(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.uniform(1.0)).collect()
}

#[test]
fn matvec_matches_f64_reference() {
    for seed in 1..=5u64 {
        let m = Matrix::fan_in(37, 53, seed, "m");
        let x = vec_of(seed + 100, 53);
        let mut out = vec![0.0; 37];
        m.matvec(&x, &mut out);
        let r = matvec_ref(&m, &x);
        let err = out.iter().zip(&r).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
        assert!(err <= 1e-6, "seed {seed}: max_abs_error {err}");
    }
}

#[test]
fn matmul_matches_f64_reference() {
    let a = Matrix::fan_in(11, 17, 3, "a");
    let b = Matrix::fan_in(17, 9, 4, "b");
    let c = a.matmul(&b);
    for i in 0..11 {
        for j in 0..9 {
            let r: f64 = (0..17).map(|k| a.data[i * 17 + k] as f64 * b.data[k * 9 + j] as f64).sum();
            assert!((c.data[i * 9 + j] as f64 - r).abs() <= 1e-6);
        }
    }
}

#[test]
fn softmax_and_logsumexp_match_reference() {
    let v = vec_of(9, 64);
    let mut p = v.clone();
    softmax_inplace(&mut p);
    let z: f64 = v.iter().map(|x| (*x as f64).exp()).sum();
    for (pi, vi) in p.iter().zip(&v) {
        assert!((*pi as f64 - (*vi as f64).exp() / z).abs() <= 1e-6);
    }
    assert!((log_sum_exp(&v) as f64 - z.ln()).abs() <= 1e-5);
}

#[test]
fn softmax_survives_extreme_inputs() {
    let mut v = vec![1000.0, -1000.0, 999.0];
    softmax_inplace(&mut v);
    assert!(all_finite(&v));
    assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn rms_norm_has_unit_rms() {
    let mut v = vec_of(2, 40);
    rms_norm_inplace(&mut v, 1e-8);
    let ms = v.iter().map(|x| x * x).sum::<f32>() / 40.0;
    assert!((ms - 1.0).abs() < 1e-4);
}

#[test]
fn init_is_deterministic_and_seed_sensitive() {
    let a = Matrix::fan_in(8, 8, 42, "w");
    let b = Matrix::fan_in(8, 8, 42, "w");
    let c = Matrix::fan_in(8, 8, 43, "w");
    let d = Matrix::fan_in(8, 8, 42, "other");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, d);
}

#[test]
fn rng_golden_values_are_stable() {
    // Pins the integer stream: changing it breaks reproducibility of all checkpoints.
    let mut r = Rng::new(0);
    assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
}

#[test]
fn shuffle_is_a_deterministic_permutation() {
    let mut a: Vec<u32> = (0..50).collect();
    let mut b = a.clone();
    shuffle(&mut a, 7);
    shuffle(&mut b, 7);
    assert_eq!(a, b);
    let mut s = a.clone();
    s.sort();
    assert_eq!(s, (0..50).collect::<Vec<_>>());
    assert_ne!(a, (0..50).collect::<Vec<_>>());
}

#[test]
fn finite_checks() {
    assert!(all_finite(&[1.0, 2.0]));
    assert!(!all_finite(&[1.0, f32::NAN]));
    assert!(!all_finite(&[f32::INFINITY]));
}
