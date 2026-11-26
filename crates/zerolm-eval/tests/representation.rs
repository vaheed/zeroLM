use std::path::Path;
use zerolm_eval::representation::*;
use zerolm_learning::plastic::Core;

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

#[test]
fn the_spec_loads_and_pilot_and_main_seeds_are_disjoint() {
    let s = Spec::load(root()).unwrap();
    assert!(s.main_seeds.len() >= 3);
    let mut bad = s.clone();
    bad.main_seeds[0] = bad.pilot_seeds[0];
    assert!(bad.validate().unwrap_err().contains("disjoint"));
    // An unfrozen spec may not carry chosen constants.
    let mut bad = s.clone();
    bad.eprop = Some(EpropChoice { eta: 0.01, max_core_update: 0.05 });
    assert!(bad.validate().is_err());
    // A frozen spec needs everything.
    let mut bad = s.clone();
    bad.frozen = true;
    assert!(bad.validate().is_err());
}

fn thresholds() -> Thresholds {
    Thresholds { h: 0.1, r: 0.05, rho_min: 0.3, kappa_min: 0.5, s: 0.2, max_shared_window_fraction: 0.01 }
}

fn m(local: f64, bptt: f64, pert: f64) -> SeedMeasures {
    SeedMeasures { delta_r_local: local, delta_r_bptt: bptt, delta_r_perturbed: pert, novel_gain_local: 0.05, kappa_local: 0.8, finite: true, late_degradation_local: 0.05 }
}

fn seeds(local: f64, bptt: f64, pert: f64) -> Vec<SeedMeasures> {
    [0.0, 0.01, -0.01].iter().map(|j| m(local + j, bptt + j, pert + j)).collect()
}

#[test]
fn decision_rule_pass_fail_and_inconclusive() {
    let t = thresholds();
    let (v, c) = decide(&seeds(0.4, 0.8, 0.0), &t);
    assert_eq!(outcome(&v, &c, false).0, "pass");
    // Local learning gains nothing while backpropagation does: fail / pivot on P1 and P2.
    let (v, c) = decide(&seeds(0.0, 0.8, 0.0), &t);
    let (o, ids) = outcome(&v, &c, false);
    assert_eq!(o, "fail_pivot");
    assert!(ids.contains(&"P1") && ids.contains(&"P2"));
    // No backpropagation headroom: inconclusive, unless this is the largest affordable setup.
    let (v, c) = decide(&seeds(0.4, 0.01, 0.0), &t);
    assert_eq!(outcome(&v, &c, false), ("inconclusive", vec!["V2"]));
    assert_eq!(outcome(&v, &c, true).0, "fail_pivot");
    // The perturbed control gains as much as A2: the probe is not valid.
    let (v, c) = decide(&seeds(0.4, 0.8, 0.4), &t);
    assert_eq!(outcome(&v, &c, false), ("inconclusive", vec!["V3"]));
    // Not finite or unstable: P5.
    let mut s = seeds(0.4, 0.8, 0.0);
    s[1].finite = false;
    let (v, c) = decide(&s, &t);
    assert!(outcome(&v, &c, false).1.contains(&"P5"));
}

#[test]
fn significance_needs_mean_minus_two_sd_above_the_bound() {
    assert!(significant_above(&[0.5, 0.52, 0.48], 0.0));
    assert!(!significant_above(&[0.5, -0.4, 0.45], 0.0));
    assert!(!significant_above(&[0.1, 0.1, 0.1], 0.1));
}

#[test]
fn milestones_novel_positions_ece_and_degradation() {
    assert_eq!(milestone_list(&[100, 1000, 10_000], 500, 2000), vec![100, 500, 1000, 2000]);
    let exposure = b"abcdefghij".repeat(20);
    let idx = kgram_index(&exposure, 4);
    let held = b"abcdefXXXXghij";
    let mask = novel_positions(held, &idx, 4);
    assert!(!mask[5] && mask[10], "seen context is not novel, a context with XXXX is");
    assert!(!mask[0], "positions without a full context are excluded");
    // ECE: confident and always right is 0; confident and always wrong is large.
    assert!(ece(&vec![[0.0, 1.0, 1.0]; 100]) < 1e-9);
    assert!((ece(&vec![[0.0, 0.9, 0.0]; 100]) - 0.9).abs() < 1e-6);
    assert!((late_degradation(&[1.0, 0.8, 0.9, 1.1, 0.85, 0.95]) - 0.3).abs() < 1e-12);
}

#[test]
fn bptt_core_export_equals_the_shared_initialization() {
    // A3 and A2 start from the same core at every budget.
    for budget in [4000, 20_000] {
        let b = zerolm_eval::bptt::Bptt::with_budget(5, zerolm_eval::bptt::Config::default(), budget);
        assert_eq!(b.core(), Core::from_seed_budget(5, budget));
    }
}
