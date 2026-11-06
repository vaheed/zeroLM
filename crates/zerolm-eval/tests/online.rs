use std::path::Path;
use zerolm_eval::online::*;
use zerolm_learning::Constants;

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn small(p: &Protocol) -> Protocol {
    let mut q = p.clone();
    q.exposure_bytes = 3000;
    q.eval_bytes = 600;
    q.milestones = vec![0, 1000, 3000];
    q.checkpoint_every = 500;
    q
}

#[test]
fn frozen_protocol_is_valid_and_matches_the_code_defaults() {
    let p = Protocol::load(root()).unwrap();
    assert_eq!(p.constants, Constants::default(), "code defaults drifted from model/specs/online-learning.json");
    assert_eq!(p.primary_rule, "delta");
    assert!(p.main_seeds.len() >= 3);
}

#[test]
fn protocol_validation_rejects_unsafe_setups() {
    let p = Protocol::load(root()).unwrap();
    let mut q = p.clone();
    q.main_seeds = vec![p.pilot_seeds[0], 1, 2];
    assert!(q.validate().unwrap_err().contains("disjoint"));
    let mut q = p.clone();
    q.main_seeds = vec![1, 2];
    assert!(q.validate().is_err());
    let mut q = p.clone();
    q.local_rules.push("frozen".into());
    assert!(q.validate().is_err());
    let mut q = p.clone();
    q.baseline_orders = vec![3];
    assert!(q.validate().is_err());
    let mut q = p.clone();
    q.milestones = vec![0, 10];
    assert!(q.validate().is_err());
}

#[test]
fn small_run_is_clean_deterministic_and_logs_every_event() {
    let p = small(&Protocol::load(root()).unwrap());
    let a = run(root(), &p, &[1, 2, 3], false).unwrap();
    assert!(a.failures.is_empty(), "{:?}", a.failures);
    assert!(a.seeds.iter().all(|s| s.contamination.clean() && s.exposure_bytes == 3000));
    for r in &a.runs {
        assert!(r.log_failures.is_empty(), "{} {:?}", r.arm, r.log_failures);
        assert_eq!(r.curve.len(), 3);
        if r.arm == "frozen" {
            assert_eq!((r.update_events, r.initial_sha256 == r.final_sha256), (0, true));
        } else {
            assert_eq!(r.update_events, 3000);
            assert_ne!(r.initial_sha256, r.final_sha256);
            assert!(r.heldout.bpb < r.curve[0].heldout.bpb, "{}: no held-out improvement", r.arm);
        }
    }
    assert_eq!(a.gates.len(), 2 * p.local_rules.len());
    let b = run(root(), &p, &[1, 2, 3], false).unwrap();
    let hashes = |r: &Report| r.runs.iter().map(|x| x.final_sha256.clone()).collect::<Vec<_>>();
    assert_eq!(hashes(&a), hashes(&b), "runs are bit-reproducible");
    let csv = learning_curve_csv(&a);
    assert_eq!(csv.lines().count(), 1 + a.runs.len() * 3);
    assert!(summary_md(&a).contains("Acquisition gate"));
}

#[test]
fn log_verification_catches_tampering() {
    let p = small(&Protocol::load(root()).unwrap());
    let r = run(root(), &p, &[1, 2, 3], false).unwrap();
    let spec = zerolm_learning::streams::StreamSpec::parse(
        &std::fs::read_to_string(root().join(&p.stream_spec)).unwrap(),
    )
    .unwrap();
    let split = spec.split(1, p.exposure_bytes, p.eval_bytes);
    let ex = &split.exposure.as_bytes()[..p.exposure_bytes];
    let delta = r.runs.iter().find(|x| x.seed == 1 && x.arm == "delta").unwrap().clone();
    assert!(verify_log(zerolm_learning::Rule::Delta, ex, &delta).is_empty());

    let mut t = delta.clone();
    t.events[10].target ^= 1;
    assert!(!verify_log(zerolm_learning::Rule::Delta, ex, &t).is_empty(), "wrong target");
    let mut t = delta.clone();
    t.events.pop();
    assert!(!verify_log(zerolm_learning::Rule::Delta, ex, &t).is_empty(), "missing event");
    let mut t = delta.clone();
    t.final_sha256 = t.initial_sha256.clone();
    assert!(!verify_log(zerolm_learning::Rule::Delta, ex, &t).is_empty(), "unchanged weights");
    let mut t = delta.clone();
    t.checkpoint_log.remove(1);
    assert!(!verify_log(zerolm_learning::Rule::Delta, ex, &t).is_empty(), "gap in checkpoints");
    let mut frozen = r.runs.iter().find(|x| x.arm == "frozen").unwrap().clone();
    frozen.final_sha256 = "0".repeat(64);
    assert!(!verify_log(zerolm_learning::Rule::Frozen, ex, &frozen).is_empty(), "frozen changed");
}
