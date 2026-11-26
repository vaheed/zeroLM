use std::path::Path;
use zerolm_eval::levels::{realtext_summary, Levels};
use zerolm_eval::ngram::{Kn, Wb};

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn text() -> Vec<u8> {
    b"the cat sat on the mat. the dog sat on the log. ".repeat(40)
}

#[test]
fn ngram_distributions_sum_to_one() {
    let t = text();
    let mut wb = Wb::new(5);
    let mut kn = Kn::new(5);
    for i in 0..t.len() {
        let h = &t[i.saturating_sub(4)..i];
        wb.observe(h, t[i]);
        kn.observe(h, t[i]);
    }
    let h = b"the c";
    let sums = [
        (0..=255u8).map(|b| wb.prob(h, b)).sum::<f64>(),
        (0..=255u8).map(|b| wb.ppm_prob(h, b)).sum::<f64>(),
        (0..=255u8).map(|b| kn.prob(h, b)).sum::<f64>(),
    ];
    for s in sums {
        assert!((s - 1.0).abs() < 1e-6, "distribution sums to {s}");
    }
}

#[test]
fn ngram_models_learn_repetition_and_do_not_see_the_future() {
    let t = text();
    let (train, test) = t.split_at(1200);
    let mut wb = Wb::new(5);
    for i in 0..train.len() {
        wb.observe(&train[i.saturating_sub(4)..i], train[i]);
    }
    let seen = wb.bpb(test, false);
    let fresh = Wb::new(5).bpb(test, false);
    assert!(seen < 1.0 && fresh > 7.9, "seen {seen}, untrained {fresh}");
    // Scoring never updates the model: scoring twice gives the same number.
    assert_eq!(wb.bpb(test, false), seen);
}

#[test]
fn levels_spec_is_valid_and_rejects_inversions() {
    let l = Levels::load(root()).unwrap();
    let mut bad = l.0.clone();
    bad["levels"]["L3"]["blimp_min"] = serde_json::json!(0.7);
    assert!(Levels(bad).validate().is_err());
    let mut bad = l.0.clone();
    bad["levels"]["L3"]["wikipedia_heldout_bpb_max"] = serde_json::json!(1.8);
    assert!(Levels(bad).validate().is_err());
}

#[test]
fn l1_bar_is_the_better_reference_compressor_at_equal_exposure() {
    let rows = vec![
        ("ppm-c-5".to_string(), 1000, 2.5),
        ("kneser-ney-5".to_string(), 1000, 2.4),
        ("witten-bell-5".to_string(), 1000, 2.0),
        ("kneser-ney-5".to_string(), 2000, 2.3),
    ];
    assert_eq!(Levels::l1_bar(&rows, 1000), Some(2.4));
    assert_eq!(Levels::l1_met(&rows, 1000, 2.41), Some(false));
    assert_eq!(Levels::l1_met(&rows, 1000, 2.4), Some(true));
    assert_eq!(Levels::l1_bar(&rows, 5), None);
}

#[test]
fn summary_has_every_field_the_schema_requires() {
    let schema: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root().join("research/schemas/real-text-report.schema.json")).unwrap()).unwrap();
    let h = "0".repeat(64);
    let s = realtext_summary(&[("ppm-c-5".into(), 10, 3.0)], &h, &h, 5, &h, &Levels::spec_sha256(), vec![], false);
    for k in schema["required"].as_array().unwrap() {
        assert!(s.get(k.as_str().unwrap()).is_some(), "missing {k}");
    }
    for k in schema["properties"]["rows"]["items"]["required"].as_array().unwrap() {
        assert!(s["rows"][0].get(k.as_str().unwrap()).is_some(), "row missing {k}");
    }
    assert_eq!(s["levels_sha256"].as_str().unwrap().len(), 64);
}
