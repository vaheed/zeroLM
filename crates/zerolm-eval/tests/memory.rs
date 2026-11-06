use std::path::Path;
use zerolm_eval::memory::{kind_of, score, FactSet, QueryRecord, Spec, PARAPHRASES};
use zerolm_memory::{MemoryConfig, MemoryStore};

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

#[test]
fn the_frozen_spec_validates_and_matches_the_runtime_default() {
    let s = Spec::load(root()).unwrap();
    assert!(s.facts >= 1000 && s.main_seeds.len() >= 3);
    assert_eq!(s.config, MemoryConfig::default(), "the CLI must run the configuration the benchmark gated");
    assert!(!s.gate_rationale.starts_with("PENDING"));
}

#[test]
fn facts_are_deterministic_distinct_and_paraphrases_differ_from_keys() {
    let a = FactSet::generate(5, 1200, 50);
    let b = FactSet::generate(5, 1200, 50);
    assert_eq!(a.facts, b.facts);
    let keys: std::collections::BTreeSet<_> = a.facts.iter().map(|f| f.key()).collect();
    assert_eq!(keys.len(), 1200);
    assert!(a.unseen_entities.iter().all(|e| a.facts.iter().all(|f| &f.entity != e)));
    let mut rng = zerolm_tensor::Rng::new(1);
    for (cat, trivial) in PARAPHRASES {
        let p = a.paraphrase(cat, &a.facts[0], &mut rng);
        assert_ne!(p, a.facts[0].key(), "{cat}");
        assert_eq!(kind_of(cat), if trivial { "paraphrase_trivial" } else { "paraphrase" });
    }
    assert_ne!(FactSet::generate(6, 10, 1).facts, a.facts[..10].to_vec());
}

#[test]
fn scoring_separates_hits_wrong_answers_abstentions_and_false_accepts() {
    let r = |cov: f32, decoded: &str, expected: Option<&str>| QueryRecord {
        category: "x",
        query: String::new(),
        score: 0.9,
        coverage: cov,
        decoded: decoded.into(),
        accepted: cov >= 0.5,
        pooled_ok: true,
        expected: expected.map(str::to_string),
    };
    let recs = [r(1.0, "a", Some("a")), r(1.0, "b", Some("a")), r(0.1, "a", Some("a")), r(0.9, "z", None), r(0.2, "z", None)];
    let s = score(&recs.iter().collect::<Vec<_>>(), 0.5);
    assert_eq!(s.queries, 5);
    assert!((s.recall - 1.0 / 3.0).abs() < 1e-9);
    assert!((s.precision - 0.5).abs() < 1e-9);
    assert!((s.false_accept_rate - 0.5).abs() < 1e-9);
    assert!((s.abstain_rate - 0.4).abs() < 1e-9);
}

#[test]
fn the_default_store_answers_a_generated_fact_after_a_restart() {
    let set = FactSet::generate(1, 30, 5);
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    for f in &set.facts {
        m.teach(&f.key(), &f.value, "test").unwrap();
    }
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("zerolm-eval-memory");
    let _ = std::fs::remove_dir_all(&d);
    m.save(&d).unwrap();
    let (m, _) = MemoryStore::load(&d).unwrap();
    for f in &set.facts {
        assert_eq!(m.query(&f.key()).unwrap().value.as_deref(), Some(f.value.as_str()));
    }
}
