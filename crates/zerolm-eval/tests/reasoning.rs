use std::path::Path;
use zerolm_engine::semantic_state::{parse, Utterance};
use zerolm_eval::reasoning::{generate, run_seed, Spec, CATEGORIES};
use zerolm_reasoner::ReasonerConfig;

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

#[test]
fn the_frozen_spec_matches_the_runtime_default_and_is_not_pending() {
    let s = Spec::load(root()).unwrap();
    assert_eq!(s.config, ReasonerConfig::default());
    assert!(!s.gate_rationale.starts_with("PENDING"));
}

#[test]
fn worlds_are_deterministic_parseable_and_cover_every_category() {
    let s = Spec::load(root()).unwrap();
    let (a, b) = (generate(&s, 11), generate(&s, 11));
    let texts = |w: &zerolm_eval::reasoning::World| w.statements.iter().map(|x| x.text.clone()).collect::<Vec<_>>();
    assert_eq!(texts(&a), texts(&b));
    assert_ne!(texts(&a), texts(&generate(&s, 12)));
    for st in &a.statements {
        assert!(matches!(parse(&st.text), Some(Utterance::Fact(_) | Utterance::Rule(_))), "unparseable statement: {}", st.text);
        assert!(matches!(parse(&st.canonical), Some(Utterance::Fact(_) | Utterance::Rule(_))), "{}", st.canonical);
    }
    for q in &a.questions {
        assert!(matches!(parse(&q.text), Some(Utterance::Query(_))), "{}", q.text);
        assert!(matches!(parse(&q.alt), Some(Utterance::Query(_))), "{}", q.alt);
    }
    for c in CATEGORIES {
        assert!(a.questions.iter().any(|q| q.category == c), "no {c} questions");
    }
    assert!(a.statements.iter().any(|s| s.via_pronoun));
}

#[test]
fn a_hidden_seed_result_publishes_no_seed_questions_or_labels() {
    let mut s = Spec::load(root()).unwrap();
    s.counterfactual_pairs = 3;
    s.provenance_checks = 3;
    s.per_category = 3;
    let work = Path::new(env!("CARGO_TARGET_TMPDIR")).join("zerolm-reasoning-hidden");
    let r = run_seed(&s, 424242, true, &work).unwrap();
    let json = serde_json::to_string(&r).unwrap();
    assert_eq!(r.seed, 0);
    assert!(!json.contains("424242"));
    let w = generate(&s, 424242);
    for q in &w.questions {
        assert!(!json.contains(&q.text), "question text leaked: {}", q.text);
    }
    assert_eq!(r.agreement, 1.0);
}
