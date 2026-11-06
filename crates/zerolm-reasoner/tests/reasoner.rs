use zerolm_engine::semantic_state::{parse, Utterance};
use zerolm_reasoner::lookup::Lookup;
use zerolm_reasoner::oracle::Oracle;
use zerolm_reasoner::{Ops, Pathway, Reasoner, ReasonerConfig, Verdict};

const WORLD: &str = "\
every dog is a mammal
all mammals are animals
rex is a dog
it is in kennel
kennel is in yard
yard is inside farm
birds can fly
pingu is a bird
pingu cannot fly
tweety is a bird
no fish is a mammal
nemo is a fish
tom is taller than ann
ann is taller than bob
cid is shorter than bob
storm happened before flood
harvest came after flood
tom has 3 apples
ann has 2 more apples than tom
bob has 4 fewer apples than ann
if something is wet then it is cold
if something is cold then it is sleepy
rex is wet
zed is a cat
zed is not a cat
";

fn build(ops: Ops, ablate: Option<Pathway>) -> (Reasoner, Oracle) {
    let mut r = Reasoner::new(ReasonerConfig::default()).unwrap();
    r.ops = ops;
    if let Some(p) = ablate {
        r.ablate(p);
    }
    let mut o = Oracle::new(r.cfg.max_depth);
    for line in WORLD.lines() {
        r.observe(line, "test").unwrap();
        o.observe(line);
    }
    (r, o)
}

const QUESTIONS: [(&str, &str, bool); 24] = [
    ("is rex a dog?", "yes", false),
    ("is rex an animal?", "yes", true),
    ("is rex a fish?", "unknown", false),
    ("is nemo a mammal?", "no", true),
    ("is nemo an animal?", "unknown", false),
    ("can tweety fly?", "yes", true),
    ("can pingu fly?", "no", false),
    ("can rex fly?", "unknown", false),
    ("is rex in farm?", "yes", true),
    ("is farm in rex?", "unknown", false),
    ("is tom taller than cid?", "yes", true),
    ("is cid taller than tom?", "no", true),
    ("is bob shorter than tom?", "yes", true),
    ("who is shorter, tom or cid?", "cid", true),
    ("did storm happen before harvest?", "yes", true),
    ("did harvest happen before storm?", "no", true),
    ("how many apples does tom have?", "3", false),
    ("how many apples does bob have?", "1", true),
    ("how many apples does cid have?", "unknown", false),
    ("is rex sleepy?", "yes", true),
    ("is pingu sleepy?", "unknown", false),
    ("is zed a cat?", "contradiction", false),
    ("is it a cat?", "contradiction", false),
    ("Rex is an animal, right?", "yes", true),
];

#[test]
fn every_operator_matches_the_oracle_and_the_lookup_baseline_fails_derived_questions() {
    let (r, o) = build(Ops::ALL, None);
    let lookup = Lookup { oracle: &o };
    for (q, want, derived) in QUESTIONS {
        let a = r.ask(q).unwrap_or_else(|| panic!("not a question: {q}"));
        let Some(Utterance::Query(pq)) = parse(q) else { panic!() };
        let l = o.answer(&pq);
        assert_eq!(l.verdict.text(), want, "oracle on {q}");
        assert_eq!(a.verdict.text(), want, "reasoner on {q}: {a:?}");
        assert_eq!(a.derived, derived, "derived flag on {q}: {a:?}");
        assert_eq!(l.derived, derived, "oracle derived flag on {q}");
        if derived {
            assert_ne!(lookup.answer(&pq, o.last_subject()).text(), want, "lookup should not answer derived {q}");
            assert!(a.premises.len() >= 2 || a.operators.iter().any(|x| *x == "comparison" || *x == "temporal_ordering"), "{q}: {a:?}");
        }
    }
}

#[test]
fn provenance_names_the_premises_and_confidence_carries_through_pronouns() {
    let (r, _) = build(Ops::ALL, None);
    let a = r.ask("is rex in farm?").unwrap();
    assert_eq!(a.premises.len(), 3, "{a:?}");
    assert_eq!(a.operators, ["transitivity"]);
    assert!((a.confidence - 0.9).abs() < 1e-6, "rex-in-kennel came from 'it': {a:?}");
    let a = r.ask("how many apples does bob have?").unwrap();
    assert_eq!(a.operators, ["arithmetic", "direct"]);
    assert_eq!(a.premises.len(), 3);
}

#[test]
fn each_operator_ablation_removes_exactly_its_inferences() {
    let cases = [
        ("class_inclusion", "is rex an animal?"),
        ("transitivity", "is rex in farm?"),
        ("attribute_inheritance", "can tweety fly?"),
        ("comparison", "is tom taller than cid?"),
        ("arithmetic", "how many apples does bob have?"),
        ("temporal_ordering", "did storm happen before harvest?"),
        ("negation", "can pingu fly?"),
        ("conditionals", "is rex sleepy?"),
    ];
    let (full, _) = build(Ops::ALL, None);
    for (op, q) in cases {
        let (r, _) = build(Ops::without(op), None);
        assert_ne!(r.ask(q).unwrap().verdict, full.ask(q).unwrap().verdict, "{op} off must change {q}");
        assert_eq!(r.ask("is rex a dog?").unwrap().verdict, Verdict::Yes, "{op} off must not break direct retrieval");
    }
}

#[test]
fn pathway_ablations_break_binding_state_and_retrieval() {
    let (r, _) = build(Ops::ALL, Some(Pathway::Retrieval));
    assert_eq!(r.ask("is rex a dog?").unwrap().verdict, Verdict::Unknown);
    let (r, _) = build(Ops::ALL, Some(Pathway::RecurrentState));
    assert_eq!(r.ask("is rex in farm?").unwrap().verdict, Verdict::Unknown, "'it is in kennel' was never stored");
    let (r, _) = build(Ops::ALL, Some(Pathway::Binding));
    // Without roles, "tom > ann" and "ann > tom" are the same superposition.
    assert_ne!(r.ask("is cid taller than tom?").unwrap().verdict, Verdict::No);
}

#[test]
fn memory_persists_through_a_checkpoint_and_derived_answers_are_not_written_back() {
    let (mut r, _) = build(Ops::ALL, None);
    let n = r.n_slots();
    for (q, _, _) in QUESTIONS {
        r.observe(q, "test").unwrap();
    }
    assert_eq!(r.n_slots(), n, "questions and derivations never write memory");
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("zerolm-reasoner-ckpt");
    let _ = std::fs::remove_dir_all(&dir);
    let sha = r.save(&dir).unwrap();
    assert!(r.events().iter().all(|e| e.checkpoint_sha256.as_deref() == Some(sha.as_str())));
    let l = Reasoner::load(&dir).unwrap();
    for (q, _, _) in QUESTIONS.iter().filter(|q| !q.0.contains(" it ")) {
        assert_eq!(l.ask(q), r.ask(q), "{q}");
    }
}

#[test]
fn every_tensor_belongs_to_a_declared_weight_group() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/specs/weight-groups.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let ids: Vec<&str> = v["groups"].as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap()).collect();
    for t in Reasoner::new(ReasonerConfig::default()).unwrap().tensors() {
        assert!(ids.contains(&t.group.as_str()), "{} uses undeclared group {}", t.name, t.group);
    }
}

#[test]
fn default_config_compiles_to_the_shipped_pinned_circuits() {
    use zerolm_reasoner::circuits::{graph_spec, reach_spec, GRAPH_ID, REACH_ID};
    let cfg = ReasonerConfig::default();
    for (spec, id) in [(graph_spec(&cfg), GRAPH_ID), (reach_spec(&cfg), REACH_ID)] {
        let built = zerolm_wsp::compile(&spec).unwrap();
        assert_eq!(built.sha256(), zerolm_wsp::shipped(id).unwrap().sha256(), "{id}: ReasonerConfig::default() must compile to model/prog/{id}.json");
    }
    let r = Reasoner::new(cfg).unwrap();
    let prog: Vec<_> = r.tensors().into_iter().filter(|t| t.name.starts_with("prog.")).collect();
    assert!(!prog.is_empty() && prog.iter().all(|t| t.group == "w_prog"));
}
