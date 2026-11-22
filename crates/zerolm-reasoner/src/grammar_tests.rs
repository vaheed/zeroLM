use super::*;
use crate::ReasonerConfig;
use zerolm_engine::semantic_state::{normalize_tokens, parse, tokens};

#[test]
fn compiled_rewrites_match_reference_including_order_and_boundaries() {
    let r = Reasoner::new(ReasonerConfig { dim: 64, symbols: 32, slots: 32, number_range: 20, ..Default::default() }).unwrap();
    let mut inputs: Vec<String> = vec![
        "".into(), "is it true that the dax is located in the zup right?".into(),
        "right right".into(), "the right the".into(), "the the".into(),
        "can not not able to".into(), "is is able to to".into(),
        "is it true is it true that that".into(),
        "A-5 lives in Éclair".into(), "X is unable to jump".into(),
        format!("{}dax is wet {}", "the ".repeat(128), "right ".repeat(8)),
    ];
    let seed: serde_json::Value = serde_json::from_str(include_str!("../../../model/bootstrap/reasoning-grammar.json")).unwrap();
    for rule in seed["rewrites"].as_array().unwrap() {
        let from = rule["from"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect::<Vec<_>>().join(" ");
        inputs.extend([from.clone(), format!("dax {from} zup"), format!("{from} {from}"), format!("the {from} right the")]);
    }
    let words = ["the", "is", "it", "true", "that", "able", "to", "can", "not", "unable", "inside", "located", "in", "right", "a", "thing", "dax", "zup", "happened", "before"];
    let mut state = 17u64;
    for length in 0..24 {
        let text = (0..length).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            words[(state >> 32) as usize % words.len()]
        }).collect::<Vec<_>>().join(" ");
        inputs.push(text);
    }
    for text in inputs {
        assert_eq!(r.normalized_tokens(&text), normalize_tokens(tokens(&text)), "{text:?}");
        assert_eq!(r.parse(&text), parse(&text), "{text:?}");
    }
}

#[test]
fn normalization_is_recorded_and_prior_is_in_the_checkpoint() {
    let r = Reasoner::new(ReasonerConfig { dim: 64, symbols: 32, slots: 32, number_range: 20, ..Default::default() }).unwrap();
    r.record_circuits(true);
    assert_eq!(r.normalized_tokens("the dax lives in the zup"), ["dax", "is", "in", "zup"]);
    let recorded = r.take_recorded();
    assert!(recorded.iter().any(|run| run.program == "row-lookup"));
    assert!(recorded.iter().any(|run| run.program == "sequence-rewrite"));
    let tensors = r.tensors();
    let prior: Vec<_> = tensors.iter().filter(|t| t.group == "bootstrap_grammar").collect();
    assert_eq!(prior.len(), 31);
    assert!(tensors.iter().any(|t| t.name.starts_with("prog.row-lookup.")));
    assert!(tensors.iter().any(|t| t.name.starts_with("prog.sequence-rewrite.")));
}
