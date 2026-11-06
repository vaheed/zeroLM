use zerolm_engine::bootstrap::*;
use zerolm_nn::Kind;

#[test]
fn parse_reports_exact_byte_spans() {
    let line = "teach sky => blue";
    match parse_line(line, 100).unwrap() {
        Command::Teach { key, value } => {
            assert_eq!(&line[key.start - 100..key.end - 100], "sky");
            assert_eq!(&line[value.start - 100..value.end - 100], "blue");
        }
        c => panic!("{c:?}"),
    }
}

#[test]
fn non_commands_and_malformed_commands_are_distinguished() {
    assert_eq!(parse_line("hello there", 0), Err(ParseError::NotACommand));
    assert_eq!(parse_line("teachers are great", 0), Err(ParseError::NotACommand));
    assert!(matches!(parse_line("teach sky", 0), Err(ParseError::Malformed(_))));
    assert!(matches!(parse_line("ask", 0), Err(ParseError::Malformed(_))));
    assert!(matches!(parse_line("repeat now", 0), Err(ParseError::Malformed(_))));
    assert!(matches!(parse_line("teach a => b => c", 0), Err(ParseError::Malformed(_))));
}

#[test]
fn full_protocol_roundtrip() {
    let m = default_model(DEFAULT_SEED);
    let mut s = Session::new(&m);
    assert_eq!(s.handle_line("ask sky"), "unknown sky");
    assert_eq!(s.handle_line("teach sky => blue"), "noted sky (unconfirmed)");
    assert_eq!(s.handle_line("ask sky"), "sky => blue (unconfirmed)");
    assert_eq!(s.handle_line("confirm sky"), "confirmed sky");
    assert_eq!(s.handle_line("ask sky"), "sky => blue (confirmed)");
    assert!(s.handle_line("teach sky => green").starts_with("conflict"));
    assert_eq!(s.handle_line("correct sky => grey"), "corrected sky (unconfirmed)");
    assert_eq!(s.handle_line("ask sky"), "sky => grey (unconfirmed)");
    assert_eq!(s.handle_line("repeat"), "sky => grey (unconfirmed)");
    assert_eq!(s.handle_line("forget sky"), "forgot sky");
    assert_eq!(s.handle_line("ask sky"), "unknown sky");
    assert_eq!(s.handle_line("forget sky"), "unknown sky");
}

#[test]
fn unicode_keys_and_values_work_and_spans_stay_byte_exact() {
    let m = default_model(DEFAULT_SEED);
    let mut s = Session::new(&m);
    s.handle_line("teach héllo => 世界 \u{1F600}");
    assert_eq!(s.handle_line("ask héllo"), "héllo => 世界 \u{1F600} (unconfirmed)");
    let ev = &s.events()[0];
    let line = "teach héllo => 世界 \u{1F600}";
    let v = ev.value_span.unwrap();
    assert_eq!(&line.as_bytes()[v.start..v.end], "世界 \u{1F600}".as_bytes());
}

#[test]
fn spans_accumulate_across_lines() {
    let m = default_model(DEFAULT_SEED);
    let mut s = Session::new(&m);
    s.handle_line("just text");
    s.handle_line("teach a => b");
    let e = &s.events()[1];
    assert_eq!(e.line_span, Span { start: 10, end: 22 });
    assert_eq!(e.key_span.unwrap(), Span { start: 16, end: 17 });
    assert_eq!(s.offset(), 23);
}

#[test]
fn session_never_changes_weights_and_is_deterministic() {
    let m = default_model(DEFAULT_SEED);
    let before = m.weights_sha256();
    let script = ["teach a => b", "confirm a", "free text \u{0}\u{1F600}", "ask a", "forget a", "bogus => line", "repeat"];
    let run = || {
        let mut s = Session::new(&m);
        let out: Vec<String> = script.iter().map(|l| s.handle_line(l)).collect();
        (out, s.state().to_vec())
    };
    let (o1, st1) = run();
    let (o2, st2) = run();
    assert_eq!(o1, o2);
    assert_eq!(st1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), st2.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert_eq!(m.weights_sha256(), before);
}

#[test]
fn every_event_carries_the_unchanged_weights_hash() {
    let m = default_model(DEFAULT_SEED);
    let h = m.weights_sha256();
    let mut s = Session::new(&m);
    s.handle_line("teach x => y");
    s.handle_line("ask x");
    assert!(s.events().iter().all(|e| e.weights_sha256 == h));
}

#[test]
fn reset_forgets_everything() {
    let m = default_model(DEFAULT_SEED);
    let mut s = Session::new(&m);
    s.handle_line("teach x => y");
    s.reset();
    assert_eq!(s.handle_line("ask x"), "unknown x");
    assert_eq!(s.events().len(), 1);
}

#[test]
fn pinned_initialization_hash_and_seed_manifest_agree() {
    let seed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/bootstrap/seed.json")).unwrap()).unwrap();
    assert_eq!(seed["default_seed"], DEFAULT_SEED);
    assert_eq!(seed["embed_dim"], EMBED_DIM);
    assert_eq!(seed["cell_param_budget"], CELL_BUDGET);
    assert_eq!(seed["cell"], DEFAULT_KIND.name());
    let cmds: Vec<&str> = seed["commands"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(cmds, COMMANDS);
    assert_eq!(seed["delimiter"], DELIMITER);
    assert_eq!(seed["control_tokens"]["BOS"], zerolm_tokenizer::BOS);
    assert_eq!(seed["control_tokens"]["EOS"], zerolm_tokenizer::EOS);
    assert_eq!(seed["control_tokens"]["PAD"], zerolm_tokenizer::PAD);
    let pinned = seed["init_weights_sha256"].as_str().unwrap();
    assert_eq!(default_model(DEFAULT_SEED).weights_sha256(), pinned, "initialization changed; re-pin deliberately");
    assert_ne!(default_model(DEFAULT_SEED + 1).weights_sha256(), pinned);
}

#[test]
fn architecture_manifest_matches_bootstrap_constants() {
    let a: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/specs/architecture.json")).unwrap()).unwrap();
    assert_eq!(a["selected"], Kind::name(DEFAULT_KIND));
    assert_eq!(a["embed_dim"], EMBED_DIM);
    assert_eq!(a["cell_param_budget"], CELL_BUDGET);
}
