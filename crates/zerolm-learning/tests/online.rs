use zerolm_engine::VOCAB;
use zerolm_learning::ngram::Ngram;
use zerolm_learning::online_predictor::{SIGNAL, CHANGED_TENSORS};
use zerolm_learning::streams::{sha256_hex, SpecError, StreamSpec};
use zerolm_learning::*;

const STREAM: &str = "the cat sees the red ball.\nmia holds the blue cup.\nthe dog is in the park.\n";

fn spec() -> StreamSpec {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../research/streams/compositional-v1.json");
    StreamSpec::parse(&std::fs::read_to_string(root).unwrap()).unwrap()
}

fn new(rule: Rule) -> OnlinePredictor {
    OnlinePredictor::new(1, rule, Constants::default(), "test")
}

#[test]
fn initialization_is_deterministic_and_a_real_update_changes_the_hash() {
    for rule in [Rule::Delta, Rule::Hebbian] {
        let (a, mut b) = (new(rule), new(rule));
        let h0 = a.weights_sha256();
        assert_eq!(h0, b.weights_sha256());
        assert_ne!(h0, OnlinePredictor::new(2, rule, Constants::default(), "test").weights_sha256());
        b.observe(b'x').unwrap();
        assert_eq!(b.events().len(), 1);
        assert!(b.events()[0].update_norm > 0.0);
        assert_ne!(h0, b.weights_sha256(), "{rule:?}");
    }
}

#[test]
fn no_update_when_learning_is_disabled_or_frozen() {
    let mut f = new(Rule::Frozen);
    let h = f.weights_sha256();
    f.set_learning(true); // Frozen ignores this.
    f.observe_all(STREAM.as_bytes()).unwrap();
    f.close_block();
    assert_eq!(f.weights_sha256(), h);
    assert!(f.events().is_empty() && f.checkpoints().is_empty());

    let mut d = new(Rule::Delta);
    let h = d.weights_sha256();
    d.set_learning(false);
    assert!(!d.learning());
    d.observe_all(STREAM.as_bytes()).unwrap();
    d.close_block();
    assert_eq!(d.weights_sha256(), h);
    assert!(d.events().is_empty() && d.checkpoints().is_empty());
    assert_eq!(d.offset(), STREAM.len(), "the state still consumes every byte");
}

#[test]
fn event_log_corresponds_exactly_to_observed_bytes_and_targets() {
    let bytes = STREAM.as_bytes();
    let mut p = new(Rule::Delta);
    p.set_checkpoint_every(17);
    let mut probs = vec![0.0; VOCAB];
    for &b in bytes {
        p.predict(&mut probs);
        let obs = p.observe(b).unwrap();
        // The logged probability is the one predicted before the reveal.
        assert_eq!(obs.p_target, probs[b as usize]);
    }
    p.close_block();
    let ev = p.events();
    assert_eq!(ev.len(), bytes.len());
    for (i, e) in ev.iter().enumerate() {
        assert_eq!((e.seq, e.stream_offset, e.target), (i, i, bytes[i]));
        assert!(e.p_target > 0.0 && e.p_target <= 1.0);
    }
    let cps = p.checkpoints();
    assert_eq!(cps.first().unwrap().first_seq, 0);
    assert_eq!(cps.last().unwrap().last_seq, bytes.len() - 1);
    for w in cps.windows(2) {
        assert_eq!(w[1].first_seq, w[0].last_seq + 1, "blocks are contiguous");
    }
    for c in cps {
        assert_eq!(c.targets_sha256, sha256_hex(&bytes[c.first_seq..=c.last_seq]));
        assert_eq!((c.signal, c.rule, c.input_stream.as_str()), (SIGNAL, "delta", "test"));
        assert_eq!(c.changed_tensors, CHANGED_TENSORS.to_vec());
    }
    assert_eq!(cps.last().unwrap().checkpoint_sha256, p.weights_sha256());
}

#[test]
fn no_future_byte_leaks_into_the_prediction() {
    // Same prefix, different next bytes and different futures.
    let prefix = b"the cat sees the ";
    let (mut a, mut b) = (new(Rule::Delta), new(Rule::Delta));
    a.observe_all(prefix).unwrap();
    b.observe_all(prefix).unwrap();
    assert_eq!(a.weights_sha256(), b.weights_sha256());
    assert_eq!(a.features(), b.features());
    let (mut pa, mut pb) = (vec![0.0; VOCAB], vec![0.0; VOCAB]);
    a.predict(&mut pa);
    b.predict(&mut pb);
    assert_eq!(pa, pb, "prediction at t depends only on bytes < t");
    let oa = a.observe(b'r').unwrap();
    let ob = b.observe(b'z').unwrap();
    assert_eq!(oa.p_target, pb[b'r' as usize]);
    assert_eq!(ob.p_target, pa[b'z' as usize]);
    assert_ne!(a.weights_sha256(), b.weights_sha256());
}

#[test]
fn evaluation_never_changes_weights_and_learning_beats_frozen() {
    let text = "abcab ".repeat(400);
    for rule in [Rule::Delta, Rule::Hebbian] {
        let mut p = new(rule);
        let before = p.evaluate(text.as_bytes()).unwrap();
        assert!((before.bpb - 8.0).abs() < 1e-3, "untrained readout is uniform");
        p.observe_all(text.as_bytes()).unwrap();
        let h = p.weights_sha256();
        let after = p.evaluate(text.as_bytes()).unwrap();
        assert_eq!(h, p.weights_sha256());
        assert!(after.bpb < 2.0, "{rule:?}: {}", after.bpb);
        assert!(after.accuracy > before.accuracy);
        assert!((0.0..=1.0).contains(&after.ece));
    }
}

#[test]
fn severed_recurrence_makes_features_history_free() {
    let (mut a, mut b) = (new(Rule::Delta), new(Rule::Delta));
    a.sever_recurrence();
    b.sever_recurrence();
    a.set_learning(false);
    b.set_learning(false);
    a.observe_all(b"xyz q").unwrap();
    b.observe_all(b"abc q").unwrap();
    assert_eq!(a.features(), b.features());
    let mut c = new(Rule::Delta);
    c.set_learning(false);
    c.observe_all(b"xyz q").unwrap();
    assert_ne!(a.features(), c.features());
}

#[test]
fn streams_are_deterministic_role_separated_and_clean() {
    let s = spec();
    let a = s.split(7, 5000, 1000);
    let b = s.split(7, 5000, 3000);
    assert_eq!(a.exposure, b.exposure, "exposure is independent of eval size");
    assert_eq!(a.heldout, b.heldout[..a.heldout.len()]);
    assert_ne!(a.exposure, s.split(8, 5000, 1000).exposure);
    assert!(a.exposure.len() >= 5000 && a.heldout.len() >= 1000 && a.novel.len() >= 1000);
    assert!(a.contamination(&s).clean(), "{:?}", a.contamination(&s));
    // Every novel sentence binds a held-out combination absent from exposure.
    for combo in &s.held_out_combinations {
        let words: Vec<&str> = combo.values().map(String::as_str).collect();
        let hit = a.exposure.lines().any(|l| {
            let ws: Vec<&str> = l.split(|c: char| !c.is_ascii_alphanumeric()).collect();
            words.iter().all(|w| ws.contains(w))
        });
        assert!(!hit, "{combo:?} leaked into exposure");
    }
    assert_eq!(s.sha256(), spec().sha256());
}

#[test]
fn contamination_is_detected() {
    let s = spec();
    let mut a = s.split(7, 3000, 500);
    // Simulate a leak of a held-out combination ("zoe drops") into exposure.
    a.exposure.push_str("zoe drops the red cup.\n");
    let c = a.contamination(&s);
    assert_eq!(c.exposure_sentences_with_held_out_combination, 1);
    assert!(!c.clean());
}

#[test]
fn invalid_specs_are_rejected() {
    let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../research/streams/compositional-v1.json")).unwrap();
    let bad_slot = base.replacen("{name} {verb} the", "{nobody} {verb} the", 1);
    assert_eq!(StreamSpec::parse(&bad_slot).unwrap_err(), SpecError::UnknownSlot("nobody".into()));
    let bad_word = base.replacen("\"color\": \"green\", \"object\": \"kite\"", "\"color\": \"green\", \"object\": \"car\"", 1);
    assert!(matches!(StreamSpec::parse(&bad_word), Err(SpecError::UnknownWord { .. })));
    assert!(matches!(StreamSpec::parse("{}"), Err(SpecError::Parse(_))));
}

#[test]
fn ngram_baselines_learn_from_exposure() {
    let text = "abcab ".repeat(200);
    let mut uni = Ngram::new(1);
    let mut tri = Ngram::new(3);
    assert!((uni.bpb(text.as_bytes()) - 8.0).abs() < 1e-9);
    uni.observe_all(text.as_bytes());
    tri.observe_all(text.as_bytes());
    let (u, t) = (uni.bpb(text.as_bytes()), tri.bpb(text.as_bytes()));
    assert!((u - 1.918).abs() < 0.05, "unigram entropy of 'abcab ' is ~1.918 bits: {u}");
    assert!(t < 0.5 && t < u);
}
