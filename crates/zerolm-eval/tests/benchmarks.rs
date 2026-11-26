use zerolm_data::evalsets::{parse_blimp, parse_lambada, verbatim_overlap};
use zerolm_eval::benchmarks::{blimp, lambada, sentence_log2, ByteLm, KneserNey};
use zerolm_eval::ngram::Kn;

/// Uniform model: no information, so every pair ties and nothing is predicted exactly.
struct Uniform;
impl ByteLm for Uniform {
    fn prob(&self, _: &[u8], _: u8) -> f64 {
        1.0 / 256.0
    }
}

fn trained() -> Kn {
    let t = b"\nthe cats sleep. the cat sleeps. the dogs bark. the dog barks. ".repeat(30);
    let mut kn = Kn::new(5);
    for i in 0..t.len() {
        kn.observe(&t[i.saturating_sub(4)..i], t[i]);
    }
    kn
}

// The bad sentences contain a byte sequence the training text never had: this tests the
// harness mechanics (scoring, ties, aggregation), not grammatical knowledge.
const PAIRS: &str = concat!(
    r#"{"sentence_good": "the cats sleep.", "sentence_bad": "the cats sleepz.", "field": "morphology", "UID": "agree", "pairID": "0"}"#, "\n",
    r#"{"sentence_good": "the dog barks.", "sentence_bad": "the dog barkz.", "field": "morphology", "UID": "agree", "pairID": "1"}"#, "\n",
    r#"{"sentence_good": "the dogs bark.", "sentence_bad": "the dogs barkz.", "field": "syntax", "UID": "other", "pairID": "0"}"#, "\n",
);

#[test]
fn blimp_parser_rejects_bad_input() {
    assert_eq!(parse_blimp(PAIRS.as_bytes()).unwrap().len(), 3);
    assert!(parse_blimp(b"").is_err());
    assert!(parse_blimp(b"{\"sentence_good\": \"a\"}").is_err());
    assert!(parse_blimp(br#"{"sentence_good":"","sentence_bad":"a","field":"f","UID":"u"}"#).is_err());
    // Identical pairs occur in the published data and are kept; they always tie.
    let same = parse_blimp(br#"{"sentence_good":"a b","sentence_bad":"a b","field":"f","UID":"u"}"#).unwrap();
    assert_eq!(blimp(&Uniform, &same).ties, 1);
}

#[test]
fn blimp_scores_a_trained_model_above_chance_and_a_uniform_model_at_zero() {
    let pairs = parse_blimp(PAIRS.as_bytes()).unwrap();
    let r = blimp(&KneserNey(&trained()), &pairs);
    assert_eq!((r.pairs, r.correct), (3, 3));
    assert_eq!(r.per_paradigm["agree"], (2, 2));
    assert_eq!(r.per_field["syntax"], (1, 1));
    assert!((r.paradigm_mean - 1.0).abs() < 1e-12);
    // Equal-length sentences tie under a uniform model; ties count as wrong.
    let u = blimp(&Uniform, &parse_blimp(br#"{"sentence_good":"abc","sentence_bad":"abd","field":"f","UID":"u"}"#).unwrap());
    assert_eq!((u.correct, u.ties), (0, 1));
}

#[test]
fn sentence_score_is_a_log_probability_and_scoring_is_pure() {
    let kn = trained();
    let lm = KneserNey(&kn);
    let s = sentence_log2(&lm, "the cat sleeps.");
    assert!(s < 0.0 && s.is_finite());
    assert_eq!(s, sentence_log2(&lm, "the cat sleeps."));
}

#[test]
fn lambada_parser_takes_the_last_word_without_its_punctuation() {
    let (items, skipped) = parse_lambada("He said \"hello, world.\"\nI went home\n...\n".as_bytes()).unwrap();
    assert_eq!(skipped, 1);
    assert_eq!(items[0].target, b"world");
    assert_eq!(items[0].context, b"He said \"hello, ");
    assert_eq!(items[1].target, b"home");
    assert_eq!(items[1].context, b"I went ");
    assert!(parse_lambada(b"\n\n").is_err());
}

#[test]
fn lambada_requires_the_whole_word_and_its_end() {
    let kn = trained();
    let (items, _) = parse_lambada(b"the cat sleeps. the dog barks\nthe cat sleeps. the dog xyzzy\n").unwrap();
    let r = lambada(&KneserNey(&kn), &items);
    assert_eq!((r.items, r.correct), (2, 1), "only the predictable final word is right");
    assert_eq!(lambada(&Uniform, &items).correct, 0);
}

#[test]
fn verbatim_overlap_counts_whole_sentences_only() {
    let exposure = b"yesterday the old man walked home. then he slept. the cat sat on the mat";
    let sentences: Vec<&[u8]> = vec![b"the old man walked home.", b"the old man sat home.", b"the cat sat on the mat"];
    assert_eq!(verbatim_overlap(exposure, &sentences), 2);
    // Very short sentences are chance fragments of longer text, not leakage, and are not checked.
    assert_eq!(verbatim_overlap(b"then he slept. Thomas sang. later", &[b"Thomas sang."]), 0);
}

#[test]
fn lambada_detokenizes_with_fixed_rules() {
    let (items, _) = parse_lambada(b"`` i do n't know , '' she said . i 'll meet you there\n").unwrap();
    assert_eq!(items[0].context, b"\"i don't know,\" she said. i'll meet you ");
    assert_eq!(items[0].target, b"there");
}
