use zerolm_engine::*;
use zerolm_nn::Kind;
use zerolm_tokenizer as tok;

fn text() -> Vec<u8> {
    "the quick brown fox jumps over the lazy dog. ".repeat(30).into_bytes()
}

#[test]
fn untrained_model_is_near_uniform_bits_per_byte() {
    for k in Kind::ALL {
        let m = Model::new(k, 1, 16, 4_000);
        let bpb = m.bits_per_byte(&text()).unwrap();
        assert!((bpb - 8.0).abs() < 0.2, "{k:?}: {bpb}");
    }
}

#[test]
fn chunked_feed_equals_bytewise_and_resume_is_exact() {
    let bytes = text();
    for k in Kind::ALL {
        let m = Model::new(k, 2, 16, 4_000);
        let mut a = m.init_state();
        m.feed(&mut a, &bytes).unwrap();
        let mut b = m.init_state();
        for chunk in bytes.chunks(101) {
            let saved = m.save_state(&b);
            b = m.load_state(&saved).unwrap();
            m.feed(&mut b, chunk).unwrap();
        }
        assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{k:?}");
    }
}

#[test]
fn state_size_is_constant_over_a_long_stream() {
    for k in Kind::ALL {
        let m = Model::new(k, 3, 16, 4_000);
        let mut st = m.init_state();
        let n = st.len();
        let bytes = text();
        for _ in 0..20 {
            m.feed(&mut st, &bytes).unwrap();
        }
        assert_eq!(st.len(), n);
    }
}

#[test]
fn any_byte_sequence_is_accepted_including_invalid_utf8() {
    let m = Model::new(Kind::Ssm, 1, 16, 4_000);
    let all: Vec<u8> = (0..=255).collect();
    let toks = tok::encode_bytes(&all);
    let bytes = tok::decode_bytes(&toks);
    let mut st = m.init_state();
    m.feed(&mut st, &bytes).unwrap();
}

#[test]
fn model_init_is_deterministic() {
    let a = Model::new(Kind::Gru, 5, 16, 4_000);
    let b = Model::new(Kind::Gru, 5, 16, 4_000);
    assert_eq!(a.embed, b.embed);
    assert_eq!(a.head, b.head);
    assert_eq!(a.param_count(), b.param_count());
}
