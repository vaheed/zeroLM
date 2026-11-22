use zerolm_engine::isa::{self, Bindings, Buf, Outputs};
use zerolm_wsp::{circuits::lexer, Compiled};

fn run(c: &Compiled, text: &[u8], tail: f32) -> Outputs {
    let l = c.program.inputs[0].cols.unwrap();
    let mut row = vec![0.0; l];
    for (dst, src) in row.iter_mut().zip(text) {
        *dst = *src as f32;
    }
    let mut bindings = Bindings::new().input("text", Buf::row(row)).input("tail", Buf::row(vec![tail]));
    for t in &c.tensors {
        bindings = bindings.weight(&t.name, t.view());
    }
    isa::run(&c.program, bindings).unwrap()
}

#[test]
fn every_byte_has_the_reference_ascii_class_and_fold() {
    let c = lexer::compile("test-lex", &lexer::Params { chunk_bytes: 1 }).unwrap();
    for byte in 0..=255u8 {
        let out = run(&c, &[byte], 1.0);
        let word = (byte.is_ascii_alphanumeric() || byte == b'-') as u8 as f32;
        for name in ["count", "join", "tail_word"] {
            assert_eq!(out.get(name).data, [word], "byte {byte}, {name}");
        }
        assert_eq!(out.get("folded").data, [byte.to_ascii_lowercase() as f32]);
    }
}

#[test]
fn blocks_causally_control_folding_segmentation_order_and_continuation() {
    let c = lexer::compile("test-lex", &lexer::Params { chunk_bytes: 8 }).unwrap();
    let text = b"AB cd-7Z";
    let base = run(&c, text, 1.0);
    assert_eq!(base.get("count").data, [2.0]);
    assert_eq!(&base.get("start").data[..2], [0.0, 3.0]);
    assert_eq!(&base.get("end").data[..2], [1.0, 7.0]);
    assert_eq!(base.get("join").data, [1.0]);
    assert_eq!(base.get("tail_word").data, [1.0]);
    for block in ["classify", "segment"] {
        let mut ablated = c.clone();
        ablated.ablate_block(block).unwrap();
        assert_eq!(run(&ablated, text, 1.0).get("count").data, [0.0], "{block}");
    }
    let mut ablated = c.clone();
    ablated.ablate_block("fold").unwrap();
    let out = run(&ablated, text, 1.0);
    assert_eq!(out.get("folded").data[0], b'A' as f32);
    assert_eq!(out.get("count").data, base.get("count").data);
    let mut ablated = c.clone();
    ablated.ablate_block("order").unwrap();
    assert_ne!(&run(&ablated, text, 1.0).get("start").data[..2], &base.get("start").data[..2]);
    let mut ablated = c.clone();
    ablated.ablate_block("carry").unwrap();
    let out = run(&ablated, text, 1.0);
    assert_eq!(out.get("join").data, [0.0]);
    assert_eq!(out.get("tail_word").data, [0.0]);
    assert_eq!(out.get("count").data, base.get("count").data);
}

#[test]
fn unsafe_chunk_dimensions_are_rejected_before_allocation() {
    for chunk_bytes in [0, 513, usize::MAX] {
        assert!(lexer::compile("invalid", &lexer::Params { chunk_bytes }).is_err());
    }
}
