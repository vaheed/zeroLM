use zerolm_engine::isa::{self, Bindings, Buf, Outputs};
use zerolm_wsp::circuits::frame_bind::{compile, Params};
use zerolm_wsp::Compiled;

const T: usize = 3;
const D2: usize = 4;

fn circuit() -> Compiled {
    compile("test-frame", &Params { tokens: T, fields: 16, dim: 2, dimensions: 2, relations: 10 }).unwrap()
}

/// Row i of a table: [base + i, 0.5, 0, 0] so every row is distinct.
fn table(rows: usize, base: f32) -> Buf {
    Buf::new(rows, D2, (0..rows).flat_map(|i| [base + i as f32, 0.5, 0.0, 0.25]).collect())
}

fn run(c: &Compiled, frame: [(usize, f32); 8]) -> Outputs {
    let mut f = vec![0.0; 16];
    for (i, v) in frame {
        f[i] = v;
    }
    let mut b = Bindings::new()
        .input("frame", Buf::row(f))
        .input("words", table(2 * T, 10.0))
        .input("dimensions", table(2, 50.0))
        .input("relations", table(10, 100.0))
        .input("quant", table(1, 70.0));
    for t in &c.tensors {
        b = b.weight(&t.name, t.view());
    }
    isa::run(&c.program, b).unwrap()
}

const NONE: (usize, f32) = (10, 0.0);

#[test]
fn a_negated_isa_fact_binds_subject_relation_and_object() {
    let c = circuit();
    // form 1, predicate 1, subject token 0, object token 2, negated.
    let o = run(&c, [(0, 1.0), (1, 1.0), (2, 0.0), (5, 2.0), (9, 1.0), NONE, NONE, NONE]);
    assert_eq!(o.get("mask").data, [1.0, 1.0, 1.0, 0.0, 0.0]);
    let parts = &o.get("parts").data;
    assert_eq!(&parts[..4], &table(2 * T, 10.0).data[..4], "subject is token 0");
    assert_eq!(&parts[4..8], &table(10, 100.0).data[..4], "relation is the first relation");
    assert_eq!(&parts[8..12], &table(2 * T, 10.0).data[2 * 4..3 * 4], "object is token 2");
    let control = &o.get("control").data;
    assert_eq!((control[0], control[1], control[2], control[3], control[7]), (1.0, 1.0, 1.0, -1.0, 3.0));
    assert_eq!(o.get("intern_mask").data, [1.0, 1.0, 0.0]);
}

#[test]
fn a_count_fact_binds_the_quantity_and_a_rule_its_premise_and_conclusion() {
    let c = circuit();
    // predicate 7: thing token 1 (singular form), number from `quant`.
    let o = run(&c, [(0, 1.0), (1, 7.0), (2, 0.0), (15, 1.0), (8, 3.0), NONE, NONE, NONE]);
    assert_eq!(o.get("mask").data, [1.0, 1.0, 0.0, 1.0, 0.0]);
    assert_eq!(&o.get("parts").data[12..16], &table(1, 70.0).data[..]);
    assert_ne!(o.get("rel").data, &table(10, 100.0).data[6 * 4..7 * 4], "relation is bound with the counted thing");
    // rule: premise token 1, conclusion token 2, scope token 0.
    let o = run(&c, [(0, 2.0), (2, 0.0), (11, 1.0), (12, 1.0), (13, 2.0), NONE, NONE, NONE]);
    assert_eq!(o.get("mask").data, [1.0, 1.0, 1.0, 0.0, 1.0]);
    let parts = &o.get("parts").data;
    assert_eq!(&parts[..4], &table(2 * T, 10.0).data[4..8], "subject filler is the premise");
    assert_eq!(&parts[4..8], &table(10, 100.0).data[8 * 4..9 * 4], "relation is `implies`");
    assert_eq!(&parts[16..20], &table(2 * T, 10.0).data[..4], "scope is the named token");
    assert_eq!(o.get("control").data[2], 9.0);
    assert_eq!(o.get("intern_mask").data, [1.0, 1.0, 1.0]);
}

#[test]
fn blocks_causally_control_token_selection_and_predicate_routing() {
    let c = circuit();
    let frame = [(0, 1.0), (1, 1.0), (2, 0.0), (5, 2.0), NONE, NONE, NONE, NONE];
    let base = run(&c, frame);
    assert_ne!(base.get("object").data, [0.0; 4]);
    assert_ne!(base.get("rel").data, [0.0; 4]);
    for (block, output) in [("index", "object"), ("route", "rel")] {
        let mut broken = c.clone();
        broken.ablate_block(block).unwrap();
        let o = run(&broken, frame);
        assert_ne!(o.get(output).data, base.get(output).data, "{block} carries {output}");
    }
    let mut broken = c.clone();
    broken.ablate_block("assemble").unwrap();
    assert_eq!(run(&broken, frame).get("parts").data, [0.0; 20], "assemble carries the stacked fillers");
}
