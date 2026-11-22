use zerolm_engine::isa::{self, Bindings, Buf, Outputs, View};
use zerolm_wsp::{circuits::{row_lookup, rewrite}, Compiled};

fn run(c: &Compiled, inputs: Vec<(&str, Buf)>, external: Vec<(&str, View<'_>)>) -> Outputs {
    let mut bindings = Bindings::new();
    for (name, buf) in inputs {
        bindings = bindings.input(name, buf);
    }
    for (name, view) in external {
        bindings = bindings.weight(name, view);
    }
    for t in &c.tensors {
        bindings = bindings.weight(&t.name, t.view());
    }
    isa::run(&c.program, bindings).unwrap()
}

fn lookup(c: &Compiled, input: &[f32], keys: &[f32], values: &[f32]) -> Outputs {
    run(c, vec![("keys", Buf::new(input.len() / 2, 2, input.to_vec()))], vec![
        ("table_keys", View::new(keys.len() / 2, 2, keys)),
        ("table_values", View::new(values.len() / 2, 2, values)),
    ])
}

#[test]
fn exact_lookup_abstains_on_unknown_and_duplicate_rows() {
    let c = row_lookup::compile("lookup", &row_lookup::Params { key_cols: 2, value_cols: 2 }).unwrap();
    let keys = [1.0, 2.0, 3.0, 4.0];
    let values = [11.0, 12.0, 21.0, 22.0];
    let input = [3.0, 4.0, 1.0, 2.0, 5.0, 6.0];
    let out = lookup(&c, &input, &keys, &values);
    assert_eq!(out.get("values").data, [21.0, 22.0, 11.0, 12.0, 0.0, 0.0]);
    assert_eq!(out.get("unique").data, [1.0, 1.0, 0.0]);
    let duplicates = lookup(&c, &[1.0, 2.0], &[1.0, 2.0, 1.0, 2.0], &values);
    assert_eq!(duplicates.get("matches").data, [2.0]);
    assert_eq!(duplicates.get("values").data, [0.0, 0.0]);
    for block in ["match", "unique", "read"] {
        let mut broken = c.clone();
        broken.ablate_block(block).unwrap();
        assert_eq!(lookup(&broken, &input, &keys, &values).get("values").data, [0.0; 6], "{block}");
    }
}

#[test]
fn lookup_is_equivariant_to_a_bijective_renaming_of_key_bytes() {
    let c = row_lookup::compile("lookup", &row_lookup::Params { key_cols: 2, value_cols: 2 }).unwrap();
    let keys = [0.0, 65.0, 255.0, 17.0];
    let input = [255.0, 17.0, 0.0, 65.0, 42.0, 43.0];
    let values = [11.0, 12.0, 21.0, 22.0];
    let renamed = |rows: &[f32]| rows.iter().map(|x| ((*x as usize * 73 + 19) % 256) as f32).collect::<Vec<_>>();
    let before = c.sha256();
    let original = lookup(&c, &input, &keys, &values);
    let permuted = lookup(&c, &renamed(&input), &renamed(&keys), &values);
    assert_eq!(original, permuted);
    assert_eq!(c.sha256(), before, "no compiler constants change with the alphabet");
    // This is a representation-equivariance test, not a claim that the
    // external seed dictionary has learned an unknown alphabet permutation.
    assert_ne!(lookup(&c, &renamed(&input), &keys, &values).get("values"), original.get("values"));
}

fn rewrite_once(c: &Compiled, ids: &[f32], cursor: f32, length: f32, at_end: f32, delete: bool) -> Outputs {
    let to_ids = if delete { [0.0, 0.0, 0.0, 0.0] } else { [7.0, 0.0, 0.0, 0.0] };
    let to_mask = if delete { [0.0, 0.0, 0.0, 0.0] } else { [1.0, 0.0, 0.0, 0.0] };
    run(c, vec![
        ("ids", Buf::row(ids.to_vec())), ("origins", Buf::row(vec![10.0, 11.0, 12.0, 13.0])),
        ("cursor", Buf::row(vec![cursor])), ("length", Buf::row(vec![length])),
        ("base", Buf::row(vec![100.0])),
    ], vec![
        ("from_ids", View::new(1, 4, &[1.0, 2.0, 0.0, 0.0])),
        ("from_mask", View::new(1, 4, &[1.0, 1.0, 0.0, 0.0])),
        ("to_ids", View::new(1, 4, &to_ids)), ("to_mask", View::new(1, 4, &to_mask)),
        ("at_end", View::new(1, 1, &[at_end])),
    ])
}

#[test]
fn rewrite_matches_emits_deletes_and_stops_at_the_input_end() {
    let c = rewrite::compile("rewrite", &rewrite::Params { window: 4 }).unwrap();
    let out = rewrite_once(&c, &[1.0, 2.0, 9.0, 8.0], 0.0, 4.0, 0.0, false);
    assert_eq!(out.get("count").data, [1.0]);
    assert_eq!(out.get("output_ids").data, [7.0, 0.0, 0.0, 0.0]);
    assert_eq!(out.get("output_origins").data, [107.0, 0.0, 0.0, 0.0]);
    assert_eq!(out.get("next_cursor").data, [2.0]);
    let deleted = rewrite_once(&c, &[1.0, 2.0, 0.0, 0.0], 0.0, 2.0, 1.0, true);
    assert_eq!(deleted.get("count").data, [0.0]);
    assert_eq!(deleted.get("next_cursor").data, [2.0]);
    for (ids, cursor, length, at_end) in [
        ([1.0, 3.0, 0.0, 0.0], 0.0, 2.0, 0.0),
        ([1.0, 2.0, 0.0, 0.0], 0.0, 1.0, 0.0),
        ([1.0, 2.0, 9.0, 8.0], 0.0, 4.0, 1.0),
    ] {
        let out = rewrite_once(&c, &ids, cursor, length, at_end, false);
        assert_eq!(out.get("matched").data, [0.0]);
        assert_eq!(out.get("output_origins").data, [10.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.get("next_cursor").data, [1.0]);
    }
    let done = rewrite_once(&c, &[0.0; 4], 4.0, 4.0, 0.0, false);
    assert_eq!(done.get("count").data, [0.0]);
    assert_eq!(done.get("next_cursor").data, [4.0]);
    let mut disabled = c.clone();
    disabled.ablate_block("end").unwrap();
    assert_eq!(rewrite_once(&disabled, &[1.0, 2.0, 0.0, 0.0], 0.0, 2.0, 0.0, false).get("matched").data, [0.0]);
}
