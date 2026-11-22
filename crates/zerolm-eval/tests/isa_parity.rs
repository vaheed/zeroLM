//! The reference interpreter reproduces the engine interpreter bit for bit on
//! seeded programs that use every ISA operation.

use zerolm_engine::isa::{self, bundle, Axis, Bindings, Buf, Decl, GateFn, Instr, NormKind, Program, ReduceFn, View};
use zerolm_eval::isa_ref::{self, T};
use zerolm_tensor::Rng;

#[test]
fn parser_normalization_replays_with_exported_seed_tables() {
    let r = zerolm_reasoner::Reasoner::new(zerolm_reasoner::ReasonerConfig {
        dim: 64, symbols: 32, slots: 32, number_range: 20, ..Default::default()
    }).unwrap();
    r.record_circuits(true);
    for text in ["the dax lives in the zup", "is it true that dax is unable to jump right?", "a thing happened before dax", "right right", "dax has -0 nubs", "dax has 1000001 nubs", "if dax is wet then dax is cold", "dax is wet?\u{2003}"] {
        r.parse(text);
    }
    let runs = r.take_recorded();
    for program in ["row-lookup", "sequence-rewrite", "byte-lookup", "token-number", "token-features", "masked-lookup", "question-tail", "row-compare", "template-match"] {
        assert!(runs.iter().any(|run| run.program == program));
    }
    for run in runs {
        let inputs: Vec<_> = run.inputs.iter().map(|(name, buf)| (name.as_str(), to_t(buf))).collect();
        let reference = isa_ref::run(&run.bundle, &inputs).unwrap();
        for (name, value) in run.outputs.values {
            let expected = &reference[&name];
            assert_eq!((value.rows, value.cols), (expected.r, expected.c));
            assert_eq!(value.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{}: {name}", run.program);
        }
    }
}

#[test]
fn lexer_edge_cases_and_ablations_replay_bit_exactly() {
    let base = zerolm_wsp::shipped("language-lex").unwrap();
    let l = base.program.inputs[0].cols.unwrap();
    for block in [None, Some("classify"), Some("fold"), Some("segment"), Some("order"), Some("carry")] {
        let mut c = base.clone();
        if let Some(block) = block {
            c.ablate_block(block).unwrap();
        }
        let weights: Vec<_> = c.tensors.iter().map(|t| (t.name.as_str(), t.view())).collect();
        let exported = bundle::encode(&c.program, &weights);
        for offset in [0usize, 128, 256] {
            let text = Buf::row((0..l).map(|i| ((offset + i) % 256) as f32).collect());
            for tail in [0.0, 1.0] {
                let tail = Buf::row(vec![tail]);
                let mut bindings = Bindings::new().input("text", text.clone()).input("tail", tail.clone());
                for (name, view) in &weights {
                    bindings = bindings.weight(name, *view);
                }
                let actual = isa::run(&c.program, bindings).unwrap();
                let expected = isa_ref::run(&exported, &[("text", to_t(&text)), ("tail", to_t(&tail))]).unwrap();
                for (name, value) in actual.values {
                    let expected = &expected[&name];
                    assert_eq!((value.rows, value.cols), (expected.r, expected.c));
                    assert_eq!(value.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{block:?}: {name}");
                }
            }
        }
    }
}

fn s(x: &str) -> String {
    x.to_string()
}

fn rand(rows: usize, cols: usize, seed: u64, tag: &str) -> Buf {
    let mut r = Rng::from_tag(seed, tag);
    Buf::new(rows, cols, (0..rows * cols).map(|_| r.uniform(2.0)).collect())
}

fn gate(f: GateFn, a: &str, b: Option<&str>, out: &str) -> Instr {
    Instr::Gate { f, a: s(a), b: b.map(s), out: s(out) }
}

/// Every op family, with broadcasting, a step loop, and a delta-rule state.
fn program() -> Program {
    let mut body = vec![
        Instr::Matvec { w: s("w"), x: s("x"), out: s("h"), transpose: false },
        Instr::Matvec { w: s("w"), x: s("h"), out: s("ht"), transpose: true },
        Instr::Matmul { a: s("q"), b: s("w"), out: s("qw"), ab: false },
        Instr::Softmax { x: s("qw"), out: s("att") },
        Instr::Norm { kind: NormKind::L2, x: s("qw"), eps: s("eps"), out: s("l2") },
        Instr::Norm { kind: NormKind::Rms, x: s("qw"), eps: s("eps"), out: s("rms") },
        Instr::Reduce { f: ReduceFn::Max, x: s("qw"), axis: Axis::Rows, out: s("rmax") },
        Instr::Reduce { f: ReduceFn::Min, x: s("qw"), axis: Axis::Cols, out: s("cmin") },
        Instr::Reduce { f: ReduceFn::Sum, x: s("qw"), axis: Axis::All, out: s("tot") },
        Instr::Topk { x: s("qw"), k: 2, idx: Some(s("ti")), val: Some(s("tv")), mask: Some(s("tm")) },
        Instr::Bind { a: s("p"), b: s("pb"), out: s("pq"), conj: false },
        Instr::Bind { a: s("pq"), b: s("pb"), out: s("pu"), conj: true },
        Instr::HashRows { bytes: s("bytes"), table: s("w"), n: 3, out: s("hr") },
        Instr::Reshape { x: s("qw"), rows: 1, cols: 15, out: s("flat") },
        Instr::Transpose { x: s("qw"), out: s("qwt") },
        Instr::Transpose { x: s("w"), out: s("wt") },
        Instr::Matmul { a: s("tm"), b: s("wt"), out: s("sparse_mm"), ab: false },
        Instr::Matmul { a: s("tm"), b: s("w"), out: s("sparse_ab"), ab: true },
        gate(GateFn::Ge, "flat", Some("zero"), "pos"),
        Instr::Scan { x: s("pos"), out: s("pre"), exclusive: false },
        Instr::Scan { x: s("pos"), out: s("pre_x"), exclusive: true },
    ];
    for (i, f) in [GateFn::Sigmoid, GateFn::Tanh, GateFn::Relu, GateFn::Exp, GateFn::Neg, GateFn::Copy].into_iter().enumerate() {
        body.push(gate(f, "qw", None, &format!("u{i}")));
    }
    for (i, f) in [GateFn::Add, GateFn::Sub, GateFn::Mul, GateFn::Max, GateFn::Min, GateFn::Ge].into_iter().enumerate() {
        body.push(gate(f, "qw", Some("rmax"), &format!("b{i}")));
        body.push(gate(f, "qw", Some("cmin"), &format!("c{i}")));
    }
    body.push(gate(GateFn::Copy, "acc0", None, "acc"));
    body.push(Instr::Step {
        n: 4,
        body: vec![
            gate(GateFn::Tanh, "acc", None, "acc"),
            Instr::Matvec { w: s("w"), x: s("acc"), out: s("acc"), transpose: true },
            Instr::Matvec { w: s("w"), x: s("acc"), out: s("acc"), transpose: false },
            Instr::DeltaWrite { m: s("m"), k: s("acc"), v: s("hk"), lr: s("lr"), decay: s("decay"), eps: s("eps") },
        ],
    });
    let mut outputs: Vec<String> = ["h", "ht", "qw", "att", "l2", "rms", "rmax", "cmin", "tot", "ti", "tv", "tm", "pq", "pu", "hr", "pre", "pre_x", "acc", "qwt", "sparse_mm", "sparse_ab"]
        .into_iter()
        .map(s)
        .collect();
    outputs.extend((0..6).map(|i| format!("u{i}")));
    outputs.extend((0..6).flat_map(|i| [format!("b{i}"), format!("c{i}")]));
    Program {
        name: s("parity"),
        inputs: vec![Decl::new("x", Some(1), 4), Decl::new("q", None, 4), Decl::new("bytes", Some(2), 6), Decl::new("acc0", Some(1), 5)],
        weights: [("w", 5, 4), ("p", 3, 8), ("pb", 1, 8), ("eps", 1, 1), ("zero", 1, 1), ("lr", 1, 1), ("decay", 1, 1), ("hk", 1, 7)]
            .iter()
            .map(|(n, r, c)| Decl::new(n, Some(*r), *c))
            .collect(),
        state: vec![Decl::new("m", Some(7), 5)],
        body,
        outputs,
    }
}

fn to_t(b: &Buf) -> T {
    T { r: b.rows, c: b.cols, d: b.data.clone() }
}

#[test]
fn reference_interpreter_matches_the_engine_bit_for_bit() {
    let p = program();
    for seed in 1..=20u64 {
        let ws: Vec<(&str, Buf)> = vec![
            ("w", rand(5, 4, seed, "w")),
            ("p", rand(3, 8, seed, "p")),
            ("pb", rand(1, 8, seed, "pb")),
            ("eps", Buf::scalar(1e-6)),
            ("zero", Buf::scalar(0.0)),
            ("lr", Buf::scalar(0.5)),
            ("decay", Buf::scalar(0.9)),
            ("hk", rand(1, 7, seed, "hk")),
        ];
        let mut rng = Rng::from_tag(seed, "bytes");
        let bytes = Buf::new(2, 6, (0..12).map(|_| rng.below(256) as f32).collect());
        let inputs: Vec<(&str, Buf)> =
            vec![("x", rand(1, 4, seed, "x")), ("q", rand(3, 4, seed, "q")), ("bytes", bytes), ("acc0", rand(1, 5, seed, "acc"))];
        let m0 = rand(7, 5, seed, "m");
        let views: Vec<(&str, View)> = ws.iter().map(|(n, b)| (*n, b.view())).collect();
        let mut bind = Bindings::new().state("m", m0.clone());
        for (n, v) in &views {
            bind = bind.weight(n, *v);
        }
        for (n, b) in &inputs {
            bind = bind.input(n, b.clone());
        }
        let got = isa::run(&p, bind).unwrap();
        let bytes = bundle::encode(&p, &views);
        let mut ref_in: Vec<(&str, T)> = inputs.iter().map(|(n, b)| (*n, to_t(b))).collect();
        ref_in.push(("m", to_t(&m0)));
        let want = isa_ref::run(&bytes, &ref_in).unwrap();
        for (name, b) in &got.values {
            let w = &want[name];
            assert_eq!((b.rows, b.cols), (w.r, w.c), "{name} shape");
            let (x, y): (Vec<u32>, Vec<u32>) = (b.data.iter().map(|v| v.to_bits()).collect(), w.d.iter().map(|v| v.to_bits()).collect());
            assert_eq!(x, y, "seed {seed}: output {name} differs");
        }
        let m = &got.state["m"];
        assert_eq!(m.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want["state:m"].d.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_ne!(m.data, m0.data, "delta_write changed the state");
    }
}
