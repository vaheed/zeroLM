use zerolm_wsp::patch::{self, PatchError};
use zerolm_wsp::{check_pins, compile, parse_spec, shipped, SPECS};

#[test]
fn compilation_is_reproducible_and_matches_the_pins() {
    for (file, json) in SPECS {
        let spec = parse_spec(json).unwrap();
        let (a, b) = (compile(&spec).unwrap(), compile(&spec).unwrap());
        assert_eq!(a.sha256(), b.sha256(), "{file}");
        let bits = |c: &zerolm_wsp::Compiled| c.tensors.iter().flat_map(|t| t.data.iter().map(|x| x.to_bits())).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b));
        // The program JSON round-trips exactly.
        let back: zerolm_engine::isa::Program = serde_json::from_str(&a.program.to_json()).unwrap();
        assert_eq!(back, a.program);
    }
    for (id, sha, ok) in check_pins().unwrap() {
        assert!(ok, "{id} compiles to {sha}, which is not the pinned hash (re-pin model/prog/pinned.json only for a deliberate change)");
    }
}

#[test]
fn a_parameter_change_changes_the_hash() {
    let mut spec = parse_spec(SPECS[0].1).unwrap();
    let base = compile(&spec).unwrap().sha256();
    let zerolm_wsp::Circuit::AttentionRead(p) = &mut spec.circuit else { panic!("memory-read is an attention_read circuit") };
    p.read_threshold += 0.01;
    assert_ne!(compile(&spec).unwrap().sha256(), base);
}

#[test]
fn patches_round_trip_and_refuse_the_wrong_base() {
    let base = shipped("memory-read").unwrap();
    let mut target = base.clone();
    let t = target.tensors.iter_mut().find(|t| t.name == "coverage_threshold").unwrap();
    t.data[0] = 0.6;
    let p = patch::diff(&base, &target, "raise-coverage", "test patch").unwrap();
    assert_eq!(p.edits.len(), 1);
    let json = serde_json::to_string_pretty(&p).unwrap();
    let p: patch::Patch = serde_json::from_str(&json).unwrap();

    let mut c = base.clone();
    patch::apply(&mut c, &p).unwrap();
    assert_eq!(c.sha256(), target.sha256());
    // Applying twice is refused and leaves the circuit unchanged.
    assert!(matches!(patch::apply(&mut c, &p), Err(PatchError::BaseMismatch { .. })));
    assert_eq!(c.sha256(), target.sha256());
    patch::revert(&mut c, &p).unwrap();
    assert_eq!(c.sha256(), base.sha256());
    assert_eq!(c, base);
    // A tampered edit is refused atomically.
    let mut bad = p.clone();
    bad.edits[0].old[0] = "00000000".into();
    let mut d = base.clone();
    assert!(patch::apply(&mut d, &bad).is_err());
    assert_eq!(d, base);
}

#[test]
fn blocks_can_be_ablated_by_name() {
    let mut c = shipped("memory-read").unwrap();
    assert_eq!(c.blocks(), vec!["attend", "decode", "select", "support", "verify"]);
    // Every inference operator has its own block in the reach circuit.
    let reach = shipped("relation-reach").unwrap();
    for op in zerolm_wsp::circuits::relational::OPERATORS {
        assert!(reach.blocks().iter().any(|b| b == op), "{op}");
    }
    c.ablate_block("verify").unwrap();
    assert!(c.tensors.iter().filter(|t| t.block == "verify").all(|t| t.data.iter().all(|x| *x == 0.0)));
    assert!(c.tensors.iter().any(|t| t.block == "attend" && t.data.iter().any(|x| *x != 0.0)));
    assert!(c.ablate_block("nope").is_err());
}
