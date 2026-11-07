use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use zerolm_memory::checkpoint::{self, CheckpointError, Fault, LoadSource};
use zerolm_memory::persistent_memory::{field, DELETED, SUPERSEDED};
use zerolm_memory::{Ablation, MemoryConfig, MemoryError, MemoryStore, Outcome};
use zerolm_tensor::Rng;

const S: &str = "test:memory";

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("zerolm-memory-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn word(rng: &mut Rng) -> String {
    const C: &[u8] = b"bdfgklmnprstvz";
    const V: &[u8] = b"aeiou";
    let n = 2 + rng.below(2);
    (0..n).flat_map(|_| [C[rng.below(C.len())] as char, V[rng.below(V.len())] as char]).collect()
}

/// `n` distinct (key, value) facts.
fn facts(seed: u64, n: usize) -> Vec<(String, String)> {
    let attrs = ["color", "home", "friend", "food", "size", "shape", "sound", "owner"];
    let mut rng = Rng::new(seed);
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    while out.len() < n {
        let k = format!("{} {}", word(&mut rng), attrs[rng.below(attrs.len())]);
        if seen.insert(k.clone()) {
            out.push((k, format!("{} {}", word(&mut rng), word(&mut rng))));
        }
    }
    out
}

fn answer(m: &MemoryStore, k: &str) -> Option<String> {
    m.query(k).unwrap().value
}

#[test]
fn a_thousand_facts_survive_a_restart_from_tensors_only() {
    let d = dir("thousand");
    let fs = facts(7, 1000);
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    for (k, v) in &fs {
        assert!(matches!(m.teach(k, v, S).unwrap(), Outcome::Stored { .. }), "{k}");
    }
    let sha = m.save(&d).unwrap();
    let before: Vec<_> = fs.iter().map(|(k, _)| m.query(k).unwrap()).collect();
    drop(m);
    let (m, info) = MemoryStore::load(&d).unwrap();
    assert_eq!(info.source, LoadSource::Current);
    assert_eq!(info.checkpoint_sha256, sha);
    let mut correct = 0;
    for ((k, v), b) in fs.iter().zip(&before) {
        let a = m.query(k).unwrap();
        assert_eq!(&a, b, "answer changed across restart for {k}");
        correct += (a.value.as_deref() == Some(v.as_str())) as usize;
    }
    assert_eq!(correct, 1000, "exact-key recall after restart");
    assert_eq!(m.stats().active, 1000);
}

#[test]
fn memory_writes_leave_core_tensors_byte_identical() {
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    let core = |m: &MemoryStore| m.tensors().into_iter().filter(|t| t.name.starts_with("core.")).collect::<Vec<_>>();
    let (c0, mem0) = (core(&m), m.digest("memory."));
    for (k, v) in facts(3, 50) {
        m.teach(&k, &v, S).unwrap();
    }
    m.correct(&facts(3, 1)[0].0, "new value", S).unwrap();
    m.forget(&facts(3, 2)[1].0, S).unwrap();
    m.compact(S);
    m.associate_fast("a b", "c", S).unwrap();
    let c1 = core(&m);
    assert!(!c0.is_empty());
    for (a, b) in c0.iter().zip(&c1) {
        assert_eq!(a.name, b.name);
        assert!(a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits()), "{} changed", a.name);
    }
    assert_ne!(mem0, m.digest("memory."), "memory tensors must change");
    let fresh = MemoryStore::new(MemoryConfig::default()).unwrap();
    assert_eq!(fresh.digest("core."), m.digest("core."));
}

#[test]
fn contradictions_are_recorded_and_never_overwrite() {
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    let Outcome::Stored { slot } = m.teach("sky color", "blue", S).unwrap() else { panic!() };
    assert_eq!(m.teach("Sky COLOR?", "blue", S).unwrap(), Outcome::Unchanged { slot });
    assert_eq!(m.teach("sky color", "green", S).unwrap(), Outcome::Conflict { slot });
    let a = m.query("sky color").unwrap();
    assert_eq!(a.value.as_deref(), Some("blue"));
    assert_eq!(a.contradictions, 1);
    assert!(!a.confirmed());
    assert_eq!(m.confirm("sky color", S).unwrap(), Outcome::Confirmed { slot });
    assert!(m.query("sky color").unwrap().confirmed());
    let ops: Vec<_> = m.history("sky color").unwrap().into_iter().map(|e| e.op).collect();
    assert_eq!(ops, ["add", "unchanged", "contradiction", "confirm"]);
}

#[test]
fn supersession_answers_the_new_value_and_keeps_the_link() {
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    m.teach("mill owner", "dara", S).unwrap();
    let Outcome::Superseded { old, new } = m.correct("mill owner", "pell", S).unwrap() else { panic!() };
    assert_eq!(answer(&m, "mill owner").as_deref(), Some("pell"));
    let t = m.tensors();
    let meta = t.iter().find(|t| t.name == "memory.semantic.meta").unwrap();
    let get = |s: usize, f: usize| meta.data[s * field::COUNT + f];
    assert_eq!(get(old, field::STATUS), SUPERSEDED);
    assert_eq!(get(old, field::SUPERSEDED_BY), new as f32);
    assert_eq!(get(new, field::SUPERSEDES), old as f32);
    assert_eq!(m.stats().superseded, 1);
    assert!(matches!(m.correct("new key", "x", S).unwrap(), Outcome::Stored { .. }));
}

#[test]
fn forgetting_erases_the_fact_and_its_superseded_history() {
    let d = dir("forget");
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    m.teach("kiln home", "vorastel", S).unwrap();
    m.correct("kiln home", "mirabund", S).unwrap();
    m.teach("other key", "kept", S).unwrap();
    let Outcome::Forgotten { slots } = m.forget("kiln home", S).unwrap() else { panic!() };
    assert_eq!(slots.len(), 2, "current and superseded value both erased");
    assert_eq!(answer(&m, "kiln home"), None, "abstains after forget");
    assert_eq!(answer(&m, "other key").as_deref(), Some("kept"));
    assert_eq!(m.forget("kiln home", S).unwrap(), Outcome::Unknown);
    let t = m.tensors();
    for name in ["memory.semantic.keys", "memory.semantic.key_words", "memory.semantic.values"] {
        let x = t.iter().find(|t| t.name == name).unwrap();
        for &s in &slots {
            assert!(x.data[s * x.cols..(s + 1) * x.cols].iter().all(|v| *v == 0.0), "{name} row {s} not erased");
        }
    }
    assert!(m.history("kiln home").unwrap().is_empty(), "episodic key vectors erased");
    m.save(&d).unwrap();
    let (m2, _) = MemoryStore::load(&d).unwrap();
    assert_eq!(answer(&m2, "kiln home"), None);
    assert_eq!(m2.stats().deleted, 2);
    let meta = m2.tensors().into_iter().find(|t| t.name == "memory.semantic.meta").unwrap();
    assert!(slots.iter().all(|&s| meta.data[s * field::COUNT + field::STATUS] == DELETED));
}

#[test]
fn compaction_reclaims_slots_without_changing_answers() {
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    let fs = facts(11, 200);
    for (k, v) in &fs {
        m.teach(k, v, S).unwrap();
    }
    for (k, _) in fs.iter().step_by(4) {
        m.correct(k, "corrected", S).unwrap();
    }
    for (k, _) in fs.iter().skip(1).step_by(5) {
        m.forget(k, S).unwrap();
    }
    let before: Vec<_> = fs.iter().map(|(k, _)| answer(&m, k)).collect();
    let st = m.stats();
    let freed = m.compact(S);
    assert_eq!(freed, st.superseded + st.deleted);
    let after: Vec<_> = fs.iter().map(|(k, _)| answer(&m, k)).collect();
    assert_eq!(before, after);
    let st2 = m.stats();
    assert_eq!((st2.active, st2.superseded, st2.deleted), (st.active, 0, 0));
    assert_eq!(m.active_slots(), (0..st.active).collect::<Vec<_>>(), "active facts packed into the lowest slots");
}

#[test]
fn capacity_is_enforced_and_a_failed_write_changes_nothing() {
    let cfg = MemoryConfig { slots: 8, ..MemoryConfig::default() };
    let mut m = MemoryStore::new(cfg).unwrap();
    let fs = facts(5, 9);
    for (k, v) in &fs[..8] {
        m.teach(k, v, S).unwrap();
    }
    let digest = m.digest("memory.");
    assert_eq!(m.teach(&fs[8].0, &fs[8].1, S), Err(MemoryError::CapacityFull));
    assert_eq!(m.correct(&fs[0].0, "zzz", S), Err(MemoryError::CapacityFull));
    assert_eq!(m.digest("memory."), digest, "a rejected write must not change memory");
    m.forget(&fs[0].0, S).unwrap();
    assert_eq!(m.teach(&fs[8].0, &fs[8].1, S), Err(MemoryError::CapacityFull), "deleted slots need compaction");
    assert_eq!(m.compact(S), 1);
    assert!(matches!(m.teach(&fs[8].0, &fs[8].1, S).unwrap(), Outcome::Stored { .. }));
    for (k, v) in &fs[1..] {
        assert_eq!(answer(&m, k).as_deref(), Some(v.as_str()));
    }
}

#[test]
fn collisions_and_near_duplicate_keys() {
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    m.teach("zorvan color", "amber", S).unwrap();
    // Same normalized key: a collision by design (case/punctuation/order).
    assert!(matches!(m.teach("COLOR, zorvan!", "teal", S).unwrap(), Outcome::Conflict { .. }));
    // Near-duplicate but distinct keys get their own slots.
    assert!(matches!(m.teach("zorvans color", "teal", S).unwrap(), Outcome::Stored { .. }));
    assert_eq!(answer(&m, "zorvan color").as_deref(), Some("amber"));
    assert_eq!(answer(&m, "zorvans color").as_deref(), Some("teal"));
    assert_eq!(answer(&m, "blorp mint"), None, "unrelated key abstains");
    assert_eq!(m.teach("?!", "x", S), Err(MemoryError::EmptyKey));
    assert_eq!(m.teach("k", "", S), Err(MemoryError::EmptyValue));
    assert_eq!(m.teach("k", "a\0b", S), Err(MemoryError::NulInValue));
    assert_eq!(m.teach("k", &"x".repeat(48), S), Err(MemoryError::ValueTooLong { max: 47 }));
    assert_eq!(m.teach("a b c d e f g h i", "x", S), Err(MemoryError::KeyTooLong { max_words: 8 }));
    // Late-interaction verification: a shared entity word is not enough.
    assert_eq!(answer(&m, "zorvan size"), None, "near-miss key abstains");
    assert_eq!(answer(&m, "what is the color of zorvan?").as_deref(), Some("amber"));
}

#[test]
fn facts_live_only_in_checkpoint_tensors() {
    let d = dir("only-tensors");
    let fs = facts(13, 100);
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    for (k, v) in &fs {
        m.teach(k, v, S).unwrap();
    }
    m.save(&d).unwrap();
    m.save(&d).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    files.sort();
    assert_eq!(files, [checkpoint::PREVIOUS, checkpoint::CURRENT], "no side files (journal, json, db)");
    for f in &files {
        let bytes = std::fs::read(d.join(f)).unwrap();
        for (k, v) in &fs {
            for s in [k, v] {
                assert!(!bytes.windows(s.len()).any(|w| w == s.as_bytes()), "plaintext '{s}' in {f}");
            }
        }
    }
    // The answer is computed from the value tensor: zero it and recall is gone.
    let (mut m2, _) = MemoryStore::load(&d).unwrap();
    assert_eq!(answer(&m2, &fs[0].0).as_deref(), Some(fs[0].1.as_str()));
    m2.ablate(Ablation::ZeroValues);
    assert!(fs.iter().all(|(k, v)| answer(&m2, k).as_deref() != Some(v.as_str())));
}

#[test]
fn shuffled_keys_break_the_key_value_binding() {
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    let fs = facts(17, 200);
    for (k, v) in &fs {
        m.teach(k, v, S).unwrap();
    }
    m.ablate(Ablation::ShuffleKeys(1));
    let right = fs.iter().filter(|(k, v)| answer(&m, k).as_deref() == Some(v.as_str())).count();
    assert!(right < 10, "{right} still right after shuffling keys");
    m.ablate(Ablation::ZeroKeys);
    assert!(fs.iter().all(|(k, _)| answer(&m, k).is_none()), "zero keys -> abstain");
}

#[test]
fn corrupt_and_interrupted_checkpoints_recover_the_last_valid_one() {
    let d = dir("corrupt");
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    m.teach("a key", "one", S).unwrap();
    m.save(&d).unwrap();
    m.teach("b key", "two", S).unwrap();
    m.save(&d).unwrap();
    // Bit flip in the current file -> previous checkpoint ("a" only).
    let cur = d.join(checkpoint::CURRENT);
    let mut bytes = std::fs::read(&cur).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x10;
    std::fs::write(&cur, &bytes).unwrap();
    let (r, info) = MemoryStore::load(&d).unwrap();
    assert_eq!(info.source, LoadSource::Previous);
    assert!(info.current_error.unwrap().contains("mismatch"));
    assert_eq!(answer(&r, "a key").as_deref(), Some("one"));
    assert_eq!(answer(&r, "b key"), None);
    // Both corrupt -> fail safely, nothing loaded.
    std::fs::write(d.join(checkpoint::PREVIOUS), &bytes[..100]).unwrap();
    assert!(matches!(MemoryStore::load(&d), Err(MemoryError::Checkpoint(CheckpointError::NoValidCheckpoint { .. }))));
    assert!(matches!(MemoryStore::open(&d, MemoryConfig::default()), Err(MemoryError::Checkpoint(CheckpointError::NoValidCheckpoint { .. }))));

    // Interrupted writes always leave the last committed checkpoint loadable.
    for fault in [Fault::DuringTempWrite, Fault::BeforeRename, Fault::BetweenRenames] {
        let d = dir(&format!("fault-{fault:?}"));
        let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
        m.teach("a key", "one", S).unwrap();
        let committed = m.save(&d).unwrap();
        m.teach("b key", "two", S).unwrap();
        assert!(m.save_with_fault(&d, fault).is_err());
        let (r, info) = MemoryStore::load(&d).unwrap();
        assert_eq!(info.checkpoint_sha256, committed, "{fault:?}");
        assert_eq!(answer(&r, "a key").as_deref(), Some("one"));
        assert_eq!(answer(&r, "b key"), None, "{fault:?}: uncommitted write must not appear");
        // The next save succeeds and commits normally.
        m.save(&d).unwrap();
        let (r, info) = MemoryStore::load(&d).unwrap();
        assert_eq!((info.source, answer(&r, "b key").as_deref()), (LoadSource::Current, Some("two")));
    }
    assert!(matches!(MemoryStore::load(&dir("empty")), Err(MemoryError::Checkpoint(CheckpointError::NotFound))));
}

#[test]
fn a_tampered_core_tensor_is_rejected() {
    let m = MemoryStore::new(MemoryConfig::default()).unwrap();
    let mut t = m.tensors();
    t[0].data[0] += 1.0;
    let bytes = checkpoint::encode(&serde_json::json!({ "config": m.config() }), &t);
    let (meta, tensors) = checkpoint::decode(&bytes).unwrap();
    assert!(matches!(MemoryStore::from_checkpoint(&meta, tensors), Err(MemoryError::CoreMismatch(n)) if n == "core.embed"));
    // Compiled W_PROG constants are frozen the same way.
    let mut t = m.tensors();
    let i = t.iter().position(|t| t.name == "prog.memory-read.read_threshold").unwrap();
    t[i].data[0] = 0.0;
    let bytes = checkpoint::encode(&serde_json::json!({ "config": m.config() }), &t);
    let (meta, tensors) = checkpoint::decode(&bytes).unwrap();
    assert!(matches!(MemoryStore::from_checkpoint(&meta, tensors), Err(MemoryError::CoreMismatch(n)) if n == "prog.memory-read.read_threshold"));
}

#[test]
fn every_update_is_logged_and_bound_to_the_saved_checkpoint() {
    let d = dir("events");
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    m.teach("x key", "1", "stream-a").unwrap();
    m.teach("x key", "2", "stream-a").unwrap();
    m.correct("x key", "3", "stream-b").unwrap();
    m.associate_fast("y key", "4", "stream-b").unwrap();
    assert!(m.events().iter().all(|e| e.checkpoint_sha256.is_none()));
    let sha = m.save(&d).unwrap();
    assert_eq!(sha, checkpoint::file_sha256(&std::fs::read(d.join(checkpoint::CURRENT)).unwrap()));
    let ops: Vec<_> = m.events().iter().map(|e| (e.op, e.signal, e.input_stream.as_str())).collect();
    assert_eq!(
        ops,
        [
            ("add", "user_statement", "stream-a"),
            ("contradiction", "user_statement", "stream-a"),
            ("supersede", "user_correction", "stream-b"),
            ("fast_associate", "user_statement", "stream-b")
        ]
    );
    for (i, e) in m.events().iter().enumerate() {
        assert_eq!(e.seq, i as u64 + 1);
        assert_eq!(e.checkpoint_sha256.as_deref(), Some(sha.as_str()));
        assert!(!e.changed_tensors.is_empty() && e.rows_sha256.len() == 64);
        assert!(e.changed_tensors.iter().all(|t| t.starts_with("memory.")), "memory writes never touch core tensors");
    }
}

#[test]
fn every_checkpoint_tensor_belongs_to_a_declared_weight_group() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/specs/weight-groups.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let declared: BTreeSet<String> = v["groups"].as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap().to_string()).collect();
    let m = MemoryStore::new(MemoryConfig::default()).unwrap();
    for t in m.tensors() {
        assert!(declared.contains(&t.group), "{} uses undeclared group {}", t.name, t.group);
    }
}

#[test]
fn fast_weights_recall_recent_associations_and_are_short_lived() {
    let mut m = MemoryStore::new(MemoryConfig::default()).unwrap();
    m.associate_fast("door code", "amber", S).unwrap();
    assert_eq!(m.recall_fast("door code").unwrap().bytes, b"amber");
    for (k, v) in facts(19, 400) {
        m.associate_fast(&k, &v, S).unwrap();
    }
    assert_ne!(m.recall_fast("door code").unwrap().bytes, b"amber", "decayed after 400 later writes");
    assert_eq!(answer(&m, "door code"), None, "fast associations are not semantic beliefs");
}

#[test]
fn the_default_read_circuit_is_the_shipped_pinned_spec() {
    use zerolm_memory::read_circuit;
    let shipped = zerolm_wsp::shipped(read_circuit::ID).unwrap();
    let cfg = zerolm_memory::MemoryConfig::default();
    let built = read_circuit::build(&cfg, 256, zerolm_engine::bootstrap::EMBED_DIM);
    assert_eq!(built.sha256(), shipped.sha256(), "MemoryConfig::default() must compile to model/prog/memory-read.json");
    let store = zerolm_memory::MemoryStore::new(cfg).unwrap();
    assert_eq!(store.read_circuit().sha256(), shipped.sha256());
    // W_PROG constants are checkpoint tensors in their own group.
    let prog: Vec<_> = store.tensors().into_iter().filter(|t| t.name.starts_with("prog.memory-read.")).collect();
    assert_eq!(prog.len(), shipped.tensors.len());
    assert!(prog.iter().all(|t| t.group == "w_prog"));
}

#[test]
fn the_default_encoder_circuit_is_the_shipped_pinned_spec() {
    use zerolm_memory::read_circuit;
    let shipped = zerolm_wsp::shipped(read_circuit::ENCODE_ID).unwrap();
    let cfg = zerolm_memory::MemoryConfig::default();
    assert_eq!(read_circuit::build_encoder(&cfg).sha256(), shipped.sha256(), "MemoryConfig::default() must compile to model/prog/memory-encode.json");
    let store = zerolm_memory::MemoryStore::new(cfg).unwrap();
    assert_eq!(store.encode_circuit().sha256(), shipped.sha256());
    let prog: Vec<_> = store.tensors().into_iter().filter(|t| t.name.starts_with("prog.memory-encode.")).collect();
    assert_eq!(prog.len(), shipped.tensors.len());
    assert!(prog.iter().all(|t| t.group == "w_prog"));
}

#[test]
fn over_long_text_is_refused_and_changes_nothing() {
    let mut m = zerolm_memory::MemoryStore::new(zerolm_memory::MemoryConfig::default()).unwrap();
    let before = m.checkpoint_bytes();
    let long = "a".repeat(zerolm_memory::read_circuit::TEXT_BYTES - 1);
    assert!(matches!(m.teach(&long, "v", "test"), Err(zerolm_memory::MemoryError::TextTooLong { .. })));
    assert!(matches!(m.query(&long), Err(zerolm_memory::MemoryError::TextTooLong { .. })));
    assert_eq!(m.checkpoint_bytes(), before);
}
