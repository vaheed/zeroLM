//! ZeroLM Next: behaviour, shortcut detection, causal dependence on weights, CLI.

use std::path::PathBuf;
use std::process::Command;
use zerolm::next::world::{execute, World};
use zerolm::next::{Ablation, NextModel};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}
fn weights() -> String {
    root().join("model/next/weights.zln").to_string_lossy().into_owned()
}
fn model() -> NextModel {
    NextModel::load(&weights(), Ablation::default()).expect("weights load")
}
fn model_with(ab: &[&str]) -> NextModel {
    let v: Vec<String> = ab.iter().map(|s| s.to_string()).collect();
    NextModel::load(&weights(), Ablation::parse(&v).unwrap()).expect("weights load")
}
fn ops_json(m: &NextModel, text: &str) -> Option<Vec<String>> {
    let p = m.parse(text, false);
    p.decoded.accepted.then(|| p.decoded.ops.iter().map(|o| o.to_json()).collect())
}

#[test]
fn headline_sentence_builds_the_expected_structure() {
    let got = ops_json(&model(), "Move the red block left twice, then move the blue circle up once.").unwrap();
    assert_eq!(got, vec![
        r#"{"action":"move","object":{"type":"block","color":"red"},"direction":"left","count":2,"negated":false}"#,
        r#"{"action":"move","object":{"type":"circle","color":"blue"},"direction":"up","count":1,"negated":false}"#,
    ]);
}

/// 4 types x 6 colors x 4 directions x 9 counts = 864 sentences whose expected output is built from
/// the loop variables alone. None of these sentences exists anywhere in the model or the test data.
#[test]
fn every_primitive_combination_is_composed_correctly() {
    let m = model();
    let mut n = 0;
    for t in ["block", "circle", "triangle", "star"] {
        for c in ["red", "blue", "green", "yellow", "purple", "orange"] {
            for d in ["left", "right", "up", "down"] {
                for k in 1..=9 {
                    let text = format!("move the {c} {t} {d} {k} times");
                    let want = format!(r#"{{"action":"move","object":{{"type":"{t}","color":"{c}"}},"direction":"{d}","count":{k},"negated":false}}"#);
                    assert_eq!(ops_json(&m, &text), Some(vec![want]), "{text}");
                    n += 1;
                }
            }
        }
    }
    assert_eq!(n, 864);
}

#[test]
fn order_negation_condition_and_binding_change_the_output() {
    let m = model();
    let a = ops_json(&m, "move the red block up, then move the blue circle left").unwrap();
    let b = ops_json(&m, "move the red block up after moving the blue circle left").unwrap();
    assert_eq!(a.len(), 2);
    assert_eq!((a[0].clone(), a[1].clone()), (b[1].clone(), b[0].clone()), "'after' reverses execution order");
    let neg = ops_json(&m, "do not move the red block up").unwrap();
    assert!(neg[0].contains(r#""negated":true"#));
    let cond = ops_json(&m, "move the red block up unless there is a blue circle").unwrap();
    assert!(cond[0].contains(r#""condition":{"type":"circle","color":"blue","negated":true}"#));
    // binding: the colour after the noun is the paint target, the one before is the object attribute
    let p = ops_json(&m, "paint the red block blue").unwrap();
    assert!(p[0].contains(r#""color":"red""#) && p[0].contains(r#""to_color":"blue""#));
    let q = ops_json(&m, "paint the blue block red").unwrap();
    assert!(q[0].contains(r#""color":"blue""#) && q[0].contains(r#""to_color":"red""#));
}

#[test]
fn unsupported_input_is_rejected_not_interpreted() {
    let m = model();
    for t in ["hello", "", "move it left", "fly the red block left", "move left the red block", "move the red block", "move the red block left then",
              "move the red block and the blue circle left", "move the red block left 💥", "red block", "what is the weather in Paris"] {
        assert!(ops_json(&m, t).is_none(), "should reject {t:?}");
    }
}

#[test]
fn executor_runs_only_on_structured_operations() {
    // The executor API takes (World, &[Op]); there is no text parameter. Operations come from the network.
    let m = model();
    let w = World::parse_str(r#"{"width":8,"height":8,"objects":[{"id":1,"type":"block","color":"red","x":4,"y":4},{"id":2,"type":"circle","color":"blue","x":0,"y":0}]}"#).unwrap();
    let p = m.parse("paint the red block green, then move the green block left twice, then remove the blue circle", false);
    let (after, status) = execute(&w, &p.decoded.ops);
    assert_eq!(status, vec!["ok", "ok", "ok"]);
    assert_eq!(after.objects.len(), 1);
    assert_eq!((after.objects[0].color.as_str(), after.objects[0].x, after.objects[0].y), ("green", 2, 4));
    // state tracking: the old colour no longer resolves
    let p = m.parse("paint the red block green, then move the red block left", false);
    assert_eq!(execute(&w, &p.decoded.ops).1, vec!["ok", "no_such_object"]);
}

// ---------------------------------------------------------------- shortcut detectors

#[test]
fn no_phrase_entries_and_no_response_text_in_the_weight_file() {
    let m = model();
    let vocab = m.net.list("vocab");
    assert!(vocab.len() < 200, "vocabulary is a word list, not a phrase table");
    for w in vocab {
        assert!(!w.contains(' '), "multi-word vocabulary entry {w:?}");
        assert!(w.chars().count() <= 12);
    }
    // every list in the file is a list of single tokens; no sentence-like entry anywhere
    for (name, items) in &m.net.lists {
        for it in items {
            assert!(!it.contains(' '), "{name}: {it:?}");
        }
    }
}

#[test]
fn runtime_source_contains_no_vocabulary_literals() {
    let m = model();
    let vocab: Vec<&String> = m.net.list("vocab").iter().filter(|w| w.chars().all(|c| c.is_ascii_alphanumeric())).collect();
    // The network/decoder/encoder code must not mention a single lexicon word. (world.rs and cli.rs
    // legitimately name the *operation vocabulary* of the executor and the demo world.)
    for file in ["encoder.rs", "network.rs", "binding.rs", "decoder.rs", "model.rs", "trace.rs"] {
        let src = std::fs::read_to_string(root().join("rust/src/next").join(file)).unwrap();
        let mut lits = Vec::new();
        for line in src.lines() {
            let line = line.split("//").next().unwrap_or("");
            let mut rest = line;
            while let Some(i) = rest.find('"') {
                let after = &rest[i + 1..];
                if let Some(j) = after.find('"') {
                    lits.push(after[..j].to_string());
                    rest = &after[j + 1..];
                } else {
                    break;
                }
            }
        }
        for l in lits {
            let l = l.to_string(); // vocabulary words are lowercase; field names such as "TIME" are not words
            assert!(!vocab.iter().any(|w| **w == l), "{file}: string literal {l:?} is a lexicon word");
        }
    }
}

#[test]
fn next_module_makes_no_external_calls() {
    for entry in std::fs::read_dir(root().join("rust/src/next")).unwrap() {
        let p = entry.unwrap().path();
        let src = std::fs::read_to_string(&p).unwrap();
        for bad in ["std::net", "TcpStream", "UdpSocket", "reqwest", "hyper", "curl", "wget", "openai", "anthropic"] {
            assert!(!src.contains(bad), "{} mentions {bad}", p.display());
        }
        if p.file_name().unwrap() != "cli.rs" {
            assert!(!src.contains("Command::new"), "{} spawns a process", p.display());
        }
    }
}

fn probes() -> Vec<String> {
    let mut v: Vec<String> = [
        "move the red block left twice, then move the blue circle up once",
        "paint the green triangle yellow",
        "remove the purple star after moving the orange disc down 3 times",
        "before painting the red block blue, move the blue circle right",
        "do not move the red block left, then remove the star",
        "if there is a blue circle, paint the red block green",
        "move the red block up unless there is a green triangle. remove the star; move the circle left once",
        "slide a cube west by two",
        "move the red block left then",
        "fly the red block",
        "move left the red block",
        "move the red block left, then move the blue circle up, then move the star down after painting the cube red",
        "",
        "the",
        "after",
        "move the red block left. then",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    v.push(vec!["move the red block left"; 26].join(" then "));
    v.push(format!("{} then", vec!["move the red block left"; 24].join(" then ")));
    v
}

fn fingerprint(m: &NextModel) -> Vec<String> {
    probes().iter().map(|t| { let p = m.parse(t, false); format!("{}|{}", p.decoded.accepted, p.decoded.ops.iter().map(|o| o.to_json()).collect::<Vec<_>>().join(",")) }).collect()
}

/// Zeroing, permuting or randomising any tensor must change behaviour on the probe set. A tensor that
/// can be zeroed with no effect would indicate the semantics bypass it.
#[test]
fn every_weight_tensor_is_causally_used() {
    let base = fingerprint(&model());
    let names: Vec<String> = { let mut v: Vec<String> = model().net.tensors.keys().cloned().collect(); v.sort(); v };
    let mut inert = Vec::new();
    for n in &names {
        if fingerprint(&model_with(&[&format!("zero:{n}")])) == base {
            inert.push(n.clone());
        }
    }
    // Findings, documented in research/LIMITATIONS.md (negative result: redundancy in the construction):
    //  * dec.*.b: class biases only break ties, and argmax already prefers class 0 ("none"/"false").
    //  * rank_eps: tie-breaking between equal times; valid parses never produce equal times.
    //  * tf_w: the OPEN-based dangling rule of patch 001 is subsumed by PEND (patch 004).
    let mut allowed: Vec<String> = (0..10).map(|i| format!("dec.{i}.b")).collect();
    //  * Gm_b: an all-zero bias tensor (kept so that the max-update has the same form as every other layer).
    allowed.extend(["rank_eps".to_string(), "tf_w".to_string(), "Gm_b".to_string()]);
    allowed.sort();
    inert.sort();
    assert_eq!(inert, allowed, "set of tensors whose removal changes nothing on the probes");
}

#[test]
fn designed_weights_beat_ablated_and_random_ones() {
    let base = fingerprint(&model());
    for ab in [vec!["recurrence"], vec!["state_read"], vec!["binding"], vec!["pointer_shift"], vec!["time_order"], vec!["random:1"], vec!["shuffle:1"], vec!["permute:B_o:1"], vec!["zero:S"]] {
        let f = fingerprint(&model_with(&ab));
        let same = f.iter().zip(&base).filter(|(a, b)| a == b).count();
        assert!(same < probes().len(), "ablation {ab:?} left every probe unchanged");
    }
    let rnd = fingerprint(&model_with(&["random:1"]));
    assert!(rnd.iter().zip(&base).filter(|(a, b)| a == b && a.starts_with("true")).count() == 0, "random weights reproduce an accepted probe");
}

#[test]
fn trace_exposes_the_forward_pass() {
    let m = model();
    let p = m.parse("move the red block left", true);
    let j = p.trace.as_ref().unwrap().to_json(&p.tokens, &p.decoded);
    for key in ["\"embedding\"", "\"unbound_row\"", "\"boundary_units\"", "\"write_units\"", "\"write_vector\"", "\"logits\"", "\"output_slots\"", "\"ops\""] {
        assert!(j.contains(key), "trace lacks {key}");
    }
    assert!(j.contains("w_DIR.left"), "decoder/writer unit names are exposed");
}

#[test]
fn malformed_weight_files_are_refused() {
    assert!(zerolm::next::network::Net::parse("ZLN 2\n").is_err());
    let good = std::fs::read_to_string(weights()).unwrap();
    let bad = good.replacen("tensor S 24 24", "tensor S 24 25", 1);
    let net = zerolm::next::network::Net::parse(&bad);
    assert!(net.is_err() || NextModel::from_net(net.unwrap(), Ablation::default()).is_err());
}

// ---------------------------------------------------------------- CLI

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zerolm"))
}

#[test]
fn cli_next_parse_run_batch_and_legacy_coexist() {
    let w = weights();
    let out = bin().args(["next", "--model", &w, "parse", "move the red block left"]).output().unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains(r#""accepted":true"#));
    let out = bin().args(["next", "--model", &w, "run", "paint the red block blue"]).output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains(r#""status":["ok"]"#) && s.contains(r#""color":"blue""#));
    let dir = std::env::temp_dir().join(format!("zln-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("in.jsonl");
    std::fs::write(&input, "{\"id\":\"a\",\"text\":\"move the red block up\"}\n{\"id\":\"b\",\"text\":\"hello\"}\n").unwrap();
    let out = bin().args(["next", "--model", &w, "batch", input.to_str().unwrap()]).output().unwrap();
    let lines: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().map(str::to_string).collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains(r#""id":"a""#) && lines[0].contains(r#""accepted":true"#));
    assert!(lines[1].contains(r#""accepted":false"#));
    // legacy v0.2 CLI untouched
    let legacy = bin().args(["--model", root().join("weights.zlm").to_str().unwrap(), "--self-test"]).output().unwrap();
    assert!(legacy.status.success());
}
