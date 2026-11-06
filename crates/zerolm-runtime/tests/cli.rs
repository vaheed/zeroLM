use std::io::Write;
use std::process::{Command, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_zerolm")
}

fn run(input: &[u8], extra: &[&str]) -> (String, String, bool) {
    let mut c = Command::new(bin()).arg("run").args(extra).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    c.stdin.take().unwrap().write_all(input).unwrap();
    let o = c.wait_with_output().unwrap();
    (String::from_utf8_lossy(&o.stdout).into(), String::from_utf8_lossy(&o.stderr).into(), o.status.success())
}

#[test]
fn clean_model_accepts_protocol_traces_and_exits_without_touching_weights() {
    let trace = std::env::temp_dir().join(format!("zerolm-cli-trace-{}.jsonl", std::process::id()));
    let input = b"teach sky => blue\nask sky\nconfirm sky\nask sky\nforget sky\nask sky\n";
    let (out, err, ok) = run(input, &["--trace", trace.to_str().unwrap()]);
    assert!(ok, "{err}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[1], "sky => blue (unconfirmed)");
    assert_eq!(lines[3], "sky => blue (confirmed)");
    assert_eq!(lines[5], "unknown sky");
    assert!(err.contains("unchanged=true"));
    let events: Vec<serde_json::Value> = std::fs::read_to_string(&trace).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(events.len(), 6);
    assert_eq!(events[0]["command"], "teach");
    assert_eq!(events[0]["key_span"]["start"], 6);
    assert_eq!(events[0]["source"], "ledger");
    assert!(events.iter().all(|e| e["weights_sha256"] == events[0]["weights_sha256"]));
}

#[test]
fn arbitrary_bytes_from_the_first_run_do_not_crash() {
    let mut input: Vec<u8> = (0..=255u8).collect();
    input.extend_from_slice(b"\n\xff\xfe\xe2\x82\nteach a => b\n");
    let (_, err, ok) = run(&input, &[]);
    assert!(ok, "{err}");
}

#[test]
fn info_prints_the_pinned_initialization_hash() {
    let o = Command::new(bin()).arg("info").output().unwrap();
    let s = String::from_utf8_lossy(&o.stdout);
    assert!(s.contains("efb8f0e1b8252cc4e53585849ec1b3ab7533736a1d920453e2aff1e7790f95b6"), "{s}");
    assert!(s.contains("no English understanding"));
}

#[test]
fn bad_usage_is_rejected() {
    assert_eq!(Command::new(bin()).output().unwrap().status.code(), Some(2));
    assert_eq!(Command::new(bin()).args(["run", "--seed", "x"]).output().unwrap().status.code(), Some(2));
}

fn memory(dir: &std::path::Path, input: &[u8], extra: &[&str]) -> (String, String, bool) {
    let mut c = Command::new(bin())
        .args(["memory", "--dir", dir.to_str().unwrap()])
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    c.stdin.take().unwrap().write_all(input).unwrap();
    let o = c.wait_with_output().unwrap();
    (String::from_utf8_lossy(&o.stdout).into(), String::from_utf8_lossy(&o.stderr).into(), o.status.success())
}

#[test]
fn memory_facts_survive_a_process_restart() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("zerolm-cli-memory");
    let _ = std::fs::remove_dir_all(&dir);
    let trace = dir.with_extension("trace.jsonl");
    let (out, err, ok) = memory(&dir, b"teach sky color => blue\nteach mill owner => dara\nconfirm sky color\nteach sky color => green\n", &["--trace", trace.to_str().unwrap()]);
    assert!(ok, "{err}");
    assert_eq!(out.lines().collect::<Vec<_>>(), ["noted sky color (unconfirmed)", "noted mill owner (unconfirmed)", "confirmed sky color", "conflict: sky color already has a different value; use correct"]);
    assert!(err.contains("starting from the clean model") && err.contains("core unchanged=true"), "{err}");
    let events: Vec<serde_json::Value> = std::fs::read_to_string(&trace).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(events[0]["memory_events"][0]["op"], "add");
    assert_eq!(events[0]["memory_events"][0]["signal"], "user_statement");
    assert!(events.iter().all(|e| e["checkpoint_sha256"].as_str().map(|s| s.len()) == Some(64)));

    // A new process: answers come from the checkpoint tensors only.
    let (out, err, ok) = memory(&dir, b"ask what is the color of the sky?\nask mill owner\nask sky size\ncorrect mill owner => pell\nforget sky color\n", &[]);
    assert!(ok, "{err}");
    assert!(err.contains("loaded Current checkpoint"), "{err}");
    assert_eq!(out.lines().collect::<Vec<_>>(), ["what is the color of the sky? => blue (confirmed)", "mill owner => dara (unconfirmed)", "unknown sky size", "corrected mill owner (unconfirmed)", "forgot sky color"]);

    let (out, _, ok) = memory(&dir, b"ask sky color\nask mill owner\n", &[]);
    assert!(ok);
    assert_eq!(out.lines().collect::<Vec<_>>(), ["unknown sky color", "mill owner => pell (unconfirmed)"]);
    let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    files.sort();
    assert_eq!(files, ["memory.prev.zlmc", "memory.zlmc"], "only checkpoint files hold memory");
    // The session-ledger runtime knows nothing of it.
    let (out, _, _) = run(b"ask mill owner\n", &[]);
    assert_eq!(out.trim(), "unknown mill owner");
}

#[test]
fn reason_answers_derived_questions_with_provenance_across_processes() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("zerolm-cli-reason");
    let _ = std::fs::remove_dir_all(&dir);
    let reason = |input: &[u8]| {
        let mut c = Command::new(bin())
            .args(["reason", "--dir", dir.to_str().unwrap()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        c.stdin.take().unwrap().write_all(input).unwrap();
        let o = c.wait_with_output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).into_owned()
    };
    let out = reason(b"every dog is a mammal\nall mammals are animals\nrex is a dog\nit is in kennel\nthe cat sat on the mat\nis rex an animal?\n");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[..5], ["noted", "noted", "noted", "noted", "not understood (outside the bounded language)"]);
    assert_eq!(lines[5], "yes (derived: class_inclusion; 3 premises; confidence 1.00)");
    let out = reason(b"is rex in kennel?\nis rex a fish?\ntom has 3 apples\nann has 2 more apples than tom\nhow many apples does ann have?\n");
    assert_eq!(out.lines().collect::<Vec<_>>(), ["yes", "unknown", "noted", "noted", "5 (derived: arithmetic, direct; 2 premises; confidence 1.00)"]);
}

fn with_env(args: &[&str], env: &[(&str, &str)], input: &[u8]) -> (String, String, bool) {
    // Start from a known state: the Docker image sets ZEROLM_VERBOSE=1 by default.
    let mut c = Command::new(bin()).args(args).env_remove("ZEROLM_VERBOSE").envs(env.iter().copied()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    c.stdin.take().unwrap().write_all(input).unwrap();
    let o = c.wait_with_output().unwrap();
    (String::from_utf8_lossy(&o.stdout).into(), String::from_utf8_lossy(&o.stderr).into(), o.status.success())
}

#[test]
fn verbose_shows_internals_on_stderr_and_leaves_replies_unchanged() {
    let input = b"rex is a dog\nevery dog is an animal\nis rex an animal?\nblorp\n";
    let (plain, plain_err, ok) = with_env(&["reason"], &[], input);
    assert!(ok, "{plain_err}");
    assert!(!plain_err.contains("premise"));
    let (out, err, ok) = with_env(&["reason", "--verbose"], &[], input);
    assert!(ok, "{err}");
    assert_eq!(out, plain, "verbose must not change stdout");
    assert!(err.contains("[0] > rex is a dog"), "{err}");
    assert!(err.contains("relation_write") && err.contains("memory.relations.slots"), "{err}");
    assert!(err.contains("class_inclusion"), "{err}");
    assert!(err.contains("premise     slot 0 <- \"rex is a dog\""), "{err}");
    assert!(err.contains("no match in the bounded grammar"), "{err}");
    // The environment variable turns it on too.
    let (out, err, ok) = with_env(&["reason"], &[("ZEROLM_VERBOSE", "1")], input);
    assert!(ok && out == plain && err.contains("premise"), "{err}");
}

#[test]
fn verbose_memory_reports_tensor_writes_and_retrieval_scores() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("zerolm-cli-verbose-memory");
    let _ = std::fs::remove_dir_all(&dir);
    let (out, err, ok) = with_env(&["memory", "--dir", dir.to_str().unwrap(), "-v"], &[], b"teach sky => blue\nask sky\n");
    assert!(ok, "{err}");
    assert_eq!(out.lines().collect::<Vec<_>>(), ["noted sky (unconfirmed)", "sky => blue (unconfirmed)"]);
    assert!(err.contains("rule=memory_write") && err.contains("memory.semantic.keys"), "{err}");
    assert!(err.contains("saved atomically") && err.contains("attention score"), "{err}");
}

#[test]
fn documented_examples_give_the_documented_answers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let read = |f: &str| std::fs::read(root.join(f)).unwrap();
    let (out, err, ok) = with_env(&["reason"], &[], &read("reasoning-more.txt"));
    assert!(ok, "{err}");
    let answers: Vec<&str> = out.lines().filter(|l| *l != "noted").collect();
    assert_eq!(answers.len(), 6, "{out}");
    for (got, want) in answers.iter().zip(["yes", "no", "yes", "cal", "yes", "yes"]) {
        assert!(got.starts_with(want), "{got} vs {want}\n{out}");
    }
    let (out, _, ok) = with_env(&["reason"], &[], &read("reasoning.txt"));
    assert!(ok && out.contains("5 (derived: arithmetic"), "{out}");
    let (out, _, ok) = with_env(&["run"], &[], &read("protocol.txt"));
    assert!(ok && out.lines().last() == Some("unknown sky"), "{out}");
}
