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
