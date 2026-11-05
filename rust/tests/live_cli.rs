//! End-to-end tests of `zerolm live`: the real binary, the real `weights.zlm`,
//! real processes and real life files. Each test works in its own temp dir.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_zerolm");

fn model() -> String {
    format!("{}/../weights.zlm", env!("CARGO_MANIFEST_DIR"))
}

fn sandbox(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zerolm-it-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Run `zerolm <args>` in `dir`, feeding `stdin`, waiting for exit.
fn run(dir: &Path, args: &[&str], stdin: &str) -> Output {
    let mut child = Command::new(BIN)
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // A process that rejects its arguments exits without reading stdin, so the write may hit
    // a closed pipe. That is expected, not a failure of the test.
    let _ = child.stdin.take().unwrap().write_all(stdin.as_bytes());
    child.wait_with_output().unwrap()
}

fn text(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// `zerolm live --jsonl` with the real model; returns one output line per input line.
fn jsonl(dir: &Path, extra: &[&str], lines: &[&str]) -> Vec<String> {
    let m = model();
    let mut args = vec!["live", "--jsonl", "--model", m.as_str()];
    args.extend_from_slice(extra);
    let o = run(dir, &args, &(lines.join("\n") + "\n"));
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
    text(&o).lines().map(str::to_string).collect()
}

fn reply(line: &str) -> String {
    let key = "\"reply\":\"";
    let start = line.find(key).unwrap_or_else(|| panic!("no reply in {line}")) + key.len();
    let mut out = String::new();
    let mut it = line[start..].chars();
    while let Some(c) = it.next() {
        match c {
            '"' => break,
            '\\' => match it.next() {
                Some('n') => out.push('\n'),
                Some(o) => out.push(o),
                None => break,
            },
            c => out.push(c),
        }
    }
    out
}

fn intent(line: &str) -> Option<String> {
    let rest = &line[line.find("\"intent\":").expect("intent field") + 9..];
    if rest.starts_with("null") {
        None
    } else {
        Some(rest[1..rest[1..].find('"').unwrap() + 1].to_string())
    }
}

fn number(line: &str, key: &str) -> u64 {
    let rest = &line[line.find(&format!("\"{key}\":")).unwrap() + key.len() + 3..];
    rest.chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap()
}

#[test]
fn facts_survive_a_restart() {
    let d = sandbox("restart");
    jsonl(&d, &["--tag", "t"], &["remember my favourite colour is green"]);
    let out = jsonl(&d, &["--tag", "t"], &["what is my favourite colour"]);
    assert_eq!(reply(&out[0]), "my favourite colour is green.");
    assert_eq!(number(&out[0], "age"), 2, "age counts turns across restarts");
}

#[test]
fn tags_are_isolated() {
    let d = sandbox("tags");
    jsonl(&d, &["--tag", "alice"], &["remember my name is Alice"]);
    jsonl(&d, &["--tag", "bob"], &["remember my name is Bob"]);
    let a = jsonl(&d, &["--tag", "alice"], &["what is my name"]);
    let b = jsonl(&d, &["--tag", "bob"], &["what is my name"]);
    assert_eq!(reply(&a[0]), "my name is Alice.");
    assert_eq!(reply(&b[0]), "my name is Bob.");
    assert!(d.join("life.alice.zlm").exists() && d.join("life.bob.zlm").exists());
}

#[test]
fn second_live_process_on_the_same_tag_is_refused_until_the_first_exits() {
    let d = sandbox("lock");
    let m = model();
    let mut first = Command::new(BIN)
        .args(["live", "--model", &m, "--tag", "busy"])
        .current_dir(&d)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // The banner is printed after the lock is taken.
    let mut banner = String::new();
    BufReader::new(first.stdout.take().unwrap()).read_line(&mut banner).unwrap();
    assert!(banner.contains("ZeroLM live"), "{banner}");

    let second = run(&d, &["live", "--model", &m, "--tag", "busy"], "hello\n");
    assert_eq!(second.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&second.stderr).contains("in use"));

    let other = run(&d, &["live", "--model", &m, "--tag", "free"], "/quit\n");
    assert!(other.status.success(), "a different tag must not be blocked");

    drop(first.stdin.take()); // EOF ends the first session
    assert!(first.wait().unwrap().success());
    let third = run(&d, &["live", "--model", &m, "--tag", "busy"], "/quit\n");
    assert!(third.status.success(), "lock must be released on exit");
}

#[test]
fn a_killed_session_does_not_leave_the_tag_locked() {
    let d = sandbox("kill");
    let m = model();
    let mut first = Command::new(BIN)
        .args(["live", "--model", &m, "--tag", "crash"])
        .current_dir(&d)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut banner = String::new();
    BufReader::new(first.stdout.take().unwrap()).read_line(&mut banner).unwrap();
    first.kill().unwrap(); // SIGKILL: no destructors run
    first.wait().unwrap();
    assert!(d.join("life.crash.zlm.lock").exists(), "kill leaves the lock file behind");
    let again = run(&d, &["live", "--model", &m, "--tag", "crash"], "/quit\n");
    assert!(again.status.success(), "{}", String::from_utf8_lossy(&again.stderr));
}

#[test]
fn jsonl_contract() {
    let d = sandbox("jsonl");
    let out = jsonl(
        &d,
        &["--tag", "j"],
        &[
            "hello",
            r#"{"text":"remember my cat is \"Tom\""}"#,
            r#"{"nope":1}"#,
            "",
            r#"{"text":"what is my cat"}"#,
            "{\"text\":\"caf\\u00e9 \\ud83d\\ude00\"}",
        ],
    );
    assert_eq!(out.len(), 5, "blank lines are skipped, one line out per line in: {out:?}");
    assert_eq!(intent(&out[0]).as_deref(), Some("greeting"));
    assert_eq!(number(&out[0], "age"), 1);
    assert_eq!(intent(&out[1]), None); // a command, not a classified intent
    assert_eq!(number(&out[1], "memories"), 1);
    assert!(out[2].starts_with("{\"error\":"), "{}", out[2]);
    assert_eq!(reply(&out[3]), "my cat is \"Tom\".");
    assert_eq!(intent(&out[4]).as_deref(), Some("unknown")); // unicode is outside the vocabulary
    for line in &out {
        assert!(line.starts_with('{') && line.ends_with('}'), "{line}");
    }
}

#[test]
fn teaching_persists_and_never_touches_the_base_weights() {
    let d = sandbox("teach");
    let before = fs::read(model()).unwrap();
    jsonl(&d, &["--tag", "t"], &["teach ahoy matey as greeting", "bad", "good"]);
    let out = jsonl(&d, &["--tag", "t"], &["ahoy matey", "what is the speed of light"]);
    assert_eq!(intent(&out[0]).as_deref(), Some("greeting"));
    assert_eq!(intent(&out[1]).as_deref(), Some("unknown"));
    assert_eq!(fs::read(model()).unwrap(), before, "weights.zlm must be byte-identical");
    let life = fs::read_to_string(d.join("life.t.zlm")).unwrap();
    assert!(life.contains("FEAT greeting"), "{life}");
}

#[test]
fn bad_feedback_makes_the_model_abstain_on_that_wording() {
    let d = sandbox("feedback");
    let out = jsonl(
        &d,
        &["--tag", "f"],
        &["show the weight file", "bad", "show the weight file", "show the weight file"],
    );
    assert_eq!(intent(&out[0]).as_deref(), Some("weights"));
    // "weight file" (3.2) minus 4 for the bad rating is below the abstain line.
    assert_eq!(intent(&out[2]).as_deref(), Some("unknown"));
    assert_eq!(intent(&out[3]).as_deref(), Some("unknown"));
}

#[test]
fn short_follow_ups_use_the_previous_question_with_the_real_model() {
    let d = sandbox("followup");
    let out = jsonl(&d, &["--tag", "c"], &["does it use training", "why?", "again", "what did I say"]);
    assert_eq!(intent(&out[0]).as_deref(), Some("no_training"));
    assert_eq!(intent(&out[1]).as_deref(), Some("no_training"));
    assert_eq!(reply(&out[1]), reply(&out[0]));
    assert_eq!(reply(&out[2]), reply(&out[1]));
    assert_eq!(reply(&out[3]), "You said: why?");
    // a long unrelated sentence must not borrow the previous intent
    let out = jsonl(&d, &["--tag", "c"], &["tell me something about the weather outside today"]);
    assert_eq!(intent(&out[0]).as_deref(), Some("unknown"));
}

#[test]
fn capacity_evicts_the_weakest_memory() {
    let d = sandbox("capacity");
    let out = jsonl(
        &d,
        &["--tag", "cap", "--capacity", "2"],
        &["remember a is 1", "remember b is 2", "remember c is 3", "what is a", "what is c"],
    );
    assert!(reply(&out[2]).contains("I let go of a"), "{}", reply(&out[2]));
    assert_eq!(number(&out[2], "memories"), 2);
    assert_ne!(reply(&out[3]), "a is 1.");
    assert_eq!(reply(&out[4]), "c is 3.");
}

#[test]
fn unused_facts_fade_but_used_facts_stay() {
    let d = sandbox("fade");
    let mut lines = vec!["remember the kept fact is alive", "remember the lost fact is gone"];
    lines.extend(vec!["hello"; 60]);
    // Recalling reinforces; ask about the kept one every 20 turns.
    let mut script: Vec<&str> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        script.push(l);
        if i > 2 && i % 20 == 0 {
            script.push("what is the kept fact");
        }
    }
    script.extend(vec!["hello"; 60]);
    script.push("what is the kept fact");
    script.push("what is the lost fact");
    let out = jsonl(&d, &["--tag", "fade"], &script);
    let n = out.len();
    assert_eq!(reply(&out[n - 2]), "the kept fact is alive.");
    assert_ne!(reply(&out[n - 1]), "the lost fact is gone.", "an unused fact must have faded");
}

#[test]
fn new_intents_work_inside_live_mode() {
    let d = sandbox("intents");
    let out = jsonl(&d, &["--tag", "i"], &["what license does it use", "how do I install ZeroLM", "how do tags work"]);
    assert_eq!(intent(&out[0]).as_deref(), Some("license"));
    assert_eq!(intent(&out[1]).as_deref(), Some("install"));
    assert_eq!(intent(&out[2]).as_deref(), Some("live"));
}

#[test]
fn hostile_inputs_do_not_crash() {
    let d = sandbox("hostile");
    let huge = "a ".repeat(60_000);
    let nul = "hello\u{0}world";
    let out = jsonl(
        &d,
        &["--tag", "h", "--context", "64"],
        &[&huge, nul, "remember  is ", "remember x is", "forget ", "teach  as ", "teach a as", "\t", "🙂🙂🙂"],
    );
    // Only a truly empty line is skipped; a tab-only line is still a turn.
    assert_eq!(out.len(), 9, "one reply per non-empty input line");
    for line in &out {
        assert!(line.starts_with('{') && line.ends_with('}'), "{line}");
    }
}

#[test]
fn corrupt_life_file_and_bad_arguments_exit_with_code_2() {
    let d = sandbox("errors");
    let m = model();
    fs::write(d.join("life.bad.zlm"), "this is not a life file\n").unwrap();
    let o = run(&d, &["live", "--model", &m, "--tag", "bad"], "hello\n");
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("life error"));
    assert_eq!(fs::read_to_string(d.join("life.bad.zlm")).unwrap(), "this is not a life file\n",
        "a corrupt life file must be left untouched");

    for args in [
        vec!["live", "--tag", "a/b"],
        vec!["live", "--tag", "x", "--life", "y"],
        vec!["live", "--capacity", "0"],
        vec!["live", "--context", "nope"],
        vec!["live", "--bogus"],
        vec!["live", "--model", "/does/not/exist"],
    ] {
        let o = run(&d, &args, "hello\n");
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    }
}

#[test]
fn one_shot_mode_stays_stateless() {
    let d = sandbox("oneshot");
    let m = model();
    let o = run(&d, &["--model", &m, "--verbose", "hello"], "");
    assert!(text(&o).contains("intent=greeting"));
    assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "one-shot must not create files");
}

#[test]
fn zerolm_model_env_selects_the_default_model() {
    let d = sandbox("env");
    let o = Command::new(BIN)
        .args(["--verbose", "hello"])
        .env("ZEROLM_MODEL", model())
        .current_dir(&d)
        .output()
        .unwrap();
    assert!(text(&o).contains("intent=greeting"), "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn interactive_mode_prints_prompts_and_quits_on_slash_quit() {
    let d = sandbox("interactive");
    let m = model();
    let o = run(&d, &["live", "--model", &m, "--tag", "i"], "hello\n\n/quit\nnever reached\n");
    let t = text(&o);
    assert!(t.contains("ZeroLM live (age 0 turns, 0 memories)"), "{t}");
    assert_eq!(t.matches("bot> ").count(), 1, "blank lines are ignored, /quit stops: {t}");
}
