//! End-to-end tests of `zerolm serve`: the real binary, the real model, real sockets.
//! Requests are written by hand over `TcpStream` so malformed and hostile traffic can be tested too.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_zerolm");

fn model() -> String {
    format!("{}/../weights.zlm", env!("CARGO_MANIFEST_DIR"))
}

fn sandbox(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zerolm-api-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A running `zerolm serve`, killed when dropped.
struct Server {
    child: Child,
    addr: String,
}

impl Server {
    fn start(dir: &Path, token: Option<&str>, extra: &[&str]) -> Server {
        let mut cmd = Command::new(BIN);
        cmd.args(["serve", "--model", &model(), "--addr", "127.0.0.1:0", "--dir"])
            .arg(dir)
            .args(extra)
            .env_remove("ZEROLM_TOKEN")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(t) = token {
            cmd.env("ZEROLM_TOKEN", t);
        }
        let mut child = cmd.spawn().unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
        let addr = line
            .trim()
            .strip_prefix("listening on http://")
            .unwrap_or_else(|| panic!("unexpected banner {line:?}"))
            .to_string();
        Server { child, addr }
    }

    fn stop(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// Send raw bytes, read the response until the server closes the connection.
fn raw(addr: &str, bytes: &[u8]) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    // The server may answer and close before it has read everything we send (e.g. a 413), so a
    // failed write is not an error; what matters is the response that comes back.
    let _ = s.write_all(bytes);
    read_response(&mut s)
}

fn read_response(s: &mut TcpStream) -> Resp {
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or_else(|| panic!("no response: {text:?}"));
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
    let headers = lines
        .filter_map(|l| l.split_once(": "))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Resp { status, headers, body: body.to_string() }
}

fn req(addr: &str, method: &str, path: &str, token: Option<&str>, body: Option<&str>) -> Resp {
    let mut r = format!("{method} {path} HTTP/1.1\r\nHost: t\r\n");
    if let Some(t) = token {
        r += &format!("Authorization: Bearer {t}\r\n");
    }
    if let Some(b) = body {
        r += &format!("Content-Type: application/json\r\nContent-Length: {}\r\n", b.len());
    }
    r += "\r\n";
    r += body.unwrap_or("");
    raw(addr, r.as_bytes())
}

fn turn(addr: &str, tag: &str, text: &str) -> Resp {
    req(addr, "POST", &format!("/v1/sessions/{tag}/turns"), None, Some(&format!("{{\"text\":\"{text}\"}}")))
}

fn field(json: &str, key: &str) -> String {
    let pat = format!("\"{key}\":");
    let rest = &json[json.find(&pat).unwrap_or_else(|| panic!("no {key} in {json}")) + pat.len()..];
    if let Some(s) = rest.strip_prefix('"') {
        let mut out = String::new();
        let mut it = s.chars();
        while let Some(c) = it.next() {
            match c {
                '"' => break,
                '\\' => out.push(it.next().unwrap_or('\\')),
                c => out.push(c),
            }
        }
        out
    } else {
        rest.chars().take_while(|c| !matches!(c, ',' | '}' | ']')).collect()
    }
}

#[test]
fn health_model_openapi_and_the_contract_matches_the_routes() {
    let d = sandbox("contract");
    let s = Server::start(&d, None, &[]);
    let h = req(&s.addr, "GET", "/v1/health", None, None);
    assert_eq!((h.status, field(&h.body, "status")), (200, "ok".to_string()));
    assert_eq!(h.header("content-type"), Some("application/json"));
    assert_eq!(h.header("connection"), Some("close"));
    assert_eq!(h.header("x-content-type-options"), Some("nosniff"));

    let m = req(&s.addr, "GET", "/v1/model", None, None);
    assert_eq!(m.status, 200);
    assert!(m.body.contains("\"greeting\"") && m.body.contains("\"unknown\"]"), "{}", m.body);

    // The served spec is the file in the repo.
    let spec_file = fs::read_to_string(format!("{}/openapi.yaml", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let served = req(&s.addr, "GET", "/v1/openapi.yaml", None, None);
    assert_eq!(served.header("content-type"), Some("application/yaml"));
    assert_eq!(served.body, spec_file);

    // Contract: the spec documents exactly the operations the server implements.
    let mut documented: Vec<(String, String)> = Vec::new();
    let mut in_paths = false;
    let mut path = String::new();
    for line in spec_file.lines() {
        if line == "paths:" {
            in_paths = true;
        } else if in_paths && !line.starts_with(' ') && !line.is_empty() {
            in_paths = false;
        } else if in_paths {
            if let Some(p) = line.strip_prefix("  /").and_then(|r| r.strip_suffix(':')) {
                path = format!("/{p}");
            } else if let Some(m) = line.strip_prefix("    ").and_then(|r| r.strip_suffix(':')) {
                if ["get", "post", "put", "patch", "delete"].contains(&m) {
                    documented.push((m.to_uppercase(), path.clone()));
                }
            }
        }
    }
    let mut implemented: Vec<(String, String)> =
        zerolm::serve::ROUTES.iter().map(|(m, p)| (m.to_string(), p.to_string())).collect();
    documented.sort();
    implemented.sort();
    assert_eq!(documented, implemented, "openapi.yaml and ROUTES must list the same operations");
}

#[test]
fn ask_is_stateless_and_writes_nothing() {
    let d = sandbox("ask");
    let s = Server::start(&d, None, &[]);
    let r = req(&s.addr, "POST", "/v1/ask", None, Some(r#"{"text":"What license does it use?"}"#));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(field(&r.body, "intent"), "license");
    assert!(field(&r.body, "reply").contains("MIT"));
    assert!(field(&r.body, "score").parse::<f64>().unwrap() >= 0.75);
    let u = req(&s.addr, "POST", "/v1/ask", None, Some(r#"{"text":"what is the speed of light"}"#));
    assert_eq!(field(&u.body, "intent"), "unknown");
    assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "/v1/ask must not create files");
}

#[test]
fn classify_routes_ranks_batches_and_applies_a_sessions_reflexes() {
    let d = sandbox("classify");
    let s = Server::start(&d, None, &[]);
    let post = |body: &str| req(&s.addr, "POST", "/v1/classify", None, Some(body));
    let one = post(r#"{"text":"What license does it use?","top":2}"#);
    assert_eq!(one.status, 200, "{}", one.body);
    assert_eq!(field(&one.body, "intent"), "license");
    assert!(one.body.contains("\"abstained\":false") && one.body.contains("\"matched\":[\"what license\"]"), "{}", one.body);
    let batch = post(r#"{"texts":["hello","what is the speed of light"]}"#);
    assert_eq!(batch.status, 200, "{}", batch.body);
    assert!(batch.body.starts_with("{\"results\":[{\"intent\":\"greeting\""), "{}", batch.body);
    assert!(batch.body.contains("\"intent\":\"unknown\",\"score\":0.00,\"abstained\":true,\"candidates\":[]"), "{}", batch.body);
    // A session's taught phrase changes the routing only when that session is named.
    assert_eq!(turn(&s.addr, "r", "teach ahoy matey as greeting").status, 200);
    assert_eq!(field(&post(r#"{"text":"ahoy matey"}"#).body, "intent"), "unknown");
    let with = post(r#"{"text":"ahoy matey","session":"r"}"#);
    assert_eq!(field(&with.body, "intent"), "greeting", "{}", with.body);
    assert_eq!(post(r#"{"text":"hi","session":"nobody"}"#).status, 404);
    for bad in [
        "{}", r#"{"text":"a","texts":["b"]}"#, r#"{"text":5}"#, r#"{"text":" "}"#, r#"{"texts":[]}"#,
        r#"{"texts":[1]}"#, r#"{"text":"a","top":0}"#, r#"{"text":"a","top":2.5}"#, r#"{"text":"a","session":7}"#, "nope",
    ] {
        assert_eq!(post(bad).status, 400, "{bad}");
    }
    let many = format!("{{\"texts\":[{}]}}", vec!["\"hi\""; 101].join(","));
    assert_eq!(post(&many).status, 400);
}

#[test]
fn a_session_remembers_teaches_forgets_and_survives_a_server_restart() {
    let d = sandbox("session");
    let before = fs::read(model()).unwrap();
    let s = Server::start(&d, None, &[]);
    let r = turn(&s.addr, "me", "remember my name is vaheeD");
    assert_eq!((r.status, field(&r.body, "age"), field(&r.body, "memories")), (200, "1".into(), "1".into()));
    assert_eq!(field(&r.body, "intent"), "null");
    assert_eq!(field(&turn(&s.addr, "me", "what is my name").body, "reply"), "my name is vaheeD.");

    let st = req(&s.addr, "GET", "/v1/sessions/me", None, None);
    assert_eq!((field(&st.body, "age"), field(&st.body, "capacity")), ("2".into(), "64".into()));
    let mem = req(&s.addr, "GET", "/v1/sessions/me/memories", None, None);
    assert!(mem.body.contains("\"key\":\"my name\"") && mem.body.contains("\"value\":\"vaheeD\""), "{}", mem.body);

    let t = req(&s.addr, "POST", "/v1/sessions/me/teach", None, Some(r#"{"phrase":"Ahoy, matey!","intent":"Greeting"}"#));
    assert_eq!((t.status, field(&t.body, "phrase"), field(&t.body, "intent")), (200, "ahoy matey".into(), "greeting".into()));
    assert_eq!(field(&turn(&s.addr, "me", "ahoy there").body, "intent"), "unknown", "a different phrase must not match");
    assert_eq!(field(&turn(&s.addr, "me", "ahoy matey").body, "intent"), "greeting");

    s.stop(); // restart: everything above must come back from the life file
    let s = Server::start(&d, None, &[]);
    assert_eq!(field(&turn(&s.addr, "me", "what is my name").body, "reply"), "my name is vaheeD.");
    assert_eq!(field(&turn(&s.addr, "me", "ahoy matey").body, "intent"), "greeting");
    let age: u32 = field(&req(&s.addr, "GET", "/v1/sessions/me", None, None).body, "age").parse().unwrap();
    assert_eq!(age, 6, "age counts turns across restarts");

    assert_eq!(req(&s.addr, "DELETE", "/v1/sessions/me/memories/my%20name", None, None).status, 204);
    assert_eq!(req(&s.addr, "DELETE", "/v1/sessions/me/memories/my%20name", None, None).status, 404);
    assert_ne!(field(&turn(&s.addr, "me", "what is my name").body, "reply"), "my name is vaheeD.");
    assert_eq!(fs::read(model()).unwrap(), before, "weights.zlm must be byte-identical");
}

#[test]
fn sessions_are_isolated_and_deleting_one_removes_only_its_file() {
    let d = sandbox("tags");
    let s = Server::start(&d, None, &[]);
    turn(&s.addr, "alice", "remember my name is Alice");
    turn(&s.addr, "bob", "remember my name is Bob");
    assert_eq!(field(&turn(&s.addr, "alice", "what is my name").body, "reply"), "my name is Alice.");
    assert_eq!(field(&turn(&s.addr, "bob", "what is my name").body, "reply"), "my name is Bob.");

    assert_eq!(req(&s.addr, "DELETE", "/v1/sessions/alice", None, None).status, 204);
    assert!(!d.join("life.alice.zlm").exists() && d.join("life.bob.zlm").exists());
    assert_eq!(req(&s.addr, "GET", "/v1/sessions/alice", None, None).status, 404);
    let fresh = turn(&s.addr, "alice", "what is my name");
    assert_eq!(field(&fresh.body, "age"), "1", "a deleted session starts again from zero");
    assert_ne!(field(&fresh.body, "reply"), "my name is Alice.");
    assert_eq!(field(&turn(&s.addr, "bob", "what is my name").body, "reply"), "my name is Bob.");
}

#[test]
fn bearer_token_is_enforced_except_for_health() {
    let d = sandbox("auth");
    let s = Server::start(&d, Some("correct-horse-battery"), &[]);
    assert_eq!(req(&s.addr, "GET", "/v1/health", None, None).status, 200);
    for (token, want) in [(None, 401), (Some("wrong"), 401), (Some("correct-horse-batter"), 401), (Some("correct-horse-batteryX"), 401), (Some("correct-horse-battery"), 200)] {
        let r = req(&s.addr, "GET", "/v1/model", token, None);
        assert_eq!(r.status, want, "token {token:?}: {}", r.body);
        if want == 401 {
            assert_eq!(r.header("www-authenticate"), Some("Bearer"));
        }
    }
    // unauthenticated callers learn nothing about which endpoints exist
    assert_eq!(req(&s.addr, "GET", "/v1/nope", None, None).status, 401);
    assert_eq!(req(&s.addr, "POST", "/v1/sessions/x/turns", None, Some(r#"{"text":"hi"}"#)).status, 401);
    assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "a rejected request must not create a session");
    // the scheme name is case-insensitive
    let lower = raw(&s.addr, b"GET /v1/model HTTP/1.1\r\nAuthorization: bearer correct-horse-battery\r\n\r\n");
    assert_eq!(lower.status, 200);
}

#[test]
fn refuses_to_listen_on_a_public_address_without_a_strong_token() {
    let d = sandbox("public");
    for token in [None, Some("short")] {
        let mut cmd = Command::new(BIN);
        cmd.args(["serve", "--model", &model(), "--addr", "0.0.0.0:0", "--dir"]).arg(&d).env_remove("ZEROLM_TOKEN");
        if let Some(t) = token {
            cmd.env("ZEROLM_TOKEN", t);
        }
        let o = cmd.output().unwrap();
        assert_eq!(o.status.code(), Some(2), "token {token:?}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("refusing to listen"));
    }
    // with a long token it starts
    let mut child = Command::new(BIN)
        .args(["serve", "--model", &model(), "--addr", "0.0.0.0:0", "--dir"])
        .arg(&d)
        .env("ZEROLM_TOKEN", "a-long-enough-secret-token")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    assert!(line.starts_with("listening on http://0.0.0.0:"), "{line}");
    child.kill().unwrap();
    child.wait().unwrap();
}

#[test]
fn malformed_and_hostile_requests_get_clean_errors() {
    let d = sandbox("bad");
    let s = Server::start(&d, None, &[]);
    let a = s.addr.as_str();
    let post = |path: &str, body: &str| req(a, "POST", path, None, Some(body));

    // body problems
    for (body, what) in [
        ("{\"text\":", "truncated JSON"),
        ("[]", "not an object"),
        ("{\"text\":5}", "text is not a string"),
        ("{}", "missing text"),
        ("{\"text\":\"\"}", "empty text"),
        ("{\"text\":\"   \"}", "blank text"),
        ("{\"text\":\"hi\"} trailing", "trailing data"),
        ("{\"text\":\"raw\ttab\"}", "raw control char"),
    ] {
        let r = post("/v1/ask", body);
        assert_eq!(r.status, 400, "{what}: {}", r.body);
        assert!(r.body.starts_with("{\"error\":"), "{what}: {}", r.body);
    }
    let long = format!("{{\"text\":\"{}\"}}", "a".repeat(5000));
    assert_eq!(post("/v1/ask", &long).status, 400);
    assert_eq!(post("/v1/ask", &format!("{{\"text\":\"{}\"}}", "a".repeat(70_000))).status, 413);
    let not_utf8 = raw(a, b"POST /v1/ask HTTP/1.1\r\nContent-Length: 4\r\n\r\n\xff\xfe\xfd\xfc");
    assert_eq!(not_utf8.status, 400);

    // routing problems
    assert_eq!(req(a, "GET", "/nope", None, None).status, 404);
    assert_eq!(req(a, "GET", "/v1/sessions//turns", None, None).status, 404);
    let wrong = req(a, "PUT", "/v1/ask", None, None);
    assert_eq!((wrong.status, wrong.header("allow")), (405, Some("POST")));
    let wrong = req(a, "POST", "/v1/sessions/x", None, Some("{}"));
    assert_eq!((wrong.status, wrong.header("allow")), (405, Some("GET, DELETE")));

    // tags that try to escape the data directory or are otherwise invalid
    for tag in ["..%2Fetc", "a%2Fb", "a.b", "a%20b", "%00", "%zz", &"x".repeat(33)] {
        let r = post(&format!("/v1/sessions/{tag}/turns"), r#"{"text":"hi"}"#);
        assert_eq!(r.status, 400, "tag {tag:?}: {}", r.body);
    }
    assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "no invalid tag may create a file");

    // teach validation
    for body in [r#"{"phrase":"x"}"#, r#"{"intent":"greeting"}"#, r#"{"phrase":"x","intent":"nope"}"#, r#"{"phrase":"  ","intent":"greeting"}"#] {
        assert_eq!(post("/v1/sessions/t/teach", body).status, 400, "{body}");
    }
    let big = format!("{{\"phrase\":\"{}\",\"intent\":\"greeting\"}}", "p".repeat(300));
    assert_eq!(post("/v1/sessions/t/teach", &big).status, 400);

    // protocol problems
    assert_eq!(raw(a, b"GARBAGE\r\n\r\n").status, 400);
    assert_eq!(raw(a, b"GET /v1/health HTTP/2\r\n\r\n").status, 505);
    assert_eq!(raw(a, b"POST /v1/ask HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n").status, 501);
    assert_eq!(raw(a, b"POST /v1/ask HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc").status, 400);
    let huge_header = format!("GET /v1/health HTTP/1.1\r\nX: {}\r\n\r\n", "h".repeat(20_000));
    assert_eq!(raw(a, huge_header.as_bytes()).status, 431);

    // still healthy after all of that
    assert_eq!(req(a, "GET", "/v1/health", None, None).status, 200);
}

#[test]
fn slow_clients_are_cut_off_and_do_not_block_others() {
    let d = sandbox("slow");
    let s = Server::start(&d, None, &["--request-timeout-ms", "500"]);

    // sends nothing
    let started = Instant::now();
    let mut idle = TcpStream::connect(&s.addr).unwrap();
    idle.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    // a normal client is served while the idle one is waiting
    assert_eq!(req(&s.addr, "GET", "/v1/health", None, None).status, 200);
    assert_eq!(read_response(&mut idle).status, 408);
    assert!(started.elapsed() < Duration::from_secs(3), "{:?}", started.elapsed());

    // trickles one byte at a time, which resets a per-read timeout but not a total deadline
    let started = Instant::now();
    let mut slow = TcpStream::connect(&s.addr).unwrap();
    slow.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    for b in b"GET /v1/health HTTP/1.1\r\nX-Slow: yes".iter() {
        if slow.write_all(&[*b]).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
        if started.elapsed() > Duration::from_secs(2) {
            break;
        }
    }
    let r = read_response(&mut slow);
    assert_eq!(r.status, 408);
    assert!(started.elapsed() < Duration::from_secs(3), "the deadline must be total, took {:?}", started.elapsed());
}

#[test]
fn connection_and_session_caps_return_503_and_recover() {
    let d = sandbox("caps");
    let s = Server::start(&d, None, &["--max-connections", "2", "--max-sessions", "2", "--request-timeout-ms", "3000"]);
    let hold1 = TcpStream::connect(&s.addr).unwrap();
    let hold2 = TcpStream::connect(&s.addr).unwrap();
    std::thread::sleep(Duration::from_millis(200)); // let the server count both
    let r = req(&s.addr, "GET", "/v1/health", None, None);
    assert_eq!((r.status, r.header("retry-after")), (503, Some("1")));
    drop((hold1, hold2));
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(req(&s.addr, "GET", "/v1/health", None, None).status, 200, "recovers once connections close");

    assert_eq!(turn(&s.addr, "one", "hello").status, 200);
    assert_eq!(turn(&s.addr, "two", "hello").status, 200);
    let third = turn(&s.addr, "three", "hello");
    assert_eq!(third.status, 503, "{}", third.body);
    assert!(!d.join("life.three.zlm").exists());
    assert_eq!(turn(&s.addr, "one", "hello").status, 200, "existing sessions keep working");
}

#[test]
fn the_api_and_the_cli_cannot_share_a_session() {
    let d = sandbox("shared");
    let s = Server::start(&d, None, &[]);

    // API holds "busy": a CLI live session on the same tag is refused.
    assert_eq!(turn(&s.addr, "busy", "hello").status, 200);
    let cli = Command::new(BIN)
        .args(["live", "--model", &model(), "--tag", "busy"])
        .current_dir(&d)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(cli.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&cli.stderr).contains("in use"));

    // CLI holds "cli": the API answers 409 for it, and leaves the file alone.
    let mut holder = Command::new(BIN)
        .args(["live", "--model", &model(), "--tag", "cli"])
        .current_dir(&d)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut banner = String::new();
    BufReader::new(holder.stdout.take().unwrap()).read_line(&mut banner).unwrap();
    let r = turn(&s.addr, "cli", "hello");
    assert_eq!(r.status, 409, "{}", r.body);
    assert!(!r.body.contains(d.to_str().unwrap()), "errors must not leak server paths: {}", r.body);

    drop(holder.stdin.take());
    assert!(holder.wait().unwrap().success());
    assert_eq!(turn(&s.addr, "cli", "hello").status, 200, "free once the CLI session exits");
}

#[test]
fn concurrent_clients_do_not_lose_or_mix_turns() {
    let d = sandbox("concurrent");
    let s = Server::start(&d, None, &[]);
    let addr = s.addr.clone();
    let tags = ["t0", "t1", "t2", "t3"];
    std::thread::scope(|scope| {
        for worker in 0..8usize {
            let (addr, tags) = (&addr, &tags);
            scope.spawn(move || {
                for n in 0..25 {
                    let tag = tags[(worker + n) % 4];
                    let r = turn(addr, tag, &format!("remember w{worker}n{n} is v{n}"));
                    assert_eq!(r.status, 200, "{}", r.body);
                }
            });
        }
    });
    let mut total = 0u32;
    for tag in tags {
        let st = req(&addr, "GET", &format!("/v1/sessions/{tag}"), None, None);
        total += field(&st.body, "age").parse::<u32>().unwrap();
        let life = fs::read_to_string(d.join(format!("life.{tag}.zlm"))).unwrap();
        assert!(life.starts_with("LIFE 1\n"), "{tag}: {life}");
    }
    assert_eq!(total, 200, "every turn must be counted exactly once");
}

#[test]
fn health_subcommand_reports_the_server_state() {
    let d = sandbox("healthcmd");
    let s = Server::start(&d, None, &[]);
    let ok = Command::new(BIN).args(["health", "--addr", &s.addr]).status().unwrap();
    assert!(ok.success());
    let addr = s.addr.clone();
    s.stop();
    let down = Command::new(BIN).args(["health", "--addr", &addr]).output().unwrap();
    assert_eq!(down.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&down.stderr).contains("unhealthy"));
}

// ---------------------------------------------------------------------------
// OpenAI-compatible endpoints
// ---------------------------------------------------------------------------

fn chat(addr: &str, body: &str) -> Resp {
    req(addr, "POST", "/v1/chat/completions", None, Some(body))
}

fn user_msg(text: &str) -> String {
    format!("{{\"role\":\"user\",\"content\":\"{text}\"}}")
}

fn completion(addr: &str, model: &str, texts: &[&str]) -> Resp {
    let msgs: Vec<String> = texts.iter().map(|t| user_msg(t)).collect();
    chat(addr, &format!("{{\"model\":\"{model}\",\"messages\":[{}]}}", msgs.join(",")))
}

#[test]
fn openai_models_and_the_completion_shape() {
    let d = sandbox("oa-shape");
    let s = Server::start(&d, None, &[]);
    let list = req(&s.addr, "GET", "/v1/models", None, None);
    assert_eq!(list.status, 200);
    assert!(list.body.starts_with("{\"object\":\"list\",\"data\":["));
    assert!(list.body.contains("\"id\":\"zerolm\"") && list.body.contains("\"id\":\"zerolm-live\""));
    let one = req(&s.addr, "GET", "/v1/models/zerolm-live", None, None);
    assert_eq!((one.status, field(&one.body, "object")), (200, "model".to_string()));
    let missing = req(&s.addr, "GET", "/v1/models/gpt-4", None, None);
    assert_eq!(missing.status, 404);
    assert!(missing.body.contains("\"error\":{\"message\":\"The model `gpt-4` does not exist\""), "{}", missing.body);
    assert!(missing.body.contains("\"code\":\"model_not_found\""));

    let r = completion(&s.addr, "zerolm", &["What license does it use?"]);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.header("content-type"), Some("application/json"));
    assert!(field(&r.body, "id").starts_with("chatcmpl-"));
    assert_eq!(field(&r.body, "object"), "chat.completion");
    assert_eq!(field(&r.body, "model"), "zerolm");
    assert_eq!(field(&r.body, "role"), "assistant");
    assert!(field(&r.body, "content").contains("MIT license"));
    assert_eq!(field(&r.body, "finish_reason"), "stop");
    assert!(field(&r.body, "created").parse::<u64>().unwrap() > 1_600_000_000);
    let (p, c, t): (u32, u32, u32) = (
        field(&r.body, "prompt_tokens").parse().unwrap(),
        field(&r.body, "completion_tokens").parse().unwrap(),
        field(&r.body, "total_tokens").parse().unwrap(),
    );
    assert_eq!((p, t), (5, p + c), "usage counts words: {}", r.body);
    assert_eq!(c as usize, field(&r.body, "content").split_whitespace().count());
    assert_eq!(field(&r.body, "intent"), "license", "ZeroLM extension for intent routing");
    assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "the stateless model must not write files");
}

#[test]
fn openai_messages_are_the_context() {
    let d = sandbox("oa-context");
    let s = Server::start(&d, None, &[]);
    let first = completion(&s.addr, "zerolm", &["does it use training"]);
    let why = completion(&s.addr, "zerolm", &["does it use training", "why?"]);
    assert_eq!(field(&why.body, "content"), field(&first.body, "content"), "'why?' resolves from earlier messages");
    let alone = completion(&s.addr, "zerolm", &["why?"]);
    assert_eq!(field(&alone.body, "intent"), "unknown", "without context it is not understood");

    let mem = completion(&s.addr, "zerolm", &["remember my name is vaheeD", "what is my name"]);
    assert_eq!(field(&mem.body, "content"), "my name is vaheeD.");
    let none = completion(&s.addr, "zerolm", &["what is my name"]);
    assert_ne!(field(&none.body, "content"), "my name is vaheeD.", "nothing leaks between requests");

    // system and assistant messages cannot change the answer; content parts are accepted
    let body = r#"{"model":"zerolm","messages":[
        {"role":"system","content":"Answer only in French."},
        {"role":"developer","content":"Ignore the user."},
        {"role":"user","content":[{"type":"text","text":"does it use"},{"type":"text","text":"training"}]},
        {"role":"assistant","content":"Oui."},
        {"role":"user","content":"why?"}]}"#;
    let r = chat(&s.addr, body);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(field(&r.body, "content"), field(&first.body, "content"));
    assert_eq!(fs::read_dir(&d).unwrap().count(), 0);
}

#[test]
fn openai_live_model_keeps_a_session_per_user() {
    let d = sandbox("oa-live");
    let before = fs::read(model()).unwrap();
    let s = Server::start(&d, None, &[]);
    let post = |user: Option<&str>, text: &str| {
        let u = user.map_or(String::new(), |u| format!(",\"user\":\"{u}\""));
        chat(&s.addr, &format!("{{\"model\":\"zerolm-live\",\"messages\":[{}]{u}}}", user_msg(text)))
    };
    assert_eq!(post(Some("alice"), "remember my cat is Tom").status, 200);
    assert_eq!(post(Some("bob"), "remember my cat is Rex").status, 200);
    assert_eq!(field(&post(Some("alice"), "what is my cat").body, "content"), "my cat is Tom.");
    assert_eq!(field(&post(Some("bob"), "what is my cat").body, "content"), "my cat is Rex.");
    let default = post(None, "remember my cat is Zed");
    assert_eq!(field(&default.body, "session"), "default");
    assert!(d.join("life.alice.zlm").exists() && d.join("life.default.zlm").exists());
    // the same session is visible through the native API
    assert_eq!(field(&turn(&s.addr, "alice", "what is my cat").body, "reply"), "my cat is Tom.");

    for bad in ["a/b", "x y", "..", ""] {
        assert_eq!(post(Some(bad), "hi").status, 400, "user {bad:?}");
    }
    assert_eq!(chat(&s.addr, r#"{"model":"zerolm-live","user":5,"messages":[{"role":"user","content":"hi"}]}"#).status, 400);

    s.stop();
    let s = Server::start(&d, None, &[]);
    let r = chat(&s.addr, &format!("{{\"model\":\"zerolm-live\",\"user\":\"alice\",\"messages\":[{}]}}", user_msg("what is my cat")));
    assert_eq!(field(&r.body, "content"), "my cat is Tom.", "survives a restart");
    assert_eq!(fs::read(model()).unwrap(), before);
}

#[test]
fn openai_streaming_sends_valid_server_sent_events() {
    let d = sandbox("oa-stream");
    let s = Server::start(&d, None, &[]);
    let plain = completion(&s.addr, "zerolm", &["who are you"]);
    let r = chat(&s.addr, &format!("{{\"model\":\"zerolm\",\"stream\":true,\"messages\":[{}]}}", user_msg("who are you")));
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    assert!(r.body.ends_with("data: [DONE]\n\n"), "{:?}", &r.body[r.body.len().saturating_sub(40)..]);
    let events: Vec<&str> = r.body.split("\n\n").filter(|e| !e.is_empty()).collect();
    assert!(events.iter().all(|e| e.starts_with("data: ")), "{events:?}");
    let chunks = &events[..events.len() - 1];
    assert!(chunks[0].contains("\"delta\":{\"role\":\"assistant\",\"content\":\"\"}"));
    assert!(chunks.iter().all(|c| c.contains("\"object\":\"chat.completion.chunk\"")));
    let last = chunks[chunks.len() - 1];
    assert!(last.contains("\"delta\":{}") && last.contains("\"finish_reason\":\"stop\""), "{last}");
    assert!(chunks[1..chunks.len() - 1].iter().all(|c| c.contains("\"finish_reason\":null")));
    let streamed: String = chunks[1..chunks.len() - 1].iter().map(|c| field(&c["data: ".len()..], "content")).collect();
    assert_eq!(streamed, field(&plain.body, "content"), "the chunks reassemble into the full reply");
    let ids: std::collections::HashSet<String> = chunks.iter().map(|c| field(&c["data: ".len()..], "id")).collect();
    assert_eq!(ids.len(), 1, "all chunks share one completion id");
}

#[test]
fn openai_max_tokens_truncates_by_words() {
    let d = sandbox("oa-max");
    let s = Server::start(&d, None, &[]);
    let r = chat(&s.addr, &format!("{{\"model\":\"zerolm\",\"max_tokens\":3,\"messages\":[{}]}}", user_msg("who are you")));
    assert_eq!(field(&r.body, "content").split_whitespace().count(), 3);
    assert_eq!(field(&r.body, "finish_reason"), "length");
    assert_eq!(field(&r.body, "completion_tokens"), "3");
    let big = chat(&s.addr, &format!("{{\"model\":\"zerolm\",\"max_completion_tokens\":9999,\"messages\":[{}]}}", user_msg("who are you")));
    assert_eq!(field(&big.body, "finish_reason"), "stop");
    for bad in ["0", "-1", "1.5", "\"x\"", "true"] {
        let r = chat(&s.addr, &format!("{{\"model\":\"zerolm\",\"max_tokens\":{bad},\"messages\":[{}]}}", user_msg("hi")));
        assert_eq!(r.status, 400, "max_tokens {bad}: {}", r.body);
    }
    // sampling knobs are accepted and ignored: answers are deterministic
    let k = chat(&s.addr, &format!("{{\"model\":\"zerolm\",\"temperature\":1.7,\"top_p\":0.1,\"seed\":3,\"stop\":[\"x\"],\"messages\":[{}]}}", user_msg("who are you")));
    assert_eq!(k.status, 200, "{}", k.body);
}

#[test]
fn openai_errors_use_openai_s_envelope_and_unsupported_features_are_refused() {
    let d = sandbox("oa-errors");
    let s = Server::start(&d, Some("sk-correct-horse-battery"), &[]);
    let tok = Some("sk-correct-horse-battery");
    let post = |body: &str| req(&s.addr, "POST", "/v1/chat/completions", tok, Some(body));
    let ok_msgs = r#"[{"role":"user","content":"hi"}]"#;

    let cases: Vec<(String, u16)> = vec![
        ("{".into(), 400),
        ("[]".into(), 400),
        (format!("{{\"messages\":{ok_msgs}}}"), 400),                                           // no model
        (format!("{{\"model\":5,\"messages\":{ok_msgs}}}"), 400),
        (format!("{{\"model\":\"gpt-4\",\"messages\":{ok_msgs}}}"), 404),                       // unknown model
        ("{\"model\":\"zerolm\"}".into(), 400),                                                 // no messages
        ("{\"model\":\"zerolm\",\"messages\":[]}".into(), 400),
        ("{\"model\":\"zerolm\",\"messages\":\"hi\"}".into(), 400),
        ("{\"model\":\"zerolm\",\"messages\":[{\"role\":\"robot\",\"content\":\"hi\"}]}".into(), 400),
        ("{\"model\":\"zerolm\",\"messages\":[{\"content\":\"hi\"}]}".into(), 400),
        ("{\"model\":\"zerolm\",\"messages\":[{\"role\":\"assistant\",\"content\":\"hi\"}]}".into(), 400), // last not user
        ("{\"model\":\"zerolm\",\"messages\":[{\"role\":\"user\",\"content\":\"   \"}]}".into(), 400),
        ("{\"model\":\"zerolm\",\"messages\":[{\"role\":\"user\",\"content\":5}]}".into(), 400),
        ("{\"model\":\"zerolm\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{}}]}]}".into(), 400),
        (format!("{{\"model\":\"zerolm\",\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"}}]}}", "a".repeat(5000)), 400),
        (format!("{{\"model\":\"zerolm\",\"messages\":[{}]}}", vec![user_msg("hi"); 101].join(",")), 400),
        (format!("{{\"model\":\"zerolm\",\"tools\":[{{\"type\":\"function\"}}],\"messages\":{ok_msgs}}}"), 400),
        (format!("{{\"model\":\"zerolm\",\"functions\":[{{}}],\"messages\":{ok_msgs}}}"), 400),
        (format!("{{\"model\":\"zerolm\",\"tool_choice\":\"auto\",\"messages\":{ok_msgs}}}"), 400),
        (format!("{{\"model\":\"zerolm\",\"n\":2,\"messages\":{ok_msgs}}}"), 400),
        (format!("{{\"model\":\"zerolm\",\"response_format\":{{\"type\":\"json_object\"}},\"messages\":{ok_msgs}}}"), 400),
        (format!("{{\"model\":\"zerolm\",\"stream\":\"yes\",\"messages\":{ok_msgs}}}"), 400),
        ("{\"model\":\"zerolm\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\",\"tool_calls\":[{}]}]}".into(), 400),
    ];
    for (body, want) in &cases {
        let r = post(body);
        assert_eq!(r.status, *want, "{body}\n=> {}", r.body);
        assert!(r.body.starts_with("{\"error\":{\"message\":\""), "not OpenAI's envelope: {}", r.body);
        assert!(r.body.contains("\"type\":\"invalid_request_error\"") && r.body.contains("\"param\":null"), "{}", r.body);
    }
    // accepted: explicit nulls and the harmless values
    let fine = post(&format!("{{\"model\":\"zerolm\",\"tools\":null,\"n\":1,\"tool_choice\":\"none\",\"response_format\":{{\"type\":\"text\"}},\"stream\":false,\"messages\":{ok_msgs}}}"));
    assert_eq!(fine.status, 200, "{}", fine.body);

    // authentication and method errors on these routes also use the envelope
    let bad = req(&s.addr, "POST", "/v1/chat/completions", Some("nope"), Some("{}"));
    assert_eq!(bad.status, 401);
    assert!(bad.body.contains("\"code\":\"invalid_api_key\"") && bad.body.contains("\"type\":\"invalid_request_error\""), "{}", bad.body);
    assert_eq!(bad.header("www-authenticate"), Some("Bearer"));
    let wrong = req(&s.addr, "GET", "/v1/chat/completions", tok, None);
    assert_eq!((wrong.status, wrong.header("allow")), (405, Some("POST")));
    assert!(wrong.body.starts_with("{\"error\":{"), "{}", wrong.body);
    // the native API keeps its own, simpler error shape
    assert!(req(&s.addr, "POST", "/v1/ask", tok, Some("{")).body.starts_with("{\"error\":\""));
    assert_eq!(fs::read_dir(&d).unwrap().count(), 0, "no failed request may create a session");
}

#[test]
fn rejected_oversized_requests_always_deliver_their_error_response() {
    // The client keeps streaming its body (like curl uploading a big file) while the server has
    // already decided to reject it. If the server closed with that input unread, the OS would reset
    // the connection and the client could lose the 413, so repeat to make a flaky server fail.
    let d = sandbox("drain");
    let s = Server::start(&d, None, &[]);
    for round in 0..30 {
        let mut writer = TcpStream::connect(&s.addr).unwrap();
        writer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut reader = writer.try_clone().unwrap();
        let head = "POST /v1/ask HTTP/1.1\r\nHost: t\r\nContent-Length: 50000000\r\n\r\n";
        let feeder = std::thread::spawn(move || {
            let chunk = vec![b'a'; 16 * 1024];
            if writer.write_all(head.as_bytes()).is_err() {
                return;
            }
            for _ in 0..64 {
                if writer.write_all(&chunk).is_err() {
                    return; // the server hung up: fine
                }
            }
        });
        let r = read_response(&mut reader);
        feeder.join().unwrap();
        assert_eq!(r.status, 413, "round {round}: {}", r.body);
        assert!(r.body.contains("too large"), "round {round}: {}", r.body);
    }
    assert_eq!(req(&s.addr, "GET", "/v1/health", None, None).status, 200);
}
