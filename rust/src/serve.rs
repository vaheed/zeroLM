//! `zerolm serve`: an HTTP/JSON API so other programs can talk to ZeroLM.
//!
//! * `/v1/ask` is stateless; `/v1/sessions/{tag}/...` is ZeroLM Live, one session per
//!   tag, each backed by `life.<tag>.zlm` and the same file lock as the CLI, so the server
//!   and a CLI/Docker session can never write the same life file.
//! * Std only. One thread per connection, one request per connection, hard limits on
//!   everything an untrusted client controls (see `Config` and `http`).
//! * The contract is `openapi.yaml` (served at `/v1/openapi.yaml`); a test checks that
//!   `ROUTES` and the spec list exactly the same operations.

use crate::http::{self, Request, MAX_BODY_BYTES};
use crate::jsonl::{escape, parse_object};
use crate::live::Session;
use crate::lock::{tagged_life_path, LifeLock};
use crate::memory::Memory;
use crate::{Model, MAX_TEXT};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The API contract, embedded so the binary can serve it.
pub const OPENAPI: &str = include_str!("../openapi.yaml");

/// Every operation the server implements: (method, path template).
pub const ROUTES: &[(&str, &str)] = &[
    ("GET", "/v1/health"),
    ("GET", "/v1/model"),
    ("GET", "/v1/openapi.yaml"),
    ("POST", "/v1/ask"),
    ("POST", "/v1/sessions/{tag}/turns"),
    ("GET", "/v1/sessions/{tag}"),
    ("DELETE", "/v1/sessions/{tag}"),
    ("GET", "/v1/sessions/{tag}/memories"),
    ("DELETE", "/v1/sessions/{tag}/memories/{key}"),
    ("POST", "/v1/sessions/{tag}/teach"),
];

const MIN_REMOTE_TOKEN_LEN: usize = 16;
const MAX_PHRASE_BYTES: usize = 200;

#[derive(Debug, Clone)]
pub struct Config {
    pub addr: String,
    /// Bearer token; `None` only allowed on loopback addresses.
    pub token: Option<String>,
    /// Directory holding `life.<tag>.zlm` files.
    pub dir: PathBuf,
    pub capacity: usize,
    pub context: usize,
    /// Total time a client has to deliver a complete request.
    pub request_timeout: Duration,
    pub max_connections: usize,
    pub max_sessions: usize,
}

impl Config {
    pub fn new(addr: &str) -> Config {
        Config {
            addr: addr.to_string(),
            token: None,
            dir: PathBuf::from("."),
            capacity: crate::memory::DEFAULT_CAPACITY,
            context: crate::context::DEFAULT_CONTEXT_BUDGET,
            request_timeout: Duration::from_secs(10),
            max_connections: 64,
            max_sessions: 256,
        }
    }
}

/// Resolve the listen address and enforce the safe-by-default rules.
pub fn validate(cfg: &Config) -> Result<SocketAddr, String> {
    let addr = cfg
        .addr
        .to_socket_addrs()
        .map_err(|e| format!("bad --addr {:?}: {e}", cfg.addr))?
        .next()
        .ok_or_else(|| format!("bad --addr {:?}: no address", cfg.addr))?;
    if let Some(t) = &cfg.token {
        if t.is_empty() || t.bytes().any(|c| c <= b' ' || c == 0x7f) {
            return Err("the token must be non-empty and contain no spaces or control characters".into());
        }
    }
    if !addr.ip().is_loopback() {
        match &cfg.token {
            Some(t) if t.len() >= MIN_REMOTE_TOKEN_LEN => {}
            _ => {
                return Err(format!(
                    "refusing to listen on non-loopback address {addr} without a token of at least \
{MIN_REMOTE_TOKEN_LEN} characters (set ZEROLM_TOKEN)"
                ))
            }
        }
    }
    if cfg.max_connections == 0 || cfg.max_sessions == 0 || cfg.capacity == 0 || cfg.context == 0 {
        return Err("limits must be positive".into());
    }
    Ok(addr)
}

struct Resp {
    status: u16,
    content_type: &'static str,
    extra: Vec<(&'static str, String)>,
    body: String,
}

fn json(status: u16, body: String) -> Resp {
    Resp { status, content_type: "application/json", extra: Vec::new(), body }
}

fn error(status: u16, msg: &str) -> Resp {
    json(status, format!("{{\"error\":\"{}\"}}", escape(msg)))
}

struct Slot<'a> {
    session: Session<'a>,
    deleted: bool,
    _lock: LifeLock,
}

struct State<'a> {
    model: &'a Model,
    cfg: &'a Config,
    sessions: Mutex<HashMap<String, Arc<Mutex<Slot<'a>>>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // State is plain data and every handler leaves it consistent, so a poisoned
    // lock (a panic in another request) is safe to keep using.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Constant-time comparison, so response timing does not reveal the token.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn authorized(cfg: &Config, req: &Request) -> bool {
    let Some(want) = &cfg.token else { return true };
    let Some(h) = req.header("authorization") else { return false };
    match h.split_once(' ') {
        Some((scheme, got)) if scheme.eq_ignore_ascii_case("bearer") => ct_eq(got.trim().as_bytes(), want.as_bytes()),
        _ => false,
    }
}

fn pattern_matches(pattern: &str, segs: &[&str]) -> bool {
    let pat: Vec<&str> = pattern.trim_start_matches('/').split('/').collect();
    pat.len() == segs.len()
        && pat.iter().zip(segs).all(|(p, s)| if p.starts_with('{') { !s.is_empty() } else { p == s })
}

impl<'a> State<'a> {
    /// Find or open the session for `tag`. `create == false` returns 404 when it
    /// has never existed (no open session and no life file).
    fn session(&self, tag: &str, create: bool) -> Result<Arc<Mutex<Slot<'a>>>, Resp> {
        let file = tagged_life_path(tag).map_err(|e| error(400, &e))?;
        let life = self.cfg.dir.join(file);
        let mut map = lock(&self.sessions);
        if let Some(s) = map.get(tag) {
            return Ok(Arc::clone(s));
        }
        if !create && !life.exists() {
            return Err(error(404, "no such session"));
        }
        if map.len() >= self.cfg.max_sessions {
            return Err(error(503, "too many open sessions"));
        }
        let guard = LifeLock::acquire(&life).map_err(|e| {
            eprintln!("session {tag}: {e}");
            error(409, "this session is in use by another ZeroLM process")
        })?;
        let memory = Memory::load(&life, self.cfg.capacity).map_err(|e| {
            eprintln!("session {tag}: {e}");
            error(500, "the life file for this session is unreadable")
        })?;
        let mut session = Session::new(self.model, memory, Some(life));
        session.set_context_budget(self.cfg.context);
        let slot = Arc::new(Mutex::new(Slot { session, deleted: false, _lock: guard }));
        map.insert(tag.to_string(), Arc::clone(&slot));
        Ok(slot)
    }
}

fn parse_body(req: &Request) -> Result<HashMap<String, String>, Resp> {
    let text = std::str::from_utf8(&req.body).map_err(|_| error(400, "body must be UTF-8 JSON"))?;
    parse_object(text).ok_or_else(|| error(400, "body must be a JSON object"))
}

fn required<'m>(m: &'m HashMap<String, String>, key: &str) -> Result<&'m str, Resp> {
    match m.get(key) {
        Some(v) if !v.trim().is_empty() => Ok(v),
        Some(_) => Err(error(400, &format!("\"{key}\" must not be empty"))),
        None => Err(error(400, &format!("missing string field \"{key}\""))),
    }
}

fn text_field(req: &Request) -> Result<String, Resp> {
    let body = parse_body(req)?;
    let t = required(&body, "text")?;
    if t.len() > MAX_TEXT {
        return Err(error(400, &format!("\"text\" is too long (max {MAX_TEXT} bytes)")));
    }
    Ok(t.to_string())
}

fn intent_json(name: Option<&str>) -> String {
    name.map_or("null".to_string(), |n| format!("\"{}\"", escape(n)))
}

fn route(state: &State<'_>, req: &Request) -> Resp {
    let segs: Vec<&str> = req.path.trim_start_matches('/').split('/').collect();
    let mut allowed: Vec<&str> = Vec::new();
    let mut pattern: Option<&str> = None;
    for (m, p) in ROUTES {
        if pattern_matches(p, &segs) {
            allowed.push(m);
            if *m == req.method {
                pattern = Some(p);
            }
        }
    }
    // Health is the only unauthenticated route (for load balancers and probes).
    let open = pattern == Some("/v1/health");
    if !open && !authorized(state.cfg, req) {
        let mut r = error(401, "missing or invalid bearer token");
        r.extra.push(("WWW-Authenticate", "Bearer".to_string()));
        return r;
    }
    if allowed.is_empty() {
        return error(404, "no such endpoint");
    }
    let Some(pattern) = pattern else {
        let mut r = error(405, "method not allowed for this endpoint");
        r.extra.push(("Allow", allowed.join(", ")));
        return r;
    };

    let decode = |i: usize| http::percent_decode(segs[i]).ok_or_else(|| error(400, "malformed percent-encoding"));
    let result = match (req.method.as_str(), pattern) {
        ("GET", "/v1/health") => Ok(json(200, format!("{{\"status\":\"ok\",\"version\":\"{}\"}}", env!("CARGO_PKG_VERSION")))),
        ("GET", "/v1/model") => Ok(model_info(state.model)),
        ("GET", "/v1/openapi.yaml") => {
            Ok(Resp { status: 200, content_type: "application/yaml", extra: Vec::new(), body: OPENAPI.to_string() })
        }
        ("POST", "/v1/ask") => ask(state, req),
        (m, p) => decode(2).and_then(|tag| session_route(state, req, m, p, &tag, segs.get(4).map(|_| 4), &decode)),
    };
    result.unwrap_or_else(|r| r)
}

fn model_info(m: &Model) -> Resp {
    let names: Vec<String> = m.intent_name.iter().map(|n| format!("\"{}\"", escape(n))).collect();
    json(
        200,
        format!(
            "{{\"version\":\"{}\",\"intents\":[{}],\"features\":{},\"states\":{},\"vocabulary\":{}}}",
            env!("CARGO_PKG_VERSION"),
            names.join(","),
            m.features.len(),
            m.state_count(),
            m.vocab.len()
        ),
    )
}

fn ask(state: &State<'_>, req: &Request) -> Result<Resp, Resp> {
    let text = text_field(req)?;
    let (intent, score) = state.model.decide(&text);
    let reply = state.model.generate(intent).join(" ");
    Ok(json(
        200,
        format!(
            "{{\"reply\":\"{}\",\"intent\":\"{}\",\"score\":{:.2}}}",
            escape(&reply),
            escape(&state.model.intent_name[intent]),
            score
        ),
    ))
}

fn session_route(
    state: &State<'_>,
    req: &Request,
    method: &str,
    pattern: &str,
    tag: &str,
    key_index: Option<usize>,
    decode: &dyn Fn(usize) -> Result<String, Resp>,
) -> Result<Resp, Resp> {
    let create = method == "POST";
    let arc = state.session(tag, create)?;
    let mut slot = lock(&arc);
    if slot.deleted {
        return Err(error(404, "this session was deleted"));
    }
    match (method, pattern) {
        ("POST", "/v1/sessions/{tag}/turns") => {
            let text = text_field(req)?;
            let reply = slot.session.turn(&text);
            Ok(json(
                200,
                format!(
                    "{{\"reply\":\"{}\",\"intent\":{},\"age\":{},\"memories\":{}}}",
                    escape(&reply),
                    intent_json(slot.session.last_intent.as_deref()),
                    slot.session.memory.clock,
                    slot.session.memory.entries.len()
                ),
            ))
        }
        ("GET", "/v1/sessions/{tag}") => {
            let m = &slot.session.memory;
            Ok(json(
                200,
                format!(
                    "{{\"tag\":\"{}\",\"age\":{},\"memories\":{},\"learned\":{},\"capacity\":{}}}",
                    escape(tag),
                    m.clock,
                    m.entries.len(),
                    m.learned.len(),
                    m.capacity
                ),
            ))
        }
        ("GET", "/v1/sessions/{tag}/memories") => {
            let mut entries: Vec<_> = slot.session.memory.entries.iter().collect();
            entries.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
            let rows: Vec<String> = entries
                .iter()
                .map(|e| {
                    format!(
                        "{{\"key\":\"{}\",\"value\":\"{}\",\"strength\":{:.4},\"last_used\":{}}}",
                        escape(&e.key),
                        escape(&e.value),
                        e.strength,
                        e.last_used
                    )
                })
                .collect();
            Ok(json(200, format!("{{\"memories\":[{}]}}", rows.join(","))))
        }
        ("DELETE", "/v1/sessions/{tag}/memories/{key}") => {
            let key = decode(key_index.unwrap_or(4))?;
            if slot.session.forget_fact(&key) {
                Ok(json(204, String::new()))
            } else {
                Err(error(404, "no such memory"))
            }
        }
        ("POST", "/v1/sessions/{tag}/teach") => {
            let body = parse_body(req)?;
            let phrase = required(&body, "phrase")?;
            let intent = required(&body, "intent")?;
            if phrase.len() > MAX_PHRASE_BYTES {
                return Err(error(400, &format!("\"phrase\" is too long (max {MAX_PHRASE_BYTES} bytes)")));
            }
            match slot.session.teach(phrase, intent) {
                Ok((p, i)) => Ok(json(200, format!("{{\"phrase\":\"{}\",\"intent\":\"{}\"}}", escape(&p), escape(&i)))),
                Err(e) => Err(error(400, &e)),
            }
        }
        ("DELETE", "/v1/sessions/{tag}") => {
            if let Ok(file) = tagged_life_path(tag) {
                match fs::remove_file(state.cfg.dir.join(file)) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => {
                        eprintln!("session {tag}: cannot delete life file: {e}");
                        return Err(error(500, "cannot delete the life file"));
                    }
                }
            }
            slot.deleted = true;
            drop(slot);
            lock(&state.sessions).remove(tag);
            Ok(json(204, String::new()))
        }
        _ => Err(error(404, "no such endpoint")),
    }
}

/// A connection wrapper that enforces one total deadline for the whole request,
/// so a client cannot hold a thread by trickling bytes.
struct Deadline {
    stream: TcpStream,
    until: Instant,
}

impl Read for Deadline {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = self.until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "request deadline exceeded"));
        }
        self.stream.set_read_timeout(Some(left))?;
        self.stream.read(buf)
    }
}

impl Write for Deadline {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

fn send(stream: &mut TcpStream, r: &Resp) {
    let extra: Vec<(&str, &str)> = r.extra.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let _ = http::write_response(stream, r.status, r.content_type, &extra, &r.body);
    let _ = stream.shutdown(Shutdown::Write);
}

fn handle(state: &State<'_>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(state.cfg.request_timeout));
    let mut conn = Deadline { stream, until: Instant::now() + state.cfg.request_timeout };
    let resp = match http::read_request(&mut conn, MAX_BODY_BYTES) {
        Ok(req) => route(state, &req),
        Err((code, msg)) => error(code, msg),
    };
    send(&mut conn.stream, &resp);
}

/// Bind, announce the address through `ready`, and serve forever.
/// Only returns on a configuration or bind error.
pub fn run(model: &Model, cfg: &Config, ready: impl FnOnce(SocketAddr)) -> Result<(), String> {
    let addr = validate(cfg)?;
    fs::create_dir_all(&cfg.dir).map_err(|e| format!("cannot create {}: {e}", cfg.dir.display()))?;
    let listener = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
    let local = listener.local_addr().map_err(|e| e.to_string())?;
    let state = State { model, cfg, sessions: Mutex::new(HashMap::new()) };
    let active = AtomicUsize::new(0);
    ready(local);
    std::thread::scope(|scope| {
        for conn in listener.incoming() {
            let mut stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("accept failed: {e}");
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
            };
            if active.fetch_add(1, Ordering::SeqCst) >= cfg.max_connections {
                active.fetch_sub(1, Ordering::SeqCst);
                let mut r = error(503, "too many connections");
                r.extra.push(("Retry-After", "1".to_string()));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                send(&mut stream, &r);
                continue;
            }
            let (state, active) = (&state, &active);
            scope.spawn(move || {
                handle(state, stream);
                active.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_patterns() {
        assert!(pattern_matches("/v1/sessions/{tag}/turns", &["v1", "sessions", "a", "turns"]));
        assert!(!pattern_matches("/v1/sessions/{tag}/turns", &["v1", "sessions", "", "turns"]));
        assert!(!pattern_matches("/v1/sessions/{tag}", &["v1", "sessions", "a", "turns"]));
        assert!(!pattern_matches("/v1/health", &["v1", "healthz"]));
    }

    #[test]
    fn constant_time_compare() {
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreT"));
        assert!(!ct_eq(b"secret", b"secret!"));
        assert!(!ct_eq(b"", b"x"));
    }

    #[test]
    fn non_loopback_requires_a_long_token() {
        let mut c = Config::new("0.0.0.0:0");
        assert!(validate(&c).unwrap_err().contains("refusing"));
        c.token = Some("short".into());
        assert!(validate(&c).is_err());
        c.token = Some("a-sufficiently-long-token".into());
        assert!(validate(&c).is_ok());
        c.token = Some("has space inside it ok?".into());
        assert!(validate(&c).unwrap_err().contains("no spaces"));
        let mut local = Config::new("127.0.0.1:0");
        assert!(validate(&local).is_ok(), "loopback needs no token");
        local.max_connections = 0;
        assert!(validate(&local).is_err());
        assert!(validate(&Config::new("not an address")).is_err());
    }

    #[test]
    fn every_route_is_unique() {
        let mut seen = std::collections::HashSet::new();
        for r in ROUTES {
            assert!(seen.insert(*r), "duplicate route {r:?}");
        }
    }
}
