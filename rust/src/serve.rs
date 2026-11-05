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
use crate::jsonl::{escape, parse_object, Json};
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
    ("POST", "/v1/chat/completions"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/{id}"),
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
    /// Set for errors, so OpenAI-compatible routes can re-wrap them in OpenAI's error shape.
    msg: Option<String>,
}

fn json(status: u16, body: String) -> Resp {
    Resp { status, content_type: "application/json", extra: Vec::new(), body, msg: None }
}

fn error(status: u16, msg: &str) -> Resp {
    let mut r = json(status, format!("{{\"error\":\"{}\"}}", escape(msg)));
    r.msg = Some(msg.to_string());
    r
}

/// OpenAI's error envelope: {"error": {"message", "type", "param", "code"}}.
fn openai_error(status: u16, msg: &str) -> Resp {
    let kind = if status >= 500 { "server_error" } else { "invalid_request_error" };
    let code = if status == 401 { "\"invalid_api_key\"" } else if status == 404 { "\"model_not_found\"" } else { "null" };
    let mut r = json(
        status,
        format!(
            "{{\"error\":{{\"message\":\"{}\",\"type\":\"{kind}\",\"param\":null,\"code\":{code}}}}}",
            escape(msg)
        ),
    );
    r.msg = Some(msg.to_string());
    r
}

/// Routes that speak OpenAI's wire format, including its error shape.
fn is_openai_path(path: &str) -> bool {
    path == "/v1/chat/completions" || path == "/v1/models" || path.starts_with("/v1/models/")
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
    let mut resp = route_inner(state, req);
    if resp.status >= 400 && is_openai_path(&req.path) {
        if let Some(msg) = resp.msg.take() {
            let mut wrapped = openai_error(resp.status, &msg);
            wrapped.extra = std::mem::take(&mut resp.extra);
            return wrapped;
        }
    }
    resp
}

fn route_inner(state: &State<'_>, req: &Request) -> Resp {
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
            Ok(Resp { status: 200, content_type: "application/yaml", extra: Vec::new(), body: OPENAPI.to_string(), msg: None })
        }
        ("POST", "/v1/ask") => ask(state, req),
        ("POST", "/v1/chat/completions") => chat_completions(state, req),
        ("GET", "/v1/models") => Ok(models_list()),
        ("GET", "/v1/models/{id}") => decode(2).and_then(|id| model_one(&id)),
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

// ---------------------------------------------------------------------------
// OpenAI-compatible endpoints: /v1/chat/completions and /v1/models
// ---------------------------------------------------------------------------

const MODEL_STATELESS: &str = "zerolm";
const MODEL_LIVE: &str = "zerolm-live";
const MAX_MESSAGES: usize = 100;
static COMPLETION_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn model_json(id: &str) -> String {
    format!("{{\"id\":\"{id}\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zerolm\"}}")
}

fn models_list() -> Resp {
    json(200, format!("{{\"object\":\"list\",\"data\":[{},{}]}}", model_json(MODEL_STATELESS), model_json(MODEL_LIVE)))
}

fn model_one(id: &str) -> Result<Resp, Resp> {
    if id == MODEL_STATELESS || id == MODEL_LIVE {
        Ok(json(200, model_json(id)))
    } else {
        Err(error(404, &format!("The model `{id}` does not exist")))
    }
}

/// `content` may be a string or an array of `{"type":"text","text":...}` parts.
fn message_text(content: Option<&Json>) -> Result<String, Resp> {
    match content {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(parts)) => {
            let mut out: Vec<&str> = Vec::new();
            for p in parts {
                match (p.get("type").and_then(Json::as_str), p.get("text").and_then(Json::as_str)) {
                    (Some("text"), Some(t)) => out.push(t),
                    _ => return Err(error(400, "only text content parts are supported")),
                }
            }
            Ok(out.join("\n"))
        }
        Some(_) => Err(error(400, "message content must be a string or an array of text parts")),
    }
}

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn sse_event(id: &str, created: u64, model: &str, delta: &str, finish: Option<&str>) -> String {
    let finish = finish.map_or("null".to_string(), |f| format!("\"{f}\""));
    format!(
        "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{model}\",\
\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{finish}}}]}}\n\n"
    )
}

fn chat_completions(state: &State<'_>, req: &Request) -> Result<Resp, Resp> {
    let text = std::str::from_utf8(&req.body).map_err(|_| error(400, "body must be UTF-8 JSON"))?;
    let root = crate::jsonl::parse(text).ok_or_else(|| error(400, "body must be a JSON object"))?;
    if !matches!(root, Json::Obj(_)) {
        return Err(error(400, "body must be a JSON object"));
    }

    // Features ZeroLM cannot honour are refused rather than silently ignored.
    for key in ["tools", "functions", "function_call", "audio"] {
        if root.get(key).is_some_and(|v| *v != Json::Null) {
            return Err(error(400, &format!("`{key}` is not supported: ZeroLM cannot call tools or functions")));
        }
    }
    if let Some(tc) = root.get("tool_choice") {
        if !matches!(tc, Json::Null) && tc.as_str() != Some("none") {
            return Err(error(400, "`tool_choice` is not supported"));
        }
    }
    if let Some(rf) = root.get("response_format") {
        if rf.get("type").and_then(Json::as_str).is_some_and(|t| t != "text") {
            return Err(error(400, "only `response_format` of type `text` is supported"));
        }
    }
    if let Some(n) = root.get("n") {
        if !matches!(n, Json::Null) && n.as_f64() != Some(1.0) {
            return Err(error(400, "`n` must be 1"));
        }
    }
    let stream = match root.get("stream") {
        None | Some(Json::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| error(400, "`stream` must be a boolean"))?,
    };
    let mut max_tokens: Option<usize> = None;
    for key in ["max_tokens", "max_completion_tokens"] {
        if let Some(v) = root.get(key).filter(|v| **v != Json::Null) {
            match v.as_f64() {
                Some(n) if n >= 1.0 && n.fract() == 0.0 && n <= 1e9 => max_tokens = Some(n as usize),
                _ => return Err(error(400, &format!("`{key}` must be a positive integer"))),
            }
        }
    }

    let model = root.get("model").and_then(Json::as_str).ok_or_else(|| error(400, "missing string field \"model\""))?;
    if model != MODEL_STATELESS && model != MODEL_LIVE {
        return Err(error(404, &format!("The model `{model}` does not exist")));
    }

    let raw = root
        .get("messages")
        .and_then(Json::as_array)
        .ok_or_else(|| error(400, "missing array field \"messages\""))?;
    if raw.is_empty() || raw.len() > MAX_MESSAGES {
        return Err(error(400, &format!("\"messages\" must contain 1 to {MAX_MESSAGES} messages")));
    }
    let mut messages: Vec<(&str, String)> = Vec::with_capacity(raw.len());
    for m in raw {
        let role = m.get("role").and_then(Json::as_str).ok_or_else(|| error(400, "every message needs a string \"role\""))?;
        if !["system", "developer", "user", "assistant"].contains(&role) {
            return Err(error(400, &format!("unsupported message role `{role}`")));
        }
        if m.get("tool_calls").is_some_and(|v| *v != Json::Null) {
            return Err(error(400, "`tool_calls` is not supported"));
        }
        let content = message_text(m.get("content"))?;
        if content.len() > MAX_TEXT {
            return Err(error(400, &format!("message content is too long (max {MAX_TEXT} bytes)")));
        }
        messages.push((role, content));
    }
    let Some((last_role, last_text)) = messages.last() else { return Err(error(400, "no messages")) };
    if *last_role != "user" || last_text.trim().is_empty() {
        return Err(error(400, "the last message must have role `user` and non-empty content"));
    }
    let prompt_tokens: usize = messages.iter().map(|(_, c)| words(c)).sum();

    // Run the conversation.
    let (reply, intent, session_name) = if model == MODEL_LIVE {
        let tag = match root.get("user") {
            None | Some(Json::Null) => "default".to_string(),
            Some(Json::Str(u)) => u.clone(),
            Some(_) => return Err(error(400, "`user` must be a string")),
        };
        let arc = state.session(&tag, true)?;
        let mut slot = lock(&arc);
        if slot.deleted {
            return Err(error(404, "this session was deleted"));
        }
        let reply = slot.session.turn(last_text);
        (reply, slot.session.last_intent.clone(), Some(tag))
    } else {
        // Stateless: the messages ARE the context. Earlier user messages are replayed through a
        // throwaway session (no life file), so follow-ups and "remember ... / what is ..." work
        // within one request. Assistant and system messages cannot change ZeroLM's behaviour.
        let mut s = Session::new(state.model, Memory::new(state.cfg.capacity), None);
        s.set_context_budget(state.cfg.context);
        for (role, content) in &messages[..messages.len() - 1] {
            if *role == "user" && !content.trim().is_empty() {
                s.turn(content);
            }
        }
        let reply = s.turn(last_text);
        (reply, s.last_intent.clone(), None)
    };

    // max_tokens counts whitespace-separated words, the closest thing ZeroLM has to tokens.
    let mut toks: Vec<&str> = reply.split_whitespace().collect();
    let mut finish = "stop";
    if let Some(m) = max_tokens {
        if toks.len() > m {
            toks.truncate(m);
            finish = "length";
        }
    }
    let content = toks.join(" ");
    let completion_tokens = toks.len();
    let created = unix_now();
    let id = format!("chatcmpl-{:x}{:04x}", created, COMPLETION_COUNTER.fetch_add(1, Ordering::Relaxed) & 0xffff);

    if stream {
        let mut body = sse_event(&id, created, model, "{\"role\":\"assistant\",\"content\":\"\"}", None);
        for (i, t) in toks.iter().enumerate() {
            let piece = if i == 0 { (*t).to_string() } else { format!(" {t}") };
            body.push_str(&sse_event(&id, created, model, &format!("{{\"content\":\"{}\"}}", escape(&piece)), None));
        }
        body.push_str(&sse_event(&id, created, model, "{}", Some(finish)));
        body.push_str("data: [DONE]\n\n");
        return Ok(Resp { status: 200, content_type: "text/event-stream", extra: Vec::new(), body, msg: None });
    }

    let session_json = session_name.map_or(String::new(), |t| format!(",\"session\":\"{}\"", escape(&t)));
    Ok(json(
        200,
        format!(
            "{{\"id\":\"{id}\",\"object\":\"chat.completion\",\"created\":{created},\"model\":\"{model}\",\
\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"{finish}\"}}],\
\"usage\":{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":{completion_tokens},\"total_tokens\":{}}},\
\"zerolm\":{{\"intent\":{}{session_json}}}}}",
            escape(&content),
            prompt_tokens + completion_tokens,
            intent_json(intent.as_deref())
        ),
    ))
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

/// After rejecting a request early (413, 431, 400, ...) the client may still be sending. Closing
/// a socket that has unread input makes the OS reset the connection, and the client can lose the
/// error response. So read and discard what is left, briefly and with a hard cap, before closing.
fn drain(stream: &mut TcpStream) {
    const MAX_DRAIN: usize = 1024 * 1024;
    const DRAIN_TIME: Duration = Duration::from_millis(500);
    let until = Instant::now() + DRAIN_TIME;
    let mut buf = [0u8; 8192];
    let mut total = 0usize;
    while total < MAX_DRAIN {
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() || stream.set_read_timeout(Some(left)).is_err() {
            break;
        }
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => total += n,
        }
    }
}

fn handle(state: &State<'_>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(state.cfg.request_timeout));
    let mut conn = Deadline { stream, until: Instant::now() + state.cfg.request_timeout };
    match http::read_request(&mut conn, MAX_BODY_BYTES) {
        Ok(req) => send(&mut conn.stream, &route(state, &req)),
        Err((code, msg)) => {
            send(&mut conn.stream, &error(code, msg));
            drain(&mut conn.stream);
        }
    }
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
