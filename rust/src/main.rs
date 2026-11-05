use std::env;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use zerolm::{Model, C_MODEL_STRUCT_BYTES, MAX_STEPS, MAX_TEXT};

/// Model path when `--model` is not given: `$ZEROLM_MODEL`, else `weights.zlm`.
fn default_model() -> String {
    env::var("ZEROLM_MODEL").unwrap_or_else(|_| "weights.zlm".to_string())
}

fn respond(m: &Model, raw: &str, verbose: bool, faithful: bool) {
    let (intent, score) = m.decide(raw);
    let toks = if faithful { m.generate_faithful(intent) } else { m.generate(intent) };
    if verbose {
        println!(
            "[intent={} score={:.2} emitted_tokens={}]",
            m.intent_name[intent],
            score,
            toks.len()
        );
    }
    println!("{}", toks.join(" "));
}

fn print_info(m: &Model) {
    let (i, f, s, v) = (m.intents, m.features.len(), m.state_count(), m.vocab.len());
    let dense = i * f + s * i + s * s + v * s;
    println!("ZeroLM v0.2.0");
    println!("Architecture: manual phrase-to-intent projection + sparse recurrent decoder");
    println!("Training: none | pretrained weights: none | runtime: CPU only");
    println!("Dimensions: intents={i} features={f} hidden_states={s} vocabulary={v}");
    println!("Dense-equivalent weight slots: {dense} (fixed, not trainable)");
    println!(
        "Model data structure upper bound: {} bytes ({:.2} KiB)",
        C_MODEL_STRUCT_BYTES,
        C_MODEL_STRUCT_BYTES as f64 / 1024.0
    );
    println!("Decoder: greedy autoregressive token generation, max {MAX_STEPS} tokens");
    println!("Important: this is a constrained phrase model, not a general-purpose LLM.");
}

fn self_test(m: &Model, faithful: bool) -> bool {
    let tests = [
        ("hello", "greeting"),
        ("good morning", "greeting"),
        ("who are you", "identity"),
        ("what can you do", "capabilities"),
        ("thank you", "thanks"),
        ("goodbye", "farewell"),
        ("how do I run ZeroLM", "help"),
        ("explain the architecture", "architecture"),
        ("show the weight file", "weights"),
        ("does it use training", "no_training"),
        ("what are its limitations", "limitations"),
        ("does it have memory", "memory"),
        ("make a plan step by step", "planning"),
        ("write code", "coding"),
        ("solve this math equation", "math"),
        ("explain this term", "explain"),
        ("run the benchmark pipeline", "testing"),
        ("tell me about weight-space programming", "inspiration"),
        ("what is the weather in Paris", "unknown"),
        ("who won the match yesterday", "unknown"),
    ];
    let total = tests.len() + m.intents;
    let mut passed = 0;
    println!("Running zero-training smoke tests:");
    for (input, expected) in tests {
        let (intent, _) = m.decide(input);
        let got = &m.intent_name[intent];
        let ok = got == expected;
        println!(
            "  {}  input=\"{input}\" expected={expected} got={got}",
            if ok { "PASS" } else { "FAIL" }
        );
        passed += usize::from(ok);
    }
    println!("Result: {}/{} classifier tests passed", passed, tests.len());
    println!("Checking recurrent decoder paths:");
    for intent in 0..m.intents {
        let toks = if faithful { m.generate_faithful(intent) } else { m.generate(intent) };
        let ok = !toks.is_empty() && !toks.join(" ").contains("<eos>");
        println!(
            "  {}  intent={} tokens={}",
            if ok { "PASS" } else { "FAIL" },
            m.intent_name[intent],
            toks.len()
        );
        passed += usize::from(ok);
    }
    println!("Total: {passed}/{total} checks passed");
    passed == total
}

/// `zerolm live [--model FILE] [--life FILE | --tag NAME] ...`: a long-lived
/// session whose memory persists across restarts in the life file. `--tag`
/// gives each session its own `life.NAME.zlm`; a lock file stops two processes
/// from sharing one life file.
fn run_live(args: &[String]) -> ExitCode {
    let (mut model_path, mut life, mut capacity) =
        (default_model(), "life.zlm".to_string(), zerolm::memory::DEFAULT_CAPACITY);
    let mut context = zerolm::context::DEFAULT_CONTEXT_BUDGET;
    let mut jsonl = false;
    let mut explicit_life = false;
    let mut tag: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--jsonl" {
            jsonl = true;
            i += 1;
            continue;
        }
        let val = args.get(i + 1);
        match (args[i].as_str(), val) {
            ("--model", Some(v)) => model_path = v.clone(),
            ("--life", Some(v)) => {
                life = v.clone();
                explicit_life = true;
            }
            ("--tag", Some(v)) => tag = Some(v.clone()),
            ("--context", Some(v)) => match v.parse::<usize>() {
                Ok(n) if n >= 1 => context = n,
                _ => {
                    eprintln!("error: --context needs a positive byte budget");
                    return ExitCode::from(2);
                }
            },
            ("--capacity", Some(v)) => match v.parse::<usize>() {
                Ok(n) if n >= 1 => capacity = n,
                _ => {
                    eprintln!("error: --capacity needs a positive integer");
                    return ExitCode::from(2);
                }
            },
            _ => {
                eprintln!("Usage: zerolm live [--model FILE] [--life FILE] [--life FILE | --tag NAME] [--capacity N] [--context BYTES] [--jsonl]");
                return ExitCode::from(2);
            }
        }
        i += 2;
    }
    if let Some(t) = &tag {
        if explicit_life {
            eprintln!("error: use either --life or --tag, not both");
            return ExitCode::from(2);
        }
        match zerolm::lock::tagged_life_path(t) {
            Ok(p) => life = p,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    }
    let model = match Model::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model error: {e}");
            return ExitCode::from(2);
        }
    };
    let path = std::path::PathBuf::from(&life);
    // Held until this function returns; released on drop.
    let _lock = match zerolm::lock::LifeLock::acquire(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let memory = match zerolm::memory::Memory::load(&path, capacity) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("life error: {e}");
            return ExitCode::from(2);
        }
    };
    let age = memory.clock;
    let mut session = zerolm::live::Session::new(&model, memory, Some(path));
    session.set_context_budget(context);
    if jsonl {
        let stdin = io::stdin();
        let mut out = io::stdout().lock();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let text = if line.starts_with('{') { zerolm::jsonl::extract_text(line) } else { Some(line.to_string()) };
            let resp = match text {
                None => "{\"error\":\"expected a string field \\\"text\\\"\"}".to_string(),
                Some(t) => {
                    let reply = session.turn(&t);
                    let intent = match &session.last_intent {
                        Some(i) => format!("\"{}\"", zerolm::jsonl::escape(i)),
                        None => "null".to_string(),
                    };
                    format!(
                        "{{\"reply\":\"{}\",\"intent\":{},\"age\":{},\"memories\":{}}}",
                        zerolm::jsonl::escape(&reply),
                        intent,
                        session.memory.clock,
                        session.memory.entries.len()
                    )
                }
            };
            if writeln!(out, "{resp}").and_then(|_| out.flush()).is_err() {
                break;
            }
        }
        return ExitCode::SUCCESS;
    }
    println!("ZeroLM live (age {age} turns, {} memories) — type /quit to exit.", session.memory.entries.len());
    let stdin = io::stdin();
    loop {
        print!("you> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line == "/quit" || line == "/exit" {
            break;
        }
        if line.is_empty() {
            continue;
        }
        println!("bot> {}", session.turn(line));
    }
    ExitCode::SUCCESS
}

/// Parse a positive integer option value, or exit with a usage error.
fn positive(name: &str, v: &str) -> Result<usize, String> {
    match v.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err(format!("{name} needs a positive integer")),
    }
}

/// `zerolm serve [--model FILE] [--addr HOST:PORT] [--dir DIR] ...`: the HTTP/JSON API.
/// The token comes from `$ZEROLM_TOKEN` (preferred, so it stays out of `ps`) or `--token`.
fn run_serve(args: &[String]) -> ExitCode {
    let fail = |msg: String| {
        eprintln!("error: {msg}");
        ExitCode::from(2)
    };
    let mut model_path = default_model();
    let mut cfg = zerolm::serve::Config::new("127.0.0.1:8080");
    cfg.token = env::var("ZEROLM_TOKEN").ok().filter(|t| !t.is_empty());
    let mut i = 0;
    while i < args.len() {
        let Some(v) = args.get(i + 1) else {
            return fail(format!("{} needs a value\nUsage: zerolm serve [--model FILE] [--addr HOST:PORT] [--dir DIR] [--token T] [--capacity N] [--context BYTES] [--request-timeout-ms N] [--max-connections N] [--max-sessions N]", args[i]));
        };
        let r: Result<(), String> = match args[i].as_str() {
            "--model" => {
                model_path = v.clone();
                Ok(())
            }
            "--addr" => {
                cfg.addr = v.clone();
                Ok(())
            }
            "--dir" => {
                cfg.dir = v.into();
                Ok(())
            }
            "--token" => {
                cfg.token = Some(v.clone());
                Ok(())
            }
            "--capacity" => positive("--capacity", v).map(|n| cfg.capacity = n),
            "--context" => positive("--context", v).map(|n| cfg.context = n),
            "--max-connections" => positive("--max-connections", v).map(|n| cfg.max_connections = n),
            "--max-sessions" => positive("--max-sessions", v).map(|n| cfg.max_sessions = n),
            "--request-timeout-ms" => positive("--request-timeout-ms", v)
                .map(|n| cfg.request_timeout = std::time::Duration::from_millis(n as u64)),
            other => Err(format!("unknown option {other}")),
        };
        if let Err(e) = r {
            return fail(e);
        }
        i += 2;
    }
    let model = match Model::load(&model_path) {
        Ok(m) => m,
        Err(e) => return fail(format!("model error: {e}")),
    };
    let ready = |addr: std::net::SocketAddr| {
        println!("listening on http://{addr}");
        let _ = io::stdout().flush();
    };
    match zerolm::serve::run(&model, &cfg, ready) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(e),
    }
}

/// `zerolm classify [--model FILE] [--top N]`: read one text per line from stdin and print one JSON
/// object per line with the routing decision and the top-N ranked intents. Stateless; no replies.
fn run_classify(args: &[String]) -> ExitCode {
    let (mut model_path, mut top) = (default_model(), 3usize);
    let mut i = 0;
    while i < args.len() {
        match (args[i].as_str(), args.get(i + 1)) {
            ("--model", Some(v)) => model_path = v.clone(),
            ("--top", Some(v)) => match v.parse::<usize>() {
                Ok(n) if (1..=10).contains(&n) => top = n,
                _ => {
                    eprintln!("error: --top must be 1-10");
                    return ExitCode::from(2);
                }
            },
            _ => {
                eprintln!("usage: zerolm classify [--model FILE] [--top N]   (texts on stdin, one per line)");
                return ExitCode::from(2);
            }
        }
        i += 2;
    }
    let model = match Model::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model error: {e}");
            return ExitCode::from(2);
        }
    };
    let out = io::stdout();
    let mut out = out.lock();
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let mut t = text.to_string();
        while t.len() > MAX_TEXT - 2 {
            t.pop();
        }
        if writeln!(out, "{}", model.rank(&t, &[], top).to_json(&model)).is_err() {
            break;
        }
    }
    ExitCode::SUCCESS
}

/// `zerolm health [--addr HOST:PORT]`: exit 0 if the server answers `GET /v1/health` with 200.
/// Used as the container health check (the image has no curl).
fn run_health(args: &[String]) -> ExitCode {
    use std::io::Read;
    use std::net::{TcpStream, ToSocketAddrs};
    let addr = match args {
        [] => "127.0.0.1:8080".to_string(),
        [flag, v] if flag == "--addr" => v.clone(),
        _ => {
            eprintln!("Usage: zerolm health [--addr HOST:PORT]");
            return ExitCode::from(2);
        }
    };
    let timeout = std::time::Duration::from_secs(3);
    let ok = (|| -> Option<bool> {
        let target = addr.to_socket_addrs().ok()?.next()?;
        let mut s = TcpStream::connect_timeout(&target, timeout).ok()?;
        s.set_read_timeout(Some(timeout)).ok()?;
        s.set_write_timeout(Some(timeout)).ok()?;
        s.write_all(b"GET /v1/health HTTP/1.1\r\nHost: health\r\nConnection: close\r\n\r\n").ok()?;
        let mut head = [0u8; 12];
        s.read_exact(&mut head).ok()?;
        Some(&head == b"HTTP/1.1 200")
    })()
    .unwrap_or(false);
    if ok {
        ExitCode::SUCCESS
    } else {
        eprintln!("unhealthy: no 200 from http://{addr}/v1/health");
        ExitCode::from(1)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("live") => return run_live(&args[2..]),
        Some("serve") => return run_serve(&args[2..]),
        Some("health") => return run_health(&args[2..]),
        Some("classify") => return run_classify(&args[2..]),
        Some("next") => return zerolm::next::cli::run(&args[2..]),
        _ => {}
    }
    let (mut model_path, mut info, mut test, mut verbose, mut faithful) =
        (default_model(), false, false, false, false);
    let mut text_start = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                i += 1;
                model_path = args[i].clone();
            }
            "--info" => info = true,
            "--self-test" => test = true,
            "--verbose" => verbose = true,
            "--faithful" => faithful = true,
            "--help" | "-h" => {
                println!("Usage: {} [--model FILE] [--info] [--self-test] [--verbose] [--faithful] [TEXT]", args[0]);
                println!("Without TEXT, starts an interactive chat. Type /quit to exit.");
                return ExitCode::SUCCESS;
            }
            _ => {
                text_start = Some(i);
                break;
            }
        }
        i += 1;
    }

    let model = match Model::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model error: {e}");
            return ExitCode::from(2);
        }
    };

    if info {
        print_info(&model);
        if !test && text_start.is_none() {
            return ExitCode::SUCCESS;
        }
    }
    if test {
        return if self_test(&model, faithful) { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    if let Some(s) = text_start {
        let mut joined = args[s..].join(" ");
        joined.truncate(MAX_TEXT - 2);
        respond(&model, &joined, verbose, faithful);
        return ExitCode::SUCCESS;
    }

    println!("ZeroLM v0.2.0 — type /quit to exit.");
    let stdin = io::stdin();
    loop {
        print!("you> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line == "/quit" || line == "/exit" {
            break;
        }
        if line.is_empty() {
            continue;
        }
        print!("bot> ");
        respond(&model, line, verbose, faithful);
    }
    ExitCode::SUCCESS
}
