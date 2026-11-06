mod memory_cmd;
mod reason_cmd;
mod verbose;

use std::io::{BufRead, Write};
use std::process::ExitCode;
use zerolm_engine::bootstrap::{default_model, Session, CELL_BUDGET, COMMANDS, DEFAULT_KIND, DEFAULT_SEED, EMBED_DIM};

const USAGE: &str = "usage: zerolm info [--seed N]\n       zerolm run [--seed N] [--trace FILE]   (reads lines on stdin; session ledger, nothing persists)\n       zerolm memory --dir DIR [--seed N] [--trace FILE] [--stream NAME]   (weight-resident memory, persists in DIR)\n       zerolm reason [--dir DIR] [--trace FILE] [--stream NAME]   (Phase 6: statements are stored, questions are reasoned over)\n\n--verbose (or ZEROLM_VERBOSE=1) shows what happens inside each step on stderr\n\nprotocol lines: teach K => V | ask K | correct K => V | confirm K | forget K | repeat";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let (mut seed, mut trace) = (DEFAULT_SEED, None::<String>);
    let (mut seed_set, mut dir, mut stream) = (false, None::<String>, "cli:stdin".to_string());
    let mut verbose = false;
    while let Some(a) = args.next() {
        if a == "--verbose" || a == "-v" {
            verbose = true;
            continue;
        }
        match (a.as_str(), args.next()) {
            ("--seed", Some(v)) => match v.parse() {
                Ok(n) => {
                    seed = n;
                    seed_set = true;
                }
                Err(_) => return bad("--seed needs an integer"),
            },
            ("--trace", Some(p)) => trace = Some(p),
            ("--dir", Some(p)) => dir = Some(p),
            ("--stream", Some(p)) => stream = p,
            _ => return bad("unknown or incomplete option"),
        }
    }
    let view = verbose::View::new(verbose);
    if cmd == "memory" {
        let Some(dir) = dir else { return bad("memory needs --dir") };
        return memory_cmd::run(std::path::Path::new(&dir), seed_set.then_some(seed), trace, &stream, &view);
    }
    if cmd == "reason" {
        return reason_cmd::run(dir.as_deref().map(std::path::Path::new), trace, &stream, &view);
    }
    if dir.is_some() {
        return bad("--dir is only valid with memory or reason");
    }
    let model = default_model(seed);
    match cmd.as_str() {
        "info" => {
            println!("architecture: {} (embed {EMBED_DIM}, cell budget {CELL_BUDGET})", DEFAULT_KIND.name());
            println!("parameters: {}", model.param_count());
            println!("session state bytes: {}", model.cell.state_len() * 4);
            println!("commands: {}", COMMANDS.join(", "));
            println!("weights_sha256: {}", model.weights_sha256());
            println!("note: bootstrap only; no learning, no English understanding");
            ExitCode::SUCCESS
        }
        "run" => run(&model, trace, &view),
        _ => bad("unknown command"),
    }
}

fn bad(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n{USAGE}");
    ExitCode::from(2)
}

fn run(model: &zerolm_engine::Model, trace: Option<String>, view: &verbose::View) -> ExitCode {
    let before = model.weights_sha256();
    view.note("model", format!("{} parameters, weights {}", model.param_count(), verbose::short(&before)));
    let mut session = Session::new(model);
    let mut sink = trace.map(|p| std::fs::File::create(p).expect("cannot create trace file"));
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for (seq, line) in stdin.lock().split(b'\n').enumerate() {
        let Ok(raw) = line else { break };
        // Arbitrary bytes are accepted; invalid UTF-8 is replaced for protocol parsing only.
        let text = String::from_utf8_lossy(&raw).into_owned();
        let text = text.strip_suffix('\r').unwrap_or(&text);
        let reply = session.handle_line(text);
        view.input(seq, text);
        if let (true, Some(ev)) = (view.on(), session.events().last()) {
            view.note("event", serde_json::to_string(ev).expect("serialize"));
            view.note("note", "session ledger only: nothing is stored in weights");
        }
        let _ = writeln!(out, "{reply}");
        if let (Some(f), Some(ev)) = (sink.as_mut(), session.events().last()) {
            let _ = writeln!(f, "{}", serde_json::to_string(ev).expect("serialize"));
        }
    }
    let after = model.weights_sha256();
    eprintln!("weights_sha256 before={before} after={after} unchanged={}", before == after);
    if before == after { ExitCode::SUCCESS } else { ExitCode::from(1) }
}
