use std::io::{BufRead, Write};
use std::process::ExitCode;
use zerolm_engine::bootstrap::{default_model, Session, CELL_BUDGET, COMMANDS, DEFAULT_KIND, DEFAULT_SEED, EMBED_DIM};

const USAGE: &str = "usage: zerolm info [--seed N]\n       zerolm run [--seed N] [--trace FILE]   (reads lines on stdin)\n\nprotocol lines: teach K => V | ask K | correct K => V | confirm K | forget K | repeat";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let (mut seed, mut trace) = (DEFAULT_SEED, None::<String>);
    while let Some(a) = args.next() {
        match (a.as_str(), args.next()) {
            ("--seed", Some(v)) => match v.parse() {
                Ok(n) => seed = n,
                Err(_) => return bad("--seed needs an integer"),
            },
            ("--trace", Some(p)) => trace = Some(p),
            _ => return bad("unknown or incomplete option"),
        }
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
        "run" => run(&model, trace),
        _ => bad("unknown command"),
    }
}

fn bad(msg: &str) -> ExitCode {
    eprintln!("error: {msg}\n{USAGE}");
    ExitCode::from(2)
}

fn run(model: &zerolm_engine::Model, trace: Option<String>) -> ExitCode {
    let before = model.weights_sha256();
    let mut session = Session::new(model);
    let mut sink = trace.map(|p| std::fs::File::create(p).expect("cannot create trace file"));
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().split(b'\n') {
        let Ok(raw) = line else { break };
        // Arbitrary bytes are accepted; invalid UTF-8 is replaced for protocol parsing only.
        let text = String::from_utf8_lossy(&raw).into_owned();
        let text = text.strip_suffix('\r').unwrap_or(&text);
        let reply = session.handle_line(text);
        let _ = writeln!(out, "{reply}");
        if let (Some(f), Some(ev)) = (sink.as_mut(), session.events().last()) {
            let _ = writeln!(f, "{}", serde_json::to_string(ev).expect("serialize"));
        }
    }
    let after = model.weights_sha256();
    eprintln!("weights_sha256 before={before} after={after} unchanged={}", before == after);
    if before == after { ExitCode::SUCCESS } else { ExitCode::from(1) }
}
