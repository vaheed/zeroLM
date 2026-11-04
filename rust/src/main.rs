use std::env;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use zerolm::{Model, C_MODEL_STRUCT_BYTES, MAX_STEPS, MAX_TEXT};

const DEFAULT_MODEL: &str = "weights.zlm";

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

/// `zerolm live [--model FILE] [--life FILE] [--capacity N]`: a long-lived
/// session whose memory persists across restarts in the life file.
fn run_live(args: &[String]) -> ExitCode {
    let (mut model_path, mut life, mut capacity) =
        (DEFAULT_MODEL.to_string(), "life.zlm".to_string(), zerolm::memory::DEFAULT_CAPACITY);
    let mut i = 0;
    while i < args.len() {
        let val = args.get(i + 1);
        match (args[i].as_str(), val) {
            ("--model", Some(v)) => model_path = v.clone(),
            ("--life", Some(v)) => life = v.clone(),
            ("--capacity", Some(v)) => match v.parse::<usize>() {
                Ok(n) if n >= 1 => capacity = n,
                _ => {
                    eprintln!("error: --capacity needs a positive integer");
                    return ExitCode::from(2);
                }
            },
            _ => {
                eprintln!("Usage: zerolm live [--model FILE] [--life FILE] [--capacity N]");
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
    let path = std::path::PathBuf::from(&life);
    let memory = match zerolm::memory::Memory::load(&path, capacity) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("life error: {e}");
            return ExitCode::from(2);
        }
    };
    let age = memory.clock;
    let mut session = zerolm::live::Session::new(&model, memory, Some(path));
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

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("live") {
        return run_live(&args[2..]);
    }
    let (mut model_path, mut info, mut test, mut verbose, mut faithful) =
        (DEFAULT_MODEL.to_string(), false, false, false, false);
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
