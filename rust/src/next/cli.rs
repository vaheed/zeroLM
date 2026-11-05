//! `zerolm next ...` command line: parse, run (parse + execute), batch, trace, repl, info, probe.

use super::model::{Ablation, NextModel};
use super::trace::{num, obj};
use super::world::{self, World};
use crate::jsonl::{self, escape};
use std::env;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "Usage: zerolm next <command> [options] [TEXT...]
Commands:
  parse   TEXT        structured operations as JSON
  run     TEXT        parse, then execute on --world FILE (default: built-in demo world)
  trace   TEXT        full forward-pass trace as JSON
  batch   [FILE|-]    JSON lines in ({\"id\",\"text\",\"world\"?}), JSON lines out; --summary FILE writes timing
  repl                interactive: one instruction per line
  probe   WORD...     show the embedding of a token
  info                dimensions, parameter counts, checksum
Options:
  --model FILE        weights (default $ZEROLM_NEXT_MODEL or model/next/weights.zln)
  --ablate SPEC       recurrence | state_read | binding | pointer_shift | time_order | zero:T | permute:T[:SEED] | shuffle:SEED | random:SEED (repeatable)
  --world FILE        world JSON for `run`";

const DEMO_WORLD: &str = r#"{"width":8,"height":8,"objects":[{"id":1,"type":"block","color":"red","x":4,"y":4},{"id":2,"type":"circle","color":"blue","x":2,"y":6},{"id":3,"type":"triangle","color":"green","x":6,"y":1},{"id":4,"type":"star","color":"yellow","x":0,"y":0}]}"#;

fn parse_json_out(p: &super::model::Parse) -> String {
    let d = &p.decoded;
    let ops: Vec<String> = d.ops.iter().map(|o| o.to_json()).collect();
    let errs: Vec<String> = d.error_units.iter().map(|(n, _)| format!("\"{}\"", escape(n))).collect();
    format!(
        "\"accepted\":{},\"error_mass\":{},\"errors\":[{}],\"ops\":[{}]",
        d.accepted, num(d.error_total), errs.join(","), ops.join(",")
    )
}

fn peak_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse().ok()
}

pub fn run(args: &[String]) -> ExitCode {
    let fail = |m: String| {
        eprintln!("error: {m}");
        ExitCode::from(2)
    };
    let mut model_path = env::var("ZEROLM_NEXT_MODEL").unwrap_or_else(|_| "model/next/weights.zln".into());
    let (mut ablate, mut world_path, mut summary) = (Vec::<String>::new(), None::<String>, None::<String>);
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "--ablate" | "--world" | "--summary" => {
                let Some(v) = args.get(i + 1) else { return fail(format!("{} needs a value", args[i])) };
                match args[i].as_str() {
                    "--model" => model_path = v.clone(),
                    "--ablate" => ablate.push(v.clone()),
                    "--world" => world_path = Some(v.clone()),
                    _ => summary = Some(v.clone()),
                }
                i += 2;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => {
                rest.push(args[i].clone());
                i += 1;
            }
        }
    }
    let ab = match Ablation::parse(&ablate) {
        Ok(a) => a,
        Err(e) => return fail(e),
    };
    let model = match NextModel::load(&model_path, ab) {
        Ok(m) => m,
        Err(e) => return fail(format!("model error: {e}")),
    };
    let Some(cmd) = rest.first().cloned() else {
        println!("{USAGE}");
        return ExitCode::from(2);
    };
    let text = rest[1..].join(" ");
    let load_world = || -> Result<World, String> {
        match &world_path {
            Some(p) => World::parse_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?)
                .ok_or(format!("{p}: not a valid world")),
            None => Ok(World::parse_str(DEMO_WORLD).unwrap()),
        }
    };
    match cmd.as_str() {
        "parse" => {
            let p = model.parse(&text, false);
            println!("{{{}}}", parse_json_out(&p));
            ExitCode::SUCCESS
        }
        "run" => {
            let w = match load_world() {
                Ok(w) => w,
                Err(e) => return fail(e),
            };
            let p = model.parse(&text, false);
            let (after, status) = if p.decoded.accepted { world::execute(&w, &p.decoded.ops) } else { (w.clone(), vec![]) };
            let st: Vec<String> = status.iter().map(|s| format!("\"{s}\"")).collect();
            println!("{{{},\"status\":[{}],\"world_before\":{},\"world_after\":{}}}", parse_json_out(&p), st.join(","), w.to_json(), after.to_json());
            ExitCode::SUCCESS
        }
        "trace" => {
            let p = model.parse(&text, true);
            println!("{}", p.trace.as_ref().unwrap().to_json(&p.tokens, &p.decoded));
            ExitCode::SUCCESS
        }
        "probe" => {
            for w in &rest[1..] {
                let e = model.embed(&w.to_lowercase());
                let names = model.net.list("features");
                let nz: Vec<(String, f64)> = e.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(i, v)| (names[i].clone(), *v)).collect();
                println!("{{\"token\":\"{}\",\"embedding\":{}}}", escape(w), obj(&nz));
            }
            ExitCode::SUCCESS
        }
        "info" => {
            let (total, nz) = model.net.param_count();
            let mut h: u64 = 0xcbf29ce484222325;
            if let Ok(bytes) = std::fs::read(&model_path) {
                for b in bytes {
                    h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                }
            }
            println!("ZeroLM Next {}", model.net.meta.get("version").map_or("?", |s| s));
            println!("Training: none | pretrained weights: none | runtime: CPU only, std only");
            println!("Dimensions: token_features={} row_fields={} rows={} vocabulary={}", model.d, model.k, model.n, model.net.list("vocab").len());
            println!("Hidden units: boundary={} write={} validity={}", model.net.list("A_units").len(), model.net.list("B_units").len(), model.net.list("V_units").len());
            println!("Weights: {total} slots, {nz} non-zero");
            println!("fnv1a64: {h:016x}");
            ExitCode::SUCCESS
        }
        "repl" => {
            let w = match load_world() {
                Ok(w) => w,
                Err(e) => return fail(e),
            };
            let mut world_now = w;
            println!("ZeroLM Next - one instruction per line, /reset restores the world, /quit exits.");
            for line in io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                let line = line.trim();
                match line {
                    "" => continue,
                    "/quit" => break,
                    "/reset" => {
                        world_now = load_world().unwrap();
                        println!("world reset");
                        continue;
                    }
                    _ => {}
                }
                let p = model.parse(line, false);
                if p.decoded.accepted {
                    let (after, status) = world::execute(&world_now, &p.decoded.ops);
                    world_now = after;
                    let st: Vec<String> = status.iter().map(|s| format!("\"{s}\"")).collect();
                    println!("{{{},\"status\":[{}],\"world\":{}}}", parse_json_out(&p), st.join(","), world_now.to_json());
                } else {
                    println!("{{{}}}", parse_json_out(&p));
                }
            }
            ExitCode::SUCCESS
        }
        "batch" => {
            let src: Box<dyn BufRead> = match rest.get(1).map(String::as_str) {
                None | Some("-") => Box::new(io::BufReader::new(io::stdin())),
                Some(p) => match std::fs::File::open(p) {
                    Ok(f) => Box::new(io::BufReader::new(f)),
                    Err(e) => return fail(format!("{p}: {e}")),
                },
            };
            let out = io::stdout();
            let mut out = io::BufWriter::new(out.lock());
            let mut lat: Vec<f64> = Vec::new();
            let t_all = Instant::now();
            for line in src.lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let Some(j) = jsonl::parse(&line) else {
                    return fail("batch line is not valid JSON".into());
                };
                let id = j.get("id").and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_f64().map(num))).unwrap_or_default();
                let text = j.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let t0 = Instant::now();
                let p = model.parse(text, false);
                lat.push(t0.elapsed().as_secs_f64() * 1e6);
                let mut extra = String::new();
                if let Some(w) = j.get("world").and_then(World::from_json) {
                    if p.decoded.accepted {
                        let (after, status) = world::execute(&w, &p.decoded.ops);
                        let st: Vec<String> = status.iter().map(|s| format!("\"{s}\"")).collect();
                        extra = format!(",\"status\":[{}],\"world_after\":{}", st.join(","), after.to_json());
                    }
                }
                let _ = writeln!(out, "{{\"id\":\"{}\",{}{}}}", escape(&id), parse_json_out(&p), extra);
            }
            let _ = out.flush();
            if let Some(sp) = summary {
                lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = lat.len().max(1);
                let pct = |q: f64| lat.get(((n as f64 - 1.0) * q) as usize).copied().unwrap_or(0.0);
                let s = format!(
                    "{{\"cases\":{},\"wall_s\":{:.3},\"parse_us_mean\":{:.1},\"parse_us_p50\":{:.1},\"parse_us_p99\":{:.1},\"peak_rss_kb\":{},\"weights_bytes\":{}}}\n",
                    lat.len(), t_all.elapsed().as_secs_f64(), lat.iter().sum::<f64>() / n as f64, pct(0.5), pct(0.99),
                    peak_rss_kb().map_or("null".into(), |v| v.to_string()),
                    std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0)
                );
                let _ = std::fs::write(sp, s);
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
