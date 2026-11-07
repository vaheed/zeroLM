//! Uniform-engine gate (constitution invariant 10, roadmap "uniform neural
//! engine and WSP compiler").
//!
//! Checks:
//! - every shipped circuit compiles reproducibly and matches its pinned hash;
//! - every shipped circuit survives a patch round trip (apply then revert
//!   restores the hash; a second apply is refused);
//! - memory queries and every recorded reasoning circuit run are reproduced
//!   bit for bit by the independent reference interpreter from the exported
//!   bundles (the run includes operator ablations, so zeroed blocks are
//!   replayed too);
//! - the host audit: model-path files process no input text, and the list
//!   of host code still making content decisions is reported.
//!
//! `passed` means every check holds. `complete` additionally requires an
//! empty pending list: only then is all model behavior in weights.

use crate::isa_ref::{self, T};
use crate::reasoning;
use serde::Serialize;
use std::path::Path;
use zerolm_engine::isa::{Buf, Outputs};
use zerolm_memory::{MemoryConfig, MemoryStore};
use zerolm_reasoner::{Ops, Reasoner};
use zerolm_wsp::{compile, parse_spec, patch, SPECS};

pub const HOST_AUDIT: &str = "model/prog/host-audit.json";

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CircuitRow {
    pub id: String,
    pub sha256: String,
    pub pinned: bool,
    pub reproducible: bool,
    pub patch_round_trip: bool,
    pub blocks: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Parity {
    pub runs: usize,
    pub mismatches: usize,
    /// Runs per program id.
    pub by_program: std::collections::BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Pending {
    pub file: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostAudit {
    pub model_path: Vec<String>,
    pub violations: Vec<String>,
    pub pending: Vec<Pending>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Report kind (always "uniform-engine").
    pub report: &'static str,
    pub passed: bool,
    /// All checks pass and no host code is pending.
    pub complete: bool,
    pub failures: Vec<String>,
    pub checks: Vec<Check>,
    pub circuits: Vec<CircuitRow>,
    pub memory_parity: Parity,
    pub reasoning_parity: Parity,
    pub host_audit: HostAudit,
    pub scope_note: &'static str,
}

pub const SCOPE_NOTE: &str = "Bit-exact parity shows that the compiled programs and their weights determine every circuit output; it does not show that all model behavior is in weights. Host code listed as pending still makes content-dependent decisions.";

fn to_t(b: &Buf) -> T {
    T { r: b.rows, c: b.cols, d: b.data.clone() }
}

/// Every output of the engine equals the reference output bit for bit.
fn same(engine: &Outputs, reference: &std::collections::BTreeMap<String, T>) -> bool {
    engine.values.iter().all(|(n, b)| {
        reference.get(n).map(|t| t.r == b.rows && t.c == b.cols && t.d.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits())).unwrap_or(false)
    })
}

fn circuits() -> Result<Vec<CircuitRow>, String> {
    let pins = zerolm_wsp::pinned()?;
    let mut rows = Vec::new();
    for (_, json) in SPECS {
        let spec = parse_spec(json)?;
        let a = compile(&spec).map_err(|e| e.to_string())?;
        let b = compile(&spec).map_err(|e| e.to_string())?;
        let sha = a.sha256();
        let pinned = pins.iter().any(|(id, p)| *id == spec.id && *p == sha);
        // Patch round trip: change one constant, apply, refuse a second apply, revert.
        let mut target = a.clone();
        let t = target.tensors.iter_mut().find(|t| !t.data.is_empty()).ok_or("circuit without constants")?;
        t.data[0] += 1.0;
        let p = patch::diff(&a, &target, "uniform-round-trip", "gate check").map_err(|e| e.to_string())?;
        let mut c = a.clone();
        let applied = patch::apply(&mut c, &p).is_ok() && c.sha256() == target.sha256();
        let refused = patch::apply(&mut c, &p).is_err();
        let reverted = patch::revert(&mut c, &p).is_ok() && c.sha256() == sha && c == a;
        rows.push(CircuitRow { id: spec.id, sha256: sha.clone(), pinned, reproducible: sha == b.sha256(), patch_round_trip: applied && refused && reverted, blocks: a.blocks() });
    }
    Ok(rows)
}

/// Memory: teach generated facts, then replay exact and negative queries.
fn memory_parity() -> Result<Parity, String> {
    let mut m = MemoryStore::new(MemoryConfig::default()).map_err(|e| e.to_string())?;
    let n = 300;
    for i in 0..n {
        m.teach(&format!("entity{i} attribute{}", i % 7), &format!("value{i}"), "uniform:synthetic-facts").map_err(|e| e.to_string())?;
    }
    let (enc, read) = (m.encode_bundle(), m.read_bundle());
    let mut p = Parity::default();
    let queries: Vec<String> = (0..n).map(|i| format!("entity{i} attribute{}", i % 7)).chain((0..100).map(|i| format!("entity{i} unrelated{i}"))).collect();
    for q in queries {
        let (enc_ok, read_ok) = crate::memory::replay_query(&m, &enc, &read, &q)?;
        for (id, ok) in [("memory-encode", enc_ok), ("memory-read", read_ok)] {
            p.runs += 1;
            p.mismatches += !ok as usize;
            *p.by_program.entry(id.into()).or_default() += 1;
        }
    }
    Ok(p)
}

/// Reasoning: answer a generated world's questions (intact, then with each
/// operator ablated in turn on a subset), record every circuit run, and
/// replay all of them on the reference interpreter in parallel.
fn reasoning_parity(root: &Path) -> Result<Parity, String> {
    let spec = reasoning::Spec::load(root)?;
    let seed = *spec.main_seeds.first().ok_or("no main seed")?;
    let world = reasoning::generate(&spec, seed);
    let mut r = Reasoner::new(spec.config).map_err(|e| e.to_string())?;
    for s in &world.statements {
        r.observe(&s.text, "uniform:synthetic-world").map_err(|e| e.to_string())?;
    }
    r.record_circuits(true);
    for q in &world.questions {
        r.ask(&q.text);
    }
    for name in Ops::NAMES {
        r.ops = Ops::without(name);
        for q in world.questions.iter().step_by(10) {
            r.ask(&q.text);
        }
    }
    r.ops = Ops::ALL;
    let recorded = r.take_recorded();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32);
    let chunk = recorded.len().div_ceil(threads).max(1);
    let results: Vec<Result<Vec<bool>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = recorded
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|rec| {
                            let inputs: Vec<(&str, T)> = rec.inputs.iter().map(|(n, b)| (n.as_str(), to_t(b))).collect();
                            isa_ref::run(&rec.bundle, &inputs).map(|reference| same(&rec.outputs, &reference))
                        })
                        .collect::<Result<Vec<bool>, String>>()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("replay thread panicked".into()))).collect()
    });
    let mut p = Parity::default();
    for (rec, ok) in recorded.iter().zip(results.into_iter().collect::<Result<Vec<_>, _>>()?.into_iter().flatten()) {
        p.runs += 1;
        p.mismatches += !ok as usize;
        *p.by_program.entry(rec.program.clone()).or_default() += 1;
    }
    Ok(p)
}

#[derive(serde::Deserialize)]
struct AuditSpec {
    model_path: Vec<String>,
    forbidden: Vec<String>,
    pending: Vec<Pending>,
}

fn host_audit(root: &Path) -> Result<HostAudit, String> {
    let text = std::fs::read_to_string(root.join(HOST_AUDIT)).map_err(|e| format!("{HOST_AUDIT}: {e}"))?;
    let spec: AuditSpec = serde_json::from_str(&text).map_err(|e| format!("{HOST_AUDIT}: {e}"))?;
    let mut violations = Vec::new();
    for f in spec.model_path.iter().chain(spec.pending.iter().map(|p| &p.file)) {
        if !root.join(f).is_file() {
            violations.push(format!("{f}: listed but missing"));
        }
    }
    for f in &spec.model_path {
        let Ok(src) = std::fs::read_to_string(root.join(f)) else { continue };
        for (n, line) in src.lines().enumerate() {
            for pat in &spec.forbidden {
                if line.contains(pat.as_str()) {
                    violations.push(format!("{f}:{}: text processing `{pat}` on the model path", n + 1));
                }
            }
        }
    }
    Ok(HostAudit { model_path: spec.model_path, violations, pending: spec.pending })
}

pub fn run(root: &Path) -> Result<Report, String> {
    let circuits = circuits()?;
    let memory_parity = memory_parity()?;
    let reasoning_parity = reasoning_parity(root)?;
    let host_audit = host_audit(root)?;
    let mut checks = Vec::new();
    let mut check = |name, passed, detail: String| checks.push(Check { name, passed, detail });
    let ids = |f: &dyn Fn(&CircuitRow) -> bool| circuits.iter().filter(|c| !f(c)).map(|c| c.id.clone()).collect::<Vec<_>>();
    check("circuits_pinned_and_reproducible", circuits.iter().all(|c| c.pinned && c.reproducible), format!("{} circuits; failing: {:?}", circuits.len(), ids(&|c| c.pinned && c.reproducible)));
    check("patch_round_trip", circuits.iter().all(|c| c.patch_round_trip), format!("failing: {:?}", ids(&|c| c.patch_round_trip)));
    check("memory_reference_parity", memory_parity.runs > 0 && memory_parity.mismatches == 0, format!("{}/{} runs bit-exact", memory_parity.runs - memory_parity.mismatches, memory_parity.runs));
    check(
        "reasoning_reference_parity",
        reasoning_parity.runs > 0 && reasoning_parity.mismatches == 0,
        format!("{}/{} runs bit-exact ({:?})", reasoning_parity.runs - reasoning_parity.mismatches, reasoning_parity.runs, reasoning_parity.by_program),
    );
    check("host_audit_model_path", host_audit.violations.is_empty(), format!("{} violations", host_audit.violations.len()));
    let failures: Vec<String> = checks.iter().filter(|c| !c.passed).map(|c| format!("{}: {}", c.name, c.detail)).chain(host_audit.violations.iter().cloned()).collect();
    let passed = failures.is_empty();
    Ok(Report {
        report: "uniform-engine",
        passed,
        complete: passed && host_audit.pending.is_empty(),
        failures,
        checks,
        circuits,
        memory_parity,
        reasoning_parity,
        host_audit,
        scope_note: SCOPE_NOTE,
    })
}

pub fn brief_md(r: &Report) -> String {
    let status = match (r.passed, r.complete) {
        (true, true) => "PASSED (milestone complete)",
        (true, false) => "PASSED (milestone in progress)",
        _ => "FAILED",
    };
    let mut s = format!("## Uniform engine: {status}\n\n| check | result |\n|---|---|\n");
    for c in &r.checks {
        s += &format!("| {} | {} — {} |\n", c.name.replace('_', " "), if c.passed { "yes" } else { "**no**" }, c.detail);
    }
    s += &format!("\nCompiled circuits: {}.\n", r.circuits.iter().map(|c| format!("`{}`", c.id)).collect::<Vec<_>>().join(", "));
    if !r.host_audit.pending.is_empty() {
        s += &format!("\nStill decided by host code ({} files, to move into circuits):\n\n", r.host_audit.pending.len());
        for p in &r.host_audit.pending {
            s += &format!("- `{}`: {}\n", p.file, p.reason);
        }
    }
    if !r.failures.is_empty() {
        s += "\n**Failures**\n\n";
        for f in &r.failures {
            s += &format!("- {f}\n");
        }
    }
    s
}
