//! `zerolm-representation --root DATA [--mode smoke|pilot|main] [--out DIR] [--budget N]`
//!
//! Gate A of the roadmap (Phase 9). `smoke` is a tiny end-to-end check whose
//! outcome is always `smoke`; `pilot` tries the pre-registered configurations
//! on the development role and pilot seeds and never reads hidden data;
//! `main` needs a frozen spec (`model/specs/representation.json`) and reads
//! the hidden role once per seed.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use zerolm_data::corpus;
use zerolm_eval::ngram::{Kn, Wb};
use zerolm_eval::representation::*;
use zerolm_learning::online_predictor::Constants;
use zerolm_learning::plastic::{Core, EpropConfig, Plasticity, PlasticModel, TENSORS};

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

fn eval_json(e: &Eval) -> Value {
    json!({"bytes": e.bytes, "prequential_bpb": e.prequential_bpb, "heldout_bpb": e.heldout_bpb, "novel_context_bpb": e.novel_bpb, "ece": e.ece, "probe_bpb": e.probe_bpb, "probe_e1_bpb": e.probe_e1_bpb})
}

fn baselines(d: &Data, k: usize) -> (Vec<Value>, bool) {
    let stream: Vec<u8> = [&d.e1[..], &d.e2[..]].concat();
    let index = kgram_index(&stream, k);
    let mask = novel_positions(&d.hidden, &index, k);
    let (mut wb, mut kn) = (Wb::new(5), Kn::new(5));
    let mut wb3 = Wb::new(3);
    for i in 0..stream.len() {
        let h = &stream[i.saturating_sub(4)..i];
        wb.observe(h, stream[i]);
        kn.observe(h, stream[i]);
        wb3.observe(&stream[i.saturating_sub(2)..i], stream[i]);
    }
    let per = |f: &dyn Fn(&[u8], u8) -> f64| -> (f64, f64) {
        let bits: Vec<f32> = (0..d.hidden.len()).map(|i| -(f(&d.hidden[i.saturating_sub(4)..i], d.hidden[i]).max(1e-30)).log2() as f32).collect();
        (bits.iter().map(|b| *b as f64).sum::<f64>() / bits.len().max(1) as f64, masked_mean(&bits, &mask))
    };
    let rows = [
        ("witten-bell-3", per(&|h, b| wb3.prob(&h[h.len().saturating_sub(2)..], b))),
        ("witten-bell-5", per(&|h, b| wb.prob(h, b))),
        ("ppm-c-5", per(&|h, b| wb.ppm_prob(h, b))),
        ("kneser-ney-5", per(&|h, b| kn.prob(h, b))),
    ];
    // V4: the models are sane and deterministic: finite, and Kneser-Ney order 5 beats Witten-Bell order 3.
    let sane = rows.iter().all(|(_, (a, _))| a.is_finite() && *a > 0.0) && rows[3].1 .0 < rows[0].1 .0;
    (rows.iter().map(|(n, (h, nv))| json!({"name": n, "heldout_bpb": h, "novel_context_bpb": nv})).collect(), sane)
}

/// V1 in-process checks: determinism, disabled means no update, no future leak.
fn pipeline_checks(d: &Data, init: &Core, rp: &RunParams) -> Vec<(String, bool)> {
    let n = d.e1.len().min(30_000);
    let bytes = &d.e1[..n];
    let run = |b: &[u8], plast| {
        let mut m = PlasticModel::new(init.clone(), plast, rp.delta, rp.eprop);
        let bits: Vec<f64> = b.iter().map(|x| m.observe(*x).expect("finite")).collect();
        (m.core.p.clone(), bits)
    };
    let (a, bits_a) = run(bytes, Plasticity::Eprop);
    let (b, _) = run(bytes, Plasticity::Eprop);
    let (frozen, _) = run(bytes, Plasticity::Frozen);
    let mut other = bytes.to_vec();
    for x in other[n / 2..].iter_mut() {
        *x = b'~';
    }
    let (_, bits_o) = run(&other, Plasticity::Eprop);
    vec![
        ("determinism: a rerun reproduces every weight".into(), a == b),
        ("disabled means no update".into(), frozen == init.p),
        ("no future leak: bits before a changed byte are unchanged".into(), bits_a[..n / 2] == bits_o[..n / 2]),
        ("the e-prop arm changes the core".into(), a != init.p),
    ]
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let mode = arg("--mode").unwrap_or_else(|| "smoke".into());
    let repo = Path::new(".");
    let spec = Spec::load(repo).expect("spec");
    let manifest_text = std::fs::read_to_string(arg("--manifest").unwrap_or_else(|| "research/corpus/manifest.json".into())).expect("manifest");
    let out = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/representation".into()));
    std::fs::create_dir_all(&out).expect("out dir");
    let delta = Constants::default();
    let sizes = match mode.as_str() {
        "smoke" => &spec.smoke,
        "pilot" => &spec.pilot,
        "main" => {
            assert!(spec.frozen, "main mode needs a frozen spec (model/specs/representation.json)");
            &spec.main
        }
        other => panic!("unknown mode {other}"),
    };
    let max_shared = spec.thresholds.as_ref().map(|t| t.max_shared_window_fraction).unwrap_or(0.01);
    let data = load_data(&root, &manifest_text, sizes, mode != "main", max_shared).expect("data");
    println!("{mode}: E1 {} + E2 {} exposure bytes, probe {}, held-out {} (contamination ok: {})", data.e1.len(), data.e2.len(), data.probe.len(), data.hidden.len(), data.contamination_passed);
    let budget: usize = arg("--budget").map(|v| v.parse().expect("number")).unwrap_or(sizes.core_budget);
    let stream: Vec<u8> = [&data.e1[..], &data.e2[..]].concat();
    let index = kgram_index(&stream, spec.novel_k);
    let novel_fraction = novel_positions(&data.hidden, &index, spec.novel_k).iter().filter(|b| **b).count() as f64 / data.hidden.len().max(1) as f64;
    println!("novel-context positions (k = {}): {:.1}% of held-out", spec.novel_k, novel_fraction * 100.0);
    let seeds: Vec<u64> = match mode.as_str() {
        "main" => spec.main_seeds.clone(),
        "smoke" => spec.pilot_seeds[..1].to_vec(),
        _ => spec.pilot_seeds.clone(),
    };
    let window = 100_000.min((stream.len() / 8).max(1000));
    let mk = |eprop: &zerolm_eval::representation::EpropChoice, lr: f32| RunParams {
        delta,
        eprop: EpropConfig { eta: eprop.eta, max_core_update: eprop.max_core_update, feedback_seed: spec.feedback_seed },
        bptt_lr: lr,
        budget,
        novel_k: spec.novel_k,
        milestones: spec.milestones.clone(),
        window,
    };
    if mode == "pilot" {
        let budgets: Vec<usize> = if arg("--budget").is_some() { vec![budget] } else { spec.core_budgets.clone() };
        return pilot(&spec, &data, &index, &seeds, &mk, &budgets, &out);
    }
    let eprop = spec.eprop.clone().unwrap_or_else(|| spec.eprop_pilot_configs[0].clone());
    let lr = spec.bptt_lr.unwrap_or(spec.bptt_pilot_lrs[0]);
    let rp = mk(&eprop, lr);
    let (base_rows, base_sane) = std::thread::scope(|s| {
        let h = s.spawn(|| baselines(&data, spec.novel_k));
        h.join().expect("baselines")
    });
    // Seeds run in parallel; each seed runs its arms in parallel.
    type SeedOut = (u64, Vec<ArmRun>, (f64, f64), Core);
    let results: Vec<SeedOut> = std::thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&seed| {
                let (data, index, rp) = (&data, &index, &rp);
                s.spawn(move || {
                    let init = Core::from_seed_budget(seed, rp.budget);
                    let arms: Vec<ArmRun> = std::thread::scope(|s2| {
                        let hs: Vec<_> = [Arm::Frozen, Arm::RandomFeatures, Arm::Local, Arm::Bptt].into_iter().map(|a| {
                            let init = &init;
                            s2.spawn(move || {
                                let r = run_arm(a, seed, data, rp, index, init);
                                println!("seed {seed} arm {}: {} bytes in {:.0}s, final probe {:.3}", a.id(), r.bytes, r.wall_s, r.evals.last().map(|e| e.probe_bpb).unwrap_or(f64::NAN));
                                r
                            })
                        }).collect();
                        hs.into_iter().map(|h| h.join().expect("arm")).collect()
                    });
                    let a2 = arms.iter().find(|r| r.arm == Arm::Local).unwrap();
                    let pert = init.perturbed(&a2.change_norms, seed ^ 0xBEEF);
                    let pert_probe = probe_core(&pert, data, &rp.delta);
                    (seed, arms, pert_probe, init)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("seed")).collect()
    });
    let mut measures = Vec::new();
    let mut arm_json: Vec<Value> = Vec::new();
    for arm in [Arm::Frozen, Arm::RandomFeatures, Arm::Local, Arm::Bptt] {
        let per_seed: Vec<Value> = results
            .iter()
            .map(|(seed, arms, pert, _)| {
                let a1 = arms.iter().find(|r| r.arm == Arm::RandomFeatures).unwrap();
                let r = arms.iter().find(|r| r.arm == arm).unwrap();
                let last = r.evals.last().expect("evals");
                let a1l = a1.evals.last().unwrap();
                let e1_at = |run: &ArmRun| run.evals.iter().find(|e| e.bytes == data.e1.len()).map(|e| e.probe_e1_bpb);
                let kappa = match (e1_at(a1), e1_at(r)) {
                    (Some(a), Some(b)) if a - b > 1e-9 => (a1l.probe_e1_bpb - last.probe_e1_bpb) / (a - b),
                    _ => f64::NAN,
                };
                let _ = pert;
                json!({"seed": seed, "prequential_bpb": last.prequential_bpb, "heldout_bpb": last.heldout_bpb, "probe_bpb": last.probe_bpb, "delta_r": a1l.probe_bpb - last.probe_bpb, "novel_context_bpb": last.novel_bpb, "ece": last.ece,
                    "retention": {"kappa": kappa, "delta_r_e1_after_e1": e1_at(a1).zip(e1_at(r)).map(|(a, b)| a - b), "delta_r_e1_after_e2": a1l.probe_e1_bpb - last.probe_e1_bpb},
                    "curve": r.evals.iter().map(eval_json).collect::<Vec<_>>(), "finite": r.finite, "max_abs_weight": r.max_abs_weight, "late_window_degradation": late_degradation(&r.windows),
                    "bytes_per_s": r.bytes_per_s, "wall_s": r.wall_s, "peak_rss_kb": r.peak_rss_kb})
            })
            .collect();
        let changed: Vec<&str> = match arm {
            Arm::Frozen => vec![],
            Arm::RandomFeatures => vec!["readout.weight", "readout.bias"],
            _ => TENSORS.iter().copied().chain(["readout.weight", "readout.bias"]).collect(),
        };
        arm_json.push(json!({"id": arm.id(), "control": arm == Arm::Bptt, "changed_tensors": changed, "per_seed": per_seed}));
    }
    // Perturbed arm: probe-only.
    let pert_seed: Vec<Value> = results.iter().map(|(seed, arms, pert, _)| {
        let a1l = arms.iter().find(|r| r.arm == Arm::RandomFeatures).unwrap().evals.last().unwrap();
        json!({"seed": seed, "probe_bpb": pert.0, "delta_r": a1l.probe_bpb - pert.0, "probe_e1_bpb": pert.1})
    }).collect();
    arm_json.push(json!({"id": "local_perturbed", "control": false, "changed_tensors": TENSORS, "per_seed": pert_seed}));
    for (seed, arms, pert, _) in &results {
        let get = |a: Arm| arms.iter().find(|r| r.arm == a).unwrap();
        let (a1, a2, a3) = (get(Arm::RandomFeatures), get(Arm::Local), get(Arm::Bptt));
        let (l1, l2, l3) = (a1.evals.last().unwrap(), a2.evals.last().unwrap(), a3.evals.last().unwrap());
        let e1 = |r: &ArmRun| r.evals.iter().find(|e| e.bytes == data.e1.len()).map(|e| e.probe_e1_bpb).unwrap_or(f64::NAN);
        let kappa = (l1.probe_e1_bpb - l2.probe_e1_bpb) / (e1(a1) - e1(a2));
        measures.push(SeedMeasures {
            delta_r_local: l1.probe_bpb - l2.probe_bpb,
            delta_r_bptt: l1.probe_bpb - l3.probe_bpb,
            delta_r_perturbed: l1.probe_bpb - pert.0,
            novel_gain_local: l1.novel_bpb - l2.novel_bpb,
            kappa_local: kappa,
            finite: a2.finite && a3.finite,
            late_degradation_local: late_degradation(&a2.windows),
        });
        let _ = seed;
    }
    // Validity V1 and V4.
    let init0 = Core::from_seed_budget(seeds[0], budget);
    let checks = pipeline_checks(&data, &init0, &rp);
    let v1_ok = data.contamination_passed && checks.iter().all(|c| c.1);
    let mut validity_json = vec![
        json!({"id": "V1", "passed": v1_ok, "detail": format!("manifest hashes verified; contamination {}; {}", data.contamination, checks.iter().map(|c| format!("{}: {}", c.0, c.1)).collect::<Vec<_>>().join("; "))}),
        json!({"id": "V4", "passed": base_sane, "detail": "baselines finite and ordered (Kneser-Ney 5 below Witten-Bell 3) on this stream; the n-gram code is the Phase 8 code (crates/zerolm-eval/src/ngram.rs)"}),
    ];
    let thresholds = spec.thresholds.clone().unwrap_or(Thresholds { h: 0.0, r: 0.0, rho_min: 0.0, kappa_min: 0.0, s: f64::INFINITY, max_shared_window_fraction: max_shared });
    let (validity, criteria) = decide(&measures, &thresholds);
    let (mut outcome_name, mut deciding) = outcome(&validity, &criteria, spec.largest_affordable);
    if !v1_ok || !base_sane {
        outcome_name = "inconclusive";
        deciding = vec![if !v1_ok { "V1" } else { "V4" }];
    }
    if mode != "main" {
        outcome_name = "smoke";
        deciding = vec![];
    }
    let chk = |c: &Check| json!({"id": c.id, "passed": c.passed, "value": c.value, "threshold": c.threshold, "detail": c.detail});
    validity_json.extend(validity.iter().map(|c| json!({"id": c.id, "passed": c.passed, "detail": c.detail})));
    let rho = {
        let (a, b) = (mean_sd(&measures.iter().map(|m| m.delta_r_local).collect::<Vec<_>>()).0, mean_sd(&measures.iter().map(|m| m.delta_r_bptt).collect::<Vec<_>>()).0);
        a / b
    };
    let resources: Vec<Value> = results.iter().flat_map(|(seed, arms, _, _)| arms.iter().map(move |r| {
        let (e, d) = (r.final_core.e, r.final_core.d);
        let gru = 3 * d * e + 3 * d * d;
        let readout = 256 * (d + e);
        let macs = match r.arm { Arm::Frozen => gru + readout, Arm::RandomFeatures => gru + 2 * readout, Arm::Local => 4 * gru + 2 * readout, Arm::Bptt => 3 * gru + 3 * readout };
        json!({"arm": r.arm.id(), "seed": seed, "bytes": r.bytes, "bytes_per_s": r.bytes_per_s, "macs_per_byte": macs, "peak_rss_bytes": r.peak_rss_kb.map(|k| k * 1024).unwrap_or(0), "wall_s": r.wall_s, "macs_note": "analytic estimate of multiply-adds per byte (forward, traces or backward, readout and its update)"})
    })).collect();
    let summary = json!({
        "report": "representation", "mode": mode, "outcome": outcome_name, "deciding_criteria": deciding,
        "decision_record": Value::Null,
        "spec_sha256": Spec::sha256(repo),
        "data": {"manifest_sha256": data.manifest_sha256, "contamination_passed": data.contamination_passed, "contamination": data.contamination,
            "splits": [
                {"role": "exposure_1", "bytes": data.e1.len(), "sha256": corpus::sha256_hex(&data.e1)},
                {"role": "exposure_2", "bytes": data.e2.len(), "sha256": corpus::sha256_hex(&data.e2)},
                {"role": "probe", "bytes": data.probe.len(), "sha256": corpus::sha256_hex(&data.probe)},
                {"role": if mode == "main" { "hidden" } else { "development" }, "bytes": data.hidden.len(), "sha256": corpus::sha256_hex(&data.hidden)}],
            "novel_context": {"k": spec.novel_k, "fraction_of_held_out": novel_fraction}},
        "arms": arm_json, "baselines": base_rows, "validity": validity_json,
        "criteria": criteria.iter().map(chk).collect::<Vec<_>>(),
        "headroom_ratio_rho": rho,
        "attempts": [{"id": format!("{mode}-{}", spec.eprop_pilot_configs.len()), "mode": mode, "rule": "eprop", "outcome": outcome_name}],
        "resources": resources,
        "core_budget": budget,
        "scope_note": "Gate A evidence: one decision, three main seeds, hidden split read once. A smoke or pilot run is never a decision. The backpropagation arm is a CONTROL and is excluded from strict results."
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).expect("write");
    let mut brief = format!("## Representation feasibility (Gate A), mode {mode}, outcome **{outcome_name}**\n\n| arm | final probe BPB (mean) | Delta R (mean) | held-out BPB | novel-context BPB |\n|---|---|---|---|---|\n");
    for a in &arm_json {
        let ps = a["per_seed"].as_array().unwrap();
        let mean = |k: &str| { let v: Vec<f64> = ps.iter().filter_map(|p| p[k].as_f64()).collect(); if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 } };
        brief += &format!("| {} | {:.3} | {:.3} | {:.3} | {:.3} |\n", a["id"].as_str().unwrap(), mean("probe_bpb"), mean("delta_r"), mean("heldout_bpb"), mean("novel_context_bpb"));
    }
    brief += &format!("\nHeadroom ratio rho = {rho:.3}. ");
    for c in criteria.iter().chain(validity.iter()) {
        brief += &format!("\n- {} {}: value {:.4}, threshold {:.4}. {}", c.id, if c.passed { "passed" } else { "FAILED" }, c.value, c.threshold, c.detail);
    }
    brief.push('\n');
    std::fs::write(out.join("brief.md"), &brief).expect("write");
    print!("{brief}");
}

/// Pilot: every pre-registered configuration on the development role and the pilot seeds. Nothing here
/// reads hidden data or decides anything; every configuration tried is reported, including failures.
fn pilot(spec: &Spec, data: &Data, index: &[u64], seeds: &[u64], mk: &dyn Fn(&EpropChoice, f32) -> RunParams, budgets: &[usize], out: &Path) {
    let mut attempts: Vec<Value> = Vec::new();
    for &budget in budgets {
        let rows: Vec<Value> = std::thread::scope(|s| {
            let mut hs = Vec::new();
            for &seed in seeds {
                let mut jobs: Vec<(String, Arm, usize, usize)> = vec![("A1".into(), Arm::RandomFeatures, 0, 0)];
                for i in 0..spec.eprop_pilot_configs.len() {
                    jobs.push((format!("A2 config {i}"), Arm::Local, i, 0));
                }
                for j in 0..spec.bptt_pilot_lrs.len() {
                    jobs.push((format!("A3 lr {}", spec.bptt_pilot_lrs[j]), Arm::Bptt, 0, j));
                }
                for (label, arm, ei, li) in jobs {
                    let mut rp = mk(&spec.eprop_pilot_configs[ei], spec.bptt_pilot_lrs[li]);
                    rp.budget = budget;
                    hs.push(s.spawn(move || {
                        let init = Core::from_seed_budget(seed, budget);
                        let r = run_arm(arm, seed, data, &rp, index, &init);
                        let e = r.evals.last().expect("evals");
                        println!("budget {budget} seed {seed} {label}: probe {:.3}, held-out {:.3}, {:.0}s", e.probe_bpb, e.heldout_bpb, r.wall_s);
                        let norms = r.change_norms.clone();
                        let pert = if arm == Arm::Local { Some(probe_core(&init.perturbed(&norms, seed ^ 0xBEEF), data, &rp.delta).0) } else { None };
                        json!({"budget": budget, "seed": seed, "label": label, "arm": arm.id(), "eta": rp.eprop.eta, "max_core_update": rp.eprop.max_core_update, "bptt_lr": rp.bptt_lr,
                            "probe_bpb": e.probe_bpb, "heldout_bpb": e.heldout_bpb, "novel_context_bpb": e.novel_bpb, "prequential_bpb": e.prequential_bpb, "finite": r.finite,
                            "probe_bpb_perturbed": pert, "wall_s": r.wall_s, "bytes_per_s": r.bytes_per_s, "curve": r.evals.iter().map(eval_json).collect::<Vec<_>>()})
                    }));
                }
            }
            hs.into_iter().map(|h| h.join().expect("pilot arm")).collect()
        });
        attempts.extend(rows);
    }
    std::fs::write(out.join("pilot.json"), serde_json::to_string_pretty(&json!({"report": "representation-pilot", "outcome": "pilot", "development_bytes": data.hidden.len(), "attempts": attempts, "note": "development role and pilot seeds only; every pre-registered configuration is listed"})).unwrap()).expect("write");
    // Summary table: Delta R per configuration (A1 probe minus the arm's probe, per budget and seed).
    let mut brief = String::from("## Gate A pilot (development role, pilot seeds)\n\n| budget | seed | configuration | probe BPB | Delta R vs A1 | perturbed Delta R | held-out BPB |\n|---|---|---|---|---|---|---|\n");
    for a in &attempts {
        let a1 = attempts.iter().find(|x| x["arm"] == "random_features" && x["budget"] == a["budget"] && x["seed"] == a["seed"]).unwrap();
        let dr = a1["probe_bpb"].as_f64().unwrap() - a["probe_bpb"].as_f64().unwrap();
        let pd = a["probe_bpb_perturbed"].as_f64().map(|p| format!("{:.3}", a1["probe_bpb"].as_f64().unwrap() - p)).unwrap_or_else(|| "-".into());
        brief += &format!("| {} | {} | {} | {:.3} | {:.3} | {} | {:.3} |\n", a["budget"], a["seed"], a["label"].as_str().unwrap(), a["probe_bpb"].as_f64().unwrap(), dr, pd, a["heldout_bpb"].as_f64().unwrap());
    }
    std::fs::write(out.join("pilot.md"), &brief).expect("write");
    print!("{brief}");
}
