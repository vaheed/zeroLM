//! `zerolm-gradient-gate --root DATA [--mode smoke|main] [--out DIR]`
//!
//! The Gate A rerun with the mainline online-gradient rule (decision record
//! `research/decisions/gate-a-2026-11-26.md`, option (a)). Arms: A0 frozen, A1 random
//! features, A2 single-pass truncated BPTT in the mainline (every update logged), and
//! A2-perturbed. The held-out role is two sources held back whole, which no earlier run saw.
//! `smoke` is a tiny end-to-end check (its outcome is always `smoke`); `main` reads the fresh
//! held-out books once. Spec: `model/specs/representation-gradient.json`.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use zerolm_data::corpus;
use zerolm_eval::bptt::{Bptt, Config as BpttConfig};
use zerolm_eval::ngram::{Kn, Wb};
use zerolm_eval::representation::*;
use zerolm_learning::online_predictor::Constants;
use zerolm_learning::plastic::{Core, EpropConfig, TENSORS};

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

fn eval_json(e: &Eval) -> Value {
    json!({"bytes": e.bytes, "prequential_bpb": e.prequential_bpb, "heldout_bpb": e.heldout_bpb, "novel_context_bpb": e.novel_bpb, "ece": e.ece, "probe_bpb": e.probe_bpb, "probe_e1_bpb": e.probe_e1_bpb})
}

/// Baselines on the same stream; the first row is byte frequency (Witten-Bell order 1).
fn baselines(d: &Data, k: usize) -> (Vec<Value>, f64, bool) {
    let stream: Vec<u8> = [&d.e1[..], &d.e2[..]].concat();
    let mask = novel_positions(&d.hidden, &kgram_index(&stream, k), k);
    let (mut wb1, mut wb3, mut wb5, mut kn) = (Wb::new(1), Wb::new(3), Wb::new(5), Kn::new(5));
    for i in 0..stream.len() {
        wb1.observe(&[], stream[i]);
        wb3.observe(&stream[i.saturating_sub(2)..i], stream[i]);
        let h = &stream[i.saturating_sub(4)..i];
        wb5.observe(h, stream[i]);
        kn.observe(h, stream[i]);
    }
    let per = |f: &dyn Fn(&[u8], u8) -> f64| -> (f64, f64) {
        let bits: Vec<f32> = (0..d.hidden.len()).map(|i| -(f(&d.hidden[i.saturating_sub(4)..i], d.hidden[i]).max(1e-30)).log2() as f32).collect();
        (bits.iter().map(|b| *b as f64).sum::<f64>() / bits.len().max(1) as f64, masked_mean(&bits, &mask))
    };
    let rows = [
        ("byte-frequency", per(&|_, b| wb1.prob(&[], b))),
        ("witten-bell-3", per(&|h, b| wb3.prob(&h[h.len().saturating_sub(2)..], b))),
        ("witten-bell-5", per(&|h, b| wb5.prob(h, b))),
        ("ppm-c-5", per(&|h, b| wb5.ppm_prob(h, b))),
        ("kneser-ney-5", per(&|h, b| kn.prob(h, b))),
    ];
    let sane = rows.iter().all(|(_, (a, _))| a.is_finite() && *a > 0.0) && rows[4].1 .0 < rows[1].1 .0 && rows[1].1 .0 < rows[0].1 .0;
    let byte_freq = rows[0].1 .0;
    (rows.iter().map(|(n, (h, nv))| json!({"name": n, "heldout_bpb": h, "novel_context_bpb": nv})).collect(), byte_freq, sane)
}

/// V1 in-process checks on the mainline gradient arm: determinism, disabled means no update,
/// no future leak, an exact update log.
fn pipeline_checks(d: &Data, seed: u64, rp: &RunParams) -> Vec<(String, bool)> {
    let n = d.e1.len().min(30_016);
    let bytes = &d.e1[..n];
    let make = || Bptt::with_budget(seed, BpttConfig { lr: rp.bptt_lr, ..Default::default() }, rp.budget);
    let run = |b: &[u8]| {
        let mut m = make();
        m.enable_log("check", 7);
        m.observe_all(b);
        m.close_log();
        m
    };
    let a = run(bytes);
    let b = run(bytes);
    let mut frozen = make();
    frozen.set_learning(false);
    frozen.observe_all(bytes);
    let mut other = bytes.to_vec();
    for x in other[n / 2..].iter_mut() {
        *x = b'~';
    }
    let o = run(&other);
    let half = (n / 2) / 32;
    let ev = a.events();
    let mut at = 0;
    let contiguous = ev.iter().enumerate().all(|(i, e)| {
        let ok = e.seq == i && e.stream_offset == at;
        at += e.bytes;
        ok
    }) && at == n;
    let cps = a.checkpoints();
    vec![
        ("determinism: a rerun reproduces every weight".into(), a.weights_sha256() == b.weights_sha256()),
        ("disabled means no update".into(), frozen.weights_sha256() == make().weights_sha256() && frozen.events().is_empty()),
        ("no future leak: scores before a changed byte are unchanged".into(), ev[..half].iter().map(|e| e.bits_before).collect::<Vec<_>>() == o.events()[..half].iter().map(|e| e.bits_before).collect::<Vec<_>>()),
        ("the gradient arm changes the weights".into(), a.weights_sha256() != make().weights_sha256()),
        ("the update log is exact: one event per update chunk, contiguous offsets covering the stream".into(), contiguous && ev.len() == n.div_ceil(32)),
        ("checkpoint blocks tile the updates and the last hash is the weights' hash".into(), cps.last().map(|c| c.last_seq == ev.len() - 1 && c.checkpoint_sha256 == a.weights_sha256()).unwrap_or(false) && cps.windows(2).all(|w| w[1].first_seq == w[0].last_seq + 1)),
    ]
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let mode = arg("--mode").unwrap_or_else(|| "smoke".into());
    let repo = Path::new(".");
    let spec = GradientSpec::load(repo).expect("spec");
    let manifest_text = std::fs::read_to_string(arg("--manifest").unwrap_or_else(|| "research/corpus/manifest.json".into())).expect("manifest");
    let out = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/representation-gradient".into()));
    std::fs::create_dir_all(&out).expect("out dir");
    let sizes = match mode.as_str() {
        "smoke" => &spec.smoke,
        "main" => &spec.main,
        other => panic!("unknown mode {other}"),
    };
    let t = &spec.thresholds;
    let data = load_data_fresh(&root, &manifest_text, sizes, &spec.fresh_hidden_sources, t.max_shared_window_fraction).expect("data");
    println!("{mode}: E1 {} + E2 {} exposure bytes, probe {}, fresh held-out {} (contamination ok: {})", data.e1.len(), data.e2.len(), data.probe.len(), data.hidden.len(), data.contamination_passed);
    let budget = sizes.core_budget;
    let stream: Vec<u8> = [&data.e1[..], &data.e2[..]].concat();
    let index = kgram_index(&stream, spec.novel_k);
    let novel_fraction = novel_positions(&data.hidden, &index, spec.novel_k).iter().filter(|b| **b).count() as f64 / data.hidden.len().max(1) as f64;
    let seeds: Vec<u64> = if mode == "main" { spec.main_seeds.clone() } else { spec.main_seeds[..1].to_vec() };
    let rp = RunParams {
        delta: Constants::default(),
        eprop: EpropConfig::default(),
        bptt_lr: spec.bptt_lr,
        budget,
        novel_k: spec.novel_k,
        milestones: spec.milestones.clone(),
        window: 100_000.min((stream.len() / 8).max(1000)),
        log_every: Some(spec.log_every),
    };
    let (base_rows, byte_freq, base_sane) = baselines(&data, spec.novel_k);
    type SeedOut = (u64, Vec<ArmRun>, (f64, f64));
    let results: Vec<SeedOut> = std::thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&seed| {
                let (data, index, rp) = (&data, &index, &rp);
                s.spawn(move || {
                    let init = Core::from_seed_budget(seed, rp.budget);
                    let arms: Vec<ArmRun> = std::thread::scope(|s2| {
                        let hs: Vec<_> = [Arm::Frozen, Arm::RandomFeatures, Arm::Bptt]
                            .into_iter()
                            .map(|a| {
                                let init = &init;
                                s2.spawn(move || {
                                    let r = run_arm(a, seed, data, rp, index, init);
                                    println!("seed {seed} arm {}: {} bytes in {:.0}s, final probe {:.3}", a.id(), r.bytes, r.wall_s, r.evals.last().map(|e| e.probe_bpb).unwrap_or(f64::NAN));
                                    r
                                })
                            })
                            .collect();
                        hs.into_iter().map(|h| h.join().expect("arm")).collect()
                    });
                    let a2 = arms.iter().find(|r| r.arm == Arm::Bptt).unwrap();
                    let pert = init.perturbed(&a2.change_norms, seed ^ 0xBEEF);
                    (seed, arms, probe_core(&pert, data, &rp.delta))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("seed")).collect()
    });
    let get = |arms: &[ArmRun], a: Arm| arms.iter().find(|r| r.arm == a).unwrap().clone();
    let mut measures = Vec::new();
    for (_, arms, pert) in &results {
        let (a1, a2) = (get(arms, Arm::RandomFeatures), get(arms, Arm::Bptt));
        let (l1, l2) = (a1.evals.last().unwrap(), a2.evals.last().unwrap());
        let e1 = |r: &ArmRun| r.evals.iter().find(|e| e.bytes == data.e1.len()).map(|e| e.probe_e1_bpb).unwrap_or(f64::NAN);
        let before = e1(&a1) - e1(&a2);
        // kappa is a ratio of probe gains: undefined (NaN, so P4 fails) when the gain after E1 is below r.
        let kappa = if before >= t.r { (l1.probe_e1_bpb - l2.probe_e1_bpb) / before } else { f64::NAN };
        measures.push(GradientMeasures {
            delta_r: l1.probe_bpb - l2.probe_bpb,
            delta_r_perturbed: l1.probe_bpb - pert.0,
            gain_over_byte_frequency: byte_freq - l2.heldout_bpb,
            novel_gain: l1.novel_bpb - l2.novel_bpb,
            kappa,
            finite: a2.finite,
            max_regression: max_regression(&a2.evals.iter().map(|e| e.heldout_bpb).collect::<Vec<_>>()),
            max_abs_weight: a2.max_abs_weight,
        });
    }
    let checks = pipeline_checks(&data, seeds[0], &rp);
    let logs_ok = results.iter().all(|(_, arms, _)| get(arms, Arm::Bptt).log.as_ref().map(|l| l.covers_stream && l.blocks_tile && l.final_checkpoint_matches_weights).unwrap_or(false));
    let v1_ok = data.contamination_passed && checks.iter().all(|c| c.1) && logs_ok;
    let (validity, criteria) = decide_gradient(&measures, t);
    let mut validity_json = vec![
        json!({"id": "V1", "passed": v1_ok, "detail": format!("manifest hashes verified; contamination {}; {}; every main-run update log covers its stream and tiles into checkpoint blocks: {logs_ok}", data.contamination, checks.iter().map(|c| format!("{}: {}", c.0, c.1)).collect::<Vec<_>>().join("; "))}),
        json!({"id": "V4", "passed": base_sane, "detail": "baselines finite and ordered (Kneser-Ney 5 below Witten-Bell 3 below byte frequency) on this stream; the n-gram code is the Phase 8 code"}),
    ];
    validity_json.extend(validity.iter().map(|c| json!({"id": c.id, "passed": c.passed, "detail": c.detail})));
    let (mut outcome, mut deciding): (&str, Vec<&str>) = {
        let failed_v: Vec<&str> = validity_json.iter().filter(|v| v["passed"] == false).map(|v| v["id"].as_str().unwrap()).collect();
        let failed_p: Vec<&str> = criteria.iter().filter(|c| !c.passed).map(|c| c.id).collect();
        if !failed_v.is_empty() { ("inconclusive", failed_v) } else if failed_p.is_empty() { ("pass", criteria.iter().map(|c| c.id).collect()) } else { ("fail_pivot", failed_p) }
    };
    if mode != "main" {
        outcome = "smoke";
        deciding = vec![];
    }
    let mut arm_json: Vec<Value> = Vec::new();
    for arm in [Arm::Frozen, Arm::RandomFeatures, Arm::Bptt] {
        let per_seed: Vec<Value> = results
            .iter()
            .map(|(seed, arms, _)| {
                let (a1, r) = (get(arms, Arm::RandomFeatures), get(arms, arm));
                let (last, a1l) = (r.evals.last().unwrap(), a1.evals.last().unwrap());
                let e1_at = |run: &ArmRun| run.evals.iter().find(|e| e.bytes == data.e1.len()).map(|e| e.probe_e1_bpb);
                let kappa = match (e1_at(&a1), e1_at(&r)) {
                    (Some(a), Some(b)) if a - b >= t.r => (a1l.probe_e1_bpb - last.probe_e1_bpb) / (a - b),
                    _ => f64::NAN,
                };
                let log = r.log.as_ref().map(|l| json!({"updates": l.updates, "checkpoint_blocks": l.checkpoint_blocks, "covers_stream": l.covers_stream, "blocks_tile": l.blocks_tile, "final_checkpoint_matches_weights": l.final_checkpoint_matches_weights, "final_checkpoint_sha256": l.final_checkpoint_sha256, "max_update_norm": l.max_update_norm, "signal": "observed_next_byte", "rule": "truncated_bptt_adam", "input_stream": format!("exposure-seed-{seed}")}));
                json!({"seed": seed, "prequential_bpb": last.prequential_bpb, "heldout_bpb": last.heldout_bpb, "probe_bpb": last.probe_bpb, "delta_r": a1l.probe_bpb - last.probe_bpb, "novel_context_bpb": last.novel_bpb, "ece": last.ece,
                    "retention": {"kappa": kappa, "delta_r_e1_after_e1": e1_at(&a1).zip(e1_at(&r)).map(|(a, b)| a - b), "delta_r_e1_after_e2": a1l.probe_e1_bpb - last.probe_e1_bpb},
                    "curve": r.evals.iter().map(eval_json).collect::<Vec<_>>(), "finite": r.finite, "max_abs_weight": r.max_abs_weight, "update_log": log,
                    "bytes_per_s": r.bytes_per_s, "wall_s": r.wall_s, "peak_rss_kb": r.peak_rss_kb})
            })
            .collect();
        let changed: Vec<&str> = match arm {
            Arm::Frozen => vec![],
            Arm::RandomFeatures => vec!["readout.weight", "readout.bias"],
            _ => TENSORS.iter().copied().chain(["readout.weight", "readout.bias"]).collect(),
        };
        let id = if arm == Arm::Bptt { "gradient_mainline" } else { arm.id() };
        arm_json.push(json!({"id": id, "control": false, "changed_tensors": changed, "per_seed": per_seed}));
    }
    arm_json.push(json!({"id": "gradient_perturbed", "control": false, "changed_tensors": TENSORS, "per_seed": results.iter().map(|(seed, arms, pert)| json!({"seed": seed, "probe_bpb": pert.0, "delta_r": get(arms, Arm::RandomFeatures).evals.last().unwrap().probe_bpb - pert.0})).collect::<Vec<_>>()}));
    let resources: Vec<Value> = results.iter().flat_map(|(seed, arms, _)| arms.iter().map(move |r| json!({"arm": r.arm.id(), "seed": seed, "bytes": r.bytes, "bytes_per_s": r.bytes_per_s, "peak_rss_bytes": r.peak_rss_kb.map(|k| k * 1024).unwrap_or(0), "wall_s": r.wall_s}))).collect();
    let chk = |c: &Check| json!({"id": c.id, "passed": c.passed, "value": c.value, "threshold": c.threshold, "detail": c.detail});
    let summary = json!({
        "report": "representation-gradient", "mode": mode, "outcome": outcome, "deciding_criteria": deciding,
        "decision_record": "research/decisions/gate-a-2026-11-26.md",
        "spec_sha256": GradientSpec::sha256(repo),
        "data": {"manifest_sha256": data.manifest_sha256, "contamination_passed": data.contamination_passed, "contamination": data.contamination,
            "splits": [
                {"role": "exposure_1", "bytes": data.e1.len(), "sha256": corpus::sha256_hex(&data.e1)},
                {"role": "exposure_2", "bytes": data.e2.len(), "sha256": corpus::sha256_hex(&data.e2)},
                {"role": "probe", "bytes": data.probe.len(), "sha256": corpus::sha256_hex(&data.probe)},
                {"role": "hidden_fresh", "bytes": data.hidden.len(), "sha256": corpus::sha256_hex(&data.hidden)}],
            "novel_context": {"k": spec.novel_k, "fraction_of_held_out": novel_fraction}},
        "arms": arm_json, "baselines": base_rows, "validity": validity_json,
        "criteria": criteria.iter().map(chk).collect::<Vec<_>>(),
        "attempts": [{"id": "gate-a-rerun-1", "mode": mode, "rule": "online_gradient", "outcome": outcome}, {"id": "gate-a-main-1", "mode": "main", "rule": "eprop", "outcome": "fail_pivot"}],
        "resources": resources, "core_budget": budget,
        "scope_note": "Gate A rerun with the mainline online-gradient rule on a fresh held-out role no earlier run saw. A smoke run is never a decision."
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).expect("write");
    let mut brief = format!("## Gate A rerun with the mainline online-gradient rule, mode {mode}, outcome **{outcome}**\n\n| arm | final probe BPB (mean) | Delta R (mean) | held-out BPB | novel-context BPB |\n|---|---|---|---|---|\n");
    for a in &arm_json {
        let ps = a["per_seed"].as_array().unwrap();
        let mean = |k: &str| {
            let v: Vec<f64> = ps.iter().filter_map(|p| p[k].as_f64()).collect();
            if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
        };
        brief += &format!("| {} | {:.3} | {:.3} | {:.3} | {:.3} |\n", a["id"].as_str().unwrap(), mean("probe_bpb"), mean("delta_r"), mean("heldout_bpb"), mean("novel_context_bpb"));
    }
    brief += &format!("\nByte frequency {byte_freq:.3} BPB on the same held-out text; Kneser-Ney order 5 {:.3}.", base_rows[4]["heldout_bpb"].as_f64().unwrap());
    for c in criteria.iter().chain(validity.iter()) {
        brief += &format!("\n- {} {}: value {:.4}, threshold {:.4}. {}", c.id, if c.passed { "passed" } else { "FAILED" }, c.value, c.threshold, c.detail);
    }
    brief.push('\n');
    std::fs::write(out.join("brief.md"), &brief).expect("write");
    print!("{brief}");
}
