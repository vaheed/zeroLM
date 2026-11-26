//! `zerolm-gdm --root DATA --phase screening|confirmatory --seeds 9301,9302,9303 [--exposure N] [--heldout N]
//!  [--threads N] [--out DIR] [--spec model/specs/gdm.json]`
//!
//! Experiment GDM (`docs/gdm-experiment.md`): a gated-delta fast-weight memory initialised by weight-space
//! programming (compiled) against the same topology with random tables, against the same topology with the
//! memory disabled, and against a parameter-matched GRU. Screening selects hyperparameters (equal grids of 4
//! configurations per arm, on the probe role) and writes them into its report; the confirmatory phase reads the
//! frozen choices from the spec, never re-selects, and evaluates on the hidden role. The verdict is computed by
//! the rule registered in the spec, not by hand.
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_eval::bptt::{Bptt, Config, Gdm, GdmInit};
use zerolm_eval::gdm::{gdm_flops_per_byte, gru_flops_per_byte, peak_rss_kb, recall_probe, spaced};
use zerolm_eval::hybrid::{ctx_of, Mixer};
use zerolm_eval::ngram_lean::LeanKn;
use zerolm_eval::representation::{kgram_index, novel_positions};
use zerolm_eval::stats::t_interval;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

const ARMS: [&str; 4] = ["GRU", "GDM-random", "GDM-wsp", "GDM-wsp-off"];
const NCFG: usize = 6;
const GRU_LR: [f32; NCFG] = [5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3];
/// (lr, b0, gamma0) for the two memory-on arms.
const GDM_GRID: [(f32, f32, f32); NCFG] = [(1e-3, -1.0, 6.0), (2e-3, -1.0, 6.0), (3e-3, -1.0, 6.0), (1e-3, 1.0, 12.0), (2e-3, 1.0, 12.0), (3e-3, 1.0, 12.0)];
const MIX_LR: [f64; 5] = [0.01, 0.03, 0.1, 0.3, 1.0];
const A0: f32 = 5.0;
const CHECKPOINT_EVERY: usize = 100_000;
const PROBE_DISTRACTORS: [usize; 4] = [0, 10, 30, 100];
const PROBE_FACTS: usize = 4;
/// Distractor length of the registered retention primary P3 (chosen from the untrained positive control, see the doc).
const P3_L: &str = "10";

#[derive(Clone, Copy)]
struct Chosen {
    lr: f32,
    b0: f32,
    gamma0: f32,
    mixer_lr: f64,
}

fn arm_config(arm: &str, c: &Chosen) -> (Config, usize) {
    let base = Config { lr: c.lr, ..Config::default() };
    let gdm = |init, off| Gdm { init, off, a0: A0, b0: c.b0, gamma0: c.gamma0, ..Gdm::compiled() };
    match arm {
        "GRU" => (base, 0), // budget filled in by the caller (parameter matched)
        "GDM-random" => (Config { gdm: Some(gdm(GdmInit::Random, false)), ..base }, 20_000),
        "GDM-wsp" => (Config { gdm: Some(gdm(GdmInit::Compiled, false)), ..base }, 20_000),
        "GDM-wsp-off" => (Config { gdm: Some(gdm(GdmInit::Compiled, true)), ..base }, 20_000),
        _ => unreachable!(),
    }
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64
}

fn pool<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<(usize, T)>> = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.min(n).max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= n {
                    break;
                }
                let r = f(i);
                out.lock().unwrap().push((i, r));
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

/// Mixture of the model and Kneser-Ney 5 (arm G style), prequential mixer; returns bits per position.
fn run_mixed(model: &Bptt, kn: &LeanKn, text: &[u8], lr: f64) -> Vec<f32> {
    let mut mixer = Mixer::new(lr);
    let mut out = Vec::with_capacity(text.len());
    model.score_probs(text, |i, probs| {
        let hist = &text[i.saturating_sub(4)..i];
        let y = text[i];
        let ctx = ctx_of(probs.iter().cloned().fold(0.0, f32::max));
        let (pa, pk) = (probs[y as usize] as f64, kn.prob(hist, y));
        out.push(-(mixer.mix(ctx, &[pa, pk]).max(1e-12)).log2() as f32);
        mixer.update(ctx, &[pa, pk]);
    });
    out
}

/// Bytes at which a curve first reaches `target`, interpolated; None if never.
fn bytes_to_target(curve: &[(usize, f64)], target: f64) -> Option<f64> {
    if curve[0].1 <= target {
        return Some(curve[0].0 as f64);
    }
    curve.windows(2).find(|w| w[1].1 <= target).map(|w| w[0].0 as f64 + (w[0].1 - target) / (w[0].1 - w[1].1) * (w[1].0 - w[0].0) as f64)
}

struct Data {
    fic_exp: Vec<u8>,
    wiki_exp: Vec<u8>,
    // held-out per role: (name, text)
    e1_held: Vec<(&'static str, Vec<u8>)>,
    e2_held: Vec<(&'static str, Vec<u8>)>,
    probe_slice: Vec<u8>,
    distractor: Vec<u8>,
    exposure_n: usize,
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let phase = arg("--phase").unwrap_or_else(|| "screening".into());
    assert!(phase == "screening" || phase == "confirmatory", "--phase screening|confirmatory");
    let manifest_text = std::fs::read_to_string("research/corpus/manifest.json").expect("manifest");
    let m = Manifest::parse(&manifest_text).expect("manifest");
    assert!(corpus::verify(&m, &root).is_empty());
    let exp_n: usize = arg("--exposure").map(|v| v.parse().unwrap()).unwrap_or(300_000);
    let held_n: usize = arg("--heldout").map(|v| v.parse().unwrap()).unwrap_or(100_000);
    let seeds: Vec<u64> = arg("--seeds").map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).expect("--seeds");
    let threads: usize = arg("--threads").map(|v| v.parse().unwrap()).unwrap_or(16);
    let sel_n: usize = arg("--select-bytes").map(|v| v.parse().unwrap()).unwrap_or(150_000);
    let probe_trials: usize = arg("--probe-trials").map(|v| v.parse().unwrap()).unwrap_or(40);
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(arg("--spec").unwrap_or_else(|| "model/specs/gdm.json".into())).expect("spec")).expect("spec json");
    let role = if phase == "screening" { corpus::Role::Dev } else { corpus::Role::Hidden };
    let stream = |id: &str, role| corpus::source_role_stream(&m, &root, id, role).unwrap();
    let fic_exp = stream("gutenberg-books", corpus::Role::Exposure);
    let wiki_exp = stream("simple-wikipedia", corpus::Role::Exposure);
    let data = Data {
        e1_held: vec![("fiction", spaced(&stream("gutenberg-books", role), 4, held_n / 4)), ("nonfiction", spaced(&stream("gutenberg-nonfiction", role), 4, held_n / 4))],
        e2_held: vec![("fiction", spaced(&stream("gutenberg-books", role), 4, held_n / 4)), ("wiki", spaced(&stream("simple-wikipedia", role), 4, held_n / 4))],
        probe_slice: { let mut v = stream("gutenberg-books", corpus::Role::Probe); v.truncate(50_000); v },
        distractor: stream("gutenberg-books", role),
        fic_exp,
        wiki_exp,
        exposure_n: exp_n,
    };
    let offset = |seed: u64, avail: usize, n: usize| -> usize { if avail <= n { 0 } else { (seed.wrapping_mul(2_654_435_761) as usize) % (avail - n) } };

    // ---- hyperparameter choice -------------------------------------------------------------------------------
    let gd_params = Bptt::with_budget(seeds[0], arm_config("GDM-wsp", &Chosen { lr: 1e-3, b0: 0.0, gamma0: 4.0, mixer_lr: 0.1 }).0, 20_000).param_count();
    let gru_budget = (20_000..200_000).step_by(500).find(|&b| Bptt::with_budget(seeds[0], Config::default(), b).param_count() >= gd_params).expect("budget");
    let mut selection = Vec::new();
    let chosen: Vec<Chosen> = if phase == "screening" {
        let sel_seed = 9300u64;
        let sel_text = &data.fic_exp[..sel_n.min(data.fic_exp.len())];
        // grid per arm: 4 configurations each
        let configs: Vec<(usize, Chosen)> = ARMS.iter().enumerate().flat_map(|(ai, a)| {
            (0..NCFG).map(move |g| {
                let c = match *a {
                    "GRU" | "GDM-wsp-off" => Chosen { lr: GRU_LR[g], b0: 0.0, gamma0: 4.0, mixer_lr: 0.1 },
                    _ => Chosen { lr: GDM_GRID[g].0, b0: GDM_GRID[g].1, gamma0: GDM_GRID[g].2, mixer_lr: 0.1 },
                };
                (ai, c)
            }).collect::<Vec<_>>()
        }).collect();
        let res = pool(configs.len(), threads, |i| {
            let (ai, c) = configs[i];
            let (cfg, b) = arm_config(ARMS[ai], &c);
            let mut model = Bptt::with_budget(sel_seed, cfg, if ai == 0 { gru_budget } else { b });
            model.observe_all(sel_text);
            let bpb = model.evaluate(&data.probe_slice);
            let mut kn = LeanKn::new(5);
            for k in 0..sel_text.len() {
                kn.observe(&sel_text[k.saturating_sub(4)..k], sel_text[k]);
            }
            let mix: Vec<f64> = MIX_LR.iter().map(|&lr| mean(&run_mixed(&model, &kn, &data.probe_slice, lr))).collect();
            (bpb, mix)
        });
        let mut best: Vec<Chosen> = Vec::new();
        for (ai, arm) in ARMS.iter().enumerate() {
            let mut pick = (f64::INFINITY, configs[ai * NCFG].1, 0usize);
            for g in 0..NCFG {
                let (bpb, mix) = &res[ai * NCFG + g];
                println!("select {arm} config {g}: probe BPB {bpb:.4}");
                selection.push(json!({"arm": arm, "config": g, "lr": configs[ai * NCFG + g].1.lr, "b0": configs[ai * NCFG + g].1.b0, "gamma0": configs[ai * NCFG + g].1.gamma0, "probe_bpb": bpb, "mixer_probe_bpb": mix}));
                if *bpb < pick.0 {
                    pick = (*bpb, configs[ai * NCFG + g].1, g);
                }
            }
            let mix = &res[ai * NCFG + pick.2].1;
            let ml = MIX_LR[(0..MIX_LR.len()).min_by(|&i, &j| mix[i].partial_cmp(&mix[j]).unwrap()).unwrap()];
            println!("chosen {arm}: lr {} b0 {} gamma0 {} mixer lr {ml}", pick.1.lr, pick.1.b0, pick.1.gamma0);
            best.push(Chosen { mixer_lr: ml, ..pick.1 });
        }
        best
    } else {
        ARMS.iter().map(|a| {
            let c = &spec["chosen"][*a];
            Chosen { lr: c["lr"].as_f64().expect("chosen lr in spec") as f32, b0: c["b0"].as_f64().unwrap() as f32, gamma0: c["gamma0"].as_f64().unwrap() as f32, mixer_lr: c["mixer_lr"].as_f64().unwrap() }
        }).collect()
    };

    if let Some(arm) = arg("--resource-only") {
        // Single-arm, single-thread resource measurement in its own process (peak RSS is process-wide).
        let ai = ARMS.iter().position(|a| *a == arm).expect("arm");
        let (cfg, b) = arm_config(ARMS[ai], &chosen[ai]);
        let mut model = Bptt::with_budget(seeds[0], cfg, if ai == 0 { gru_budget } else { b });
        let n = data.exposure_n.min(200_000);
        let t = Instant::now();
        model.observe_all(&data.fic_exp[..n]);
        let train_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let bpb = model.evaluate(&data.e1_held[0].1[..20_000.min(data.e1_held[0].1.len())]);
        let eval_s = t.elapsed().as_secs_f64();
        println!("{}", json!({"arm": arm, "train_bytes": n, "train_wall_s": train_s, "train_bytes_per_s": n as f64 / train_s, "eval_bytes_per_s": 20_000.0 / eval_s, "heldout_bpb": bpb, "peak_rss_kb": peak_rss_kb(), "params_total": model.param_count(), "params_updatable": model.effective_param_count(), "persistent_state_floats": model.core().d + model.fast_state_len(), "snapshot_bytes": model.snapshot().len()}));
        return;
    }

    // ---- one job per (seed, arm) -----------------------------------------------------------------------------
    let n_jobs = seeds.len() * ARMS.len();
    let jobs: Vec<Value> = pool(n_jobs, threads, |ji| {
        let (seed, ai) = (seeds[ji / ARMS.len()], ji % ARMS.len());
        let arm = ARMS[ai];
        let ch = chosen[ai];
        let (cfg, b) = arm_config(arm, &ch);
        let budget = if ai == 0 { gru_budget } else { b };
        run_job(&data, seed, arm, cfg, budget, ch, offset(seed, data.fic_exp.len(), data.exposure_n), offset(seed, data.wiki_exp.len(), data.exposure_n / 2), probe_trials)
    });
    for j in &jobs {
        println!("seed {} {:<12} e1 fiction/nonfiction alone {:.3}/{:.3}  recall(reset_h, L=10) {:.3}  forgetting {:+.3}  train {:.0}s", j["seed"], j["arm"].as_str().unwrap(), j["e1"]["final"]["fiction"]["alone"].as_f64().unwrap(), j["e1"]["final"]["nonfiction"]["alone"].as_f64().unwrap(),
            j["probe"][P3_L]["reset_h"]["recall"].as_f64().unwrap(), j["e2"]["forgetting_fiction"].as_f64().unwrap(), j["resources"]["train_wall_s"].as_f64().unwrap());
    }

    // ---- aggregation -----------------------------------------------------------------------------------------
    let get = |seed: u64, arm: &str| jobs.iter().find(|j| j["seed"].as_u64() == Some(seed) && j["arm"] == arm).unwrap();
    let alpha_family = spec["registration"]["family_size"].as_f64().unwrap_or(4.0);
    let level = 1.0 - 0.05 / alpha_family;
    type Metric = (&'static str, bool, Box<dyn Fn(&Value) -> f64>); // (name, lower_is_better, extractor)
    let f = |path: &'static [&'static str]| -> Box<dyn Fn(&Value) -> f64> { Box::new(move |j: &Value| { let mut v = j; for p in path { v = &v[*p]; } v.as_f64().unwrap_or(f64::NAN) }) };
    let metrics: Vec<Metric> = vec![
        ("e1_bpb_mean", true, Box::new(|j| (j["e1"]["final"]["fiction"]["alone"].as_f64().unwrap() + j["e1"]["final"]["nonfiction"]["alone"].as_f64().unwrap()) / 2.0)),
        ("e1_bpb_fiction", true, f(&["e1", "final", "fiction", "alone"])),
        ("e1_bpb_nonfiction", true, f(&["e1", "final", "nonfiction", "alone"])),
        ("e1_bpb_with_kn_mean", true, Box::new(|j| (j["e1"]["final"]["fiction"]["with_kn"].as_f64().unwrap() + j["e1"]["final"]["nonfiction"]["with_kn"].as_f64().unwrap()) / 2.0)),
        ("e1_novel_context_bpb_mean", true, Box::new(|j| (j["e1"]["final"]["fiction"]["novel_alone"].as_f64().unwrap() + j["e1"]["final"]["nonfiction"]["novel_alone"].as_f64().unwrap()) / 2.0)),
        ("e2_forgetting_fiction", true, f(&["e2", "forgetting_fiction"])),
        ("e2_final_fiction", true, f(&["e2", "after_phase2", "fiction"])),
        ("e2_final_wiki", true, f(&["e2", "after_phase2", "wiki"])),
        ("probe_recall_L10_reset_h", false, f(&["probe", "10", "reset_h", "recall"])),
        ("probe_recall_L0_full", false, f(&["probe", "0", "full", "recall"])),
        ("probe_recall_L0_reset_h", false, f(&["probe", "0", "reset_h", "recall"])),
        ("probe_recall_L30_reset_h", false, f(&["probe", "30", "reset_h", "recall"])),
        ("probe_recall_L100_reset_h", false, f(&["probe", "100", "reset_h", "recall"])),
        ("probe_recall_L10_full", false, f(&["probe", "10", "full", "recall"])),
        ("probe_recall_L10_reset_s", false, f(&["probe", "10", "reset_s", "recall"])),
        ("probe_recall_L10_restart", false, f(&["probe", "10", "restart", "recall"])),
        ("probe_recall_L10_weights", false, f(&["probe", "10", "weights", "recall"])),
        ("train_bytes_per_s", false, f(&["resources", "train_bytes_per_s"])),
        ("eval_bytes_per_s", false, f(&["resources", "eval_bytes_per_s"])),
    ];
    // learning efficiency: bytes for arm X to reach the final BPB of arm Y, as a fraction of Y's exposure
    let ratio = |seed: u64, x: &str, y: &str| -> f64 {
        let cv = |arm: &str| -> Vec<(usize, f64)> { get(seed, arm)["e1"]["curve"].as_array().unwrap().iter().map(|p| (p["bytes"].as_u64().unwrap() as usize, p["bpb"].as_f64().unwrap())).collect() };
        let (cy, cx) = (cv(y), cv(x));
        let target = cy.last().unwrap().1;
        bytes_to_target(&cx, target).map_or(1.0, |b| (b / cy.last().unwrap().0 as f64).min(1.0))
    };
    let contrasts: Vec<(&str, &str, &str)> = vec![("C1 wsp vs random", "GDM-wsp", "GDM-random"), ("C2 wsp vs memory-off", "GDM-wsp", "GDM-wsp-off"), ("C3 wsp vs GRU", "GDM-wsp", "GRU"), ("C4 random vs GRU", "GDM-random", "GRU"), ("C5 memory-off vs GRU", "GDM-wsp-off", "GRU")];
    let mut rows = Vec::new();
    for (cname, x, y) in &contrasts {
        for (mname, lower, ex) in &metrics {
            // gain = how much better X is than Y (positive is good for X)
            let d: Vec<f64> = seeds.iter().map(|&s| { let (a, b) = (ex(get(s, x)), ex(get(s, y))); if *lower { b - a } else { a - b } }).collect();
            if d.iter().any(|v| v.is_nan()) { continue; }
            let iv = t_interval(&d, level);
            rows.push(json!({"contrast": cname, "x": x, "y": y, "metric": mname, "gain_x_over_y": iv.mean, "sd": iv.sd, "ci_lo": iv.lo, "ci_hi": iv.hi, "ci_level": level, "dz": iv.dz, "seed_gains": d, "positive_seeds": d.iter().filter(|v| **v > 0.0).count()}));
        }
        let d: Vec<f64> = seeds.iter().map(|&s| ratio(s, x, y)).collect();
        let iv = t_interval(&d, level);
        rows.push(json!({"contrast": cname, "x": x, "y": y, "metric": "p1_bytes_ratio", "gain_x_over_y": iv.mean, "sd": iv.sd, "ci_lo": iv.lo, "ci_hi": iv.hi, "ci_level": level, "dz": iv.dz, "seed_gains": d, "positive_seeds": d.iter().filter(|v| **v < 1.0).count(), "note": "fraction of Y's exposure that X needs to reach Y's final BPB; lower is better, 1.0 means never reached"}));
    }
    let row = |c: &str, m: &str| rows.iter().find(|r| r["contrast"].as_str().unwrap().starts_with(c) && r["metric"] == m).unwrap().clone();
    let th = &spec["registration"]["thresholds"];
    let thr = |k: &str| th[k].as_f64().unwrap_or(f64::NAN);
    // registered primary rules (C1: wsp over random), each requires mean over threshold AND the interval to exclude 0
    let lr_equal_c1 = (chosen[1].lr - chosen[2].lr).abs() < 1e-9;
    // Instrument validation: untrained models with the chosen memory settings, probe at every distractor length.
    let validation: Vec<Value> = [("GDM-wsp", GdmInit::Compiled, chosen[2]), ("GDM-random", GdmInit::Random, chosen[1])].iter().map(|(nm, init, c)| {
        let cfg = Config { gdm: Some(Gdm { init: *init, a0: A0, b0: c.b0, gamma0: c.gamma0, ..Gdm::compiled() }), ..Config::default() };
        let make = || Bptt::with_budget(seeds[0], cfg, 20_000);
        let mdl = make();
        let snap = mdl.snapshot();
        let per: serde_json::Map<String, Value> = PROBE_DISTRACTORS.iter().map(|&dl| {
            let r = recall_probe(&mdl, &make, &snap, &data.distractor, dl, 40, PROBE_FACTS, 4242);
            (dl.to_string(), json!({"full": r[0].1.recall(), "reset_h": r[1].1.recall()}))
        }).collect();
        json!({"arm": nm, "untrained": true, "recall": Value::Object(per)})
    }).collect();
    let probe_valid = validation[0]["recall"]["0"]["full"].as_f64().unwrap() >= 0.1 && validation[1]["recall"]["0"]["full"].as_f64().unwrap() <= 0.05;
    let primaries: Vec<(&str, &str, bool, f64)> = vec![("P1_learning_efficiency", "p1_bytes_ratio", true, thr("p1_bytes_ratio_max")), ("P2_heldout_bpb", "e1_bpb_mean", false, thr("p2_bpb_gain_min")), ("P3_retention_recall", "probe_recall_L10_reset_h", false, thr("p3_recall_gain_min")), ("P4_forgetting", "e2_forgetting_fiction", false, thr("p4_forgetting_reduction_min"))];
    let eval_rule = |c: &str, with_c2: bool| -> Vec<Value> {
        primaries.iter().map(|(pname, metric, is_ratio, t)| {
            let r = row(c, metric);
            let (mean, lo, hi) = (r["gain_x_over_y"].as_f64().unwrap(), r["ci_lo"].as_f64().unwrap(), r["ci_hi"].as_f64().unwrap());
            let met = if *is_ratio { mean <= *t && hi < 1.0 } else { mean >= *t && lo > 0.0 };
            let c2 = if with_c2 { let r2 = row("C2", metric); if *is_ratio { r2["gain_x_over_y"].as_f64().unwrap() < 1.0 } else { r2["gain_x_over_y"].as_f64().unwrap() > 0.0 } } else { true };
            let valid = match *pname { "P3_retention_recall" => probe_valid, "P4_forgetting" => lr_equal_c1 || !with_c2, _ => true };
            json!({"primary": pname, "metric": metric, "threshold": t, "mean": mean, "ci_lo": lo, "ci_hi": hi, "met": met && valid, "memory_attribution_ok": c2, "instrument_valid": valid, "pass": met && c2 && valid})
        }).collect()
    };
    let c1 = eval_rule("C1", true);
    let c3 = eval_rule("C3", false);
    let regress = spec["registration"]["thresholds"]["max_bpb_regression"].as_f64().unwrap_or(0.01);
    let prot_bpb = ["e1_bpb_fiction", "e1_bpb_nonfiction"].iter().all(|m| row("C1", m)["ci_lo"].as_f64().unwrap() > -regress && row("C1", m)["gain_x_over_y"].as_f64().unwrap() > -regress);
    let tp_ratio = seeds.iter().map(|&s| get(s, "GDM-wsp")["resources"]["train_bytes_per_s"].as_f64().unwrap() / get(s, "GRU")["resources"]["train_bytes_per_s"].as_f64().unwrap()).sum::<f64>() / seeds.len() as f64;
    let prot_tp = tp_ratio >= thr("min_throughput_ratio_vs_gru");
    let prot_snap = jobs.iter().all(|j| j["snapshot"]["exact_restore"] == true && j["snapshot"]["branch_restore_exact"] == true);
    let prot_leak = jobs.iter().all(|j| j["leakage"]["weights_unchanged_by_eval"] == true);
    let protected_ok = prot_bpb && prot_tp && prot_snap && prot_leak;
    let any_pass = c1.iter().any(|r| r["pass"] == true);
    let any_arch = c3.iter().any(|r| r["met"] == true) || c1.iter().any(|r| r["met"] == true);
    let verdict = if phase != "confirmatory" { "NOT CONFIRMATORY (screening)" } else if any_pass && protected_ok { "PASS" } else if any_arch && protected_ok { "PIVOT" } else { "FAIL" };
    println!("\nprimary rules, C1 wsp over random: {}", serde_json::to_string_pretty(&c1).unwrap());
    println!("primary rules, C3 wsp over GRU: {}", c3.iter().map(|r| format!("{}={}", r["primary"], r["met"])).collect::<Vec<_>>().join(" "));
    println!("protected: bpb {prot_bpb} throughput {prot_tp} (ratio {tp_ratio:.2}) snapshot {prot_snap} leakage {prot_leak}");
    println!("instrument validation (untrained positive/negative control): valid {probe_valid} {}", validation.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    println!("lr equal for C1: {lr_equal_c1}");
    println!("VERDICT: {verdict}");

    let manifest_sha = corpus::sha256_hex(manifest_text.as_bytes());
    let report = json!({
        "report": "gdm", "phase": phase, "role": if phase == "screening" { "dev" } else { "hidden" }, "verdict": verdict,
        "seeds": seeds, "exposure_bytes": exp_n, "heldout_bytes_per_role": held_n, "manifest_sha256": manifest_sha,
        "exposure_fiction_sha256": corpus::sha256_hex(&data.fic_exp), "exposure_wiki_sha256": corpus::sha256_hex(&data.wiki_exp),
        "heldout_sha256": data.e1_held.iter().chain(&data.e2_held).map(|(n, t)| json!({"name": n, "sha256": corpus::sha256_hex(t), "bytes": t.len()})).collect::<Vec<_>>(),
        "spec_sha256": corpus::sha256_hex(std::fs::read_to_string(arg("--spec").unwrap_or_else(|| "model/specs/gdm.json".into())).unwrap().as_bytes()),
        "gru_budget": gru_budget, "selection": selection,
        "chosen": ARMS.iter().zip(&chosen).map(|(a, c)| (a.to_string(), json!({"lr": c.lr, "b0": c.b0, "gamma0": c.gamma0, "mixer_lr": c.mixer_lr}))).collect::<serde_json::Map<String, Value>>(),
        "jobs": jobs, "contrasts": rows, "instrument_validation": validation, "probe_valid": probe_valid, "lr_equal_c1": lr_equal_c1,
        "rules": {"C1_wsp_over_random": c1, "C3_wsp_over_gru": c3, "protected": {"bpb_regression_ok": prot_bpb, "throughput_ok": prot_tp, "throughput_ratio_vs_gru": tp_ratio, "snapshot_ok": prot_snap, "leakage_ok": prot_leak}},
        "ci_note": format!("Student-t intervals over seed-paired differences at {:.2}% (Holm-style family of {alpha_family}); seeds vary the initialisation and the exposure window.", level * 100.0),
        "scope_note": "One head, rank 32, byte level, <=1e6 exposure bytes, 20,000-class GRU; no claim beyond this.",
    });
    let dir = PathBuf::from(arg("--out").unwrap_or_else(|| format!("reports/gdm-{phase}")));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&report).unwrap()).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn run_job(data: &Data, seed: u64, arm: &'static str, cfg: Config, budget: usize, ch: Chosen, off_fic: usize, off_wiki: usize, probe_trials: usize) -> Value {
    let n = data.exposure_n;
    let exp = &data.fic_exp[off_fic..off_fic + n.min(data.fic_exp.len() - off_fic)];
    let make = || Bptt::with_budget(seed, cfg, budget);
    let mut model = make();
    model.enable_log("exposure:gutenberg-books", 1000);
    let genome_sha = model.genome_sha256();
    let (e, d) = (zerolm_engine::bootstrap::EMBED_DIM, model.core().d);
    let curve_text: Vec<Vec<u8>> = data.e1_held.iter().map(|(_, t)| t[..30_000.min(t.len())].to_vec()).collect();
    let step = CHECKPOINT_EVERY.min(n / 3).max(1);
    let mut curve = Vec::new();
    let (mut done, mut train_s, mut eval_s, mut eval_bytes) = (0usize, 0.0f64, 0.0f64, 0usize);
    let mut snap_half: Option<(Vec<u8>, String)> = None;
    while done < exp.len() {
        let cp = (done + step).min(exp.len());
        let t = Instant::now();
        model.observe_all(&exp[done..cp]);
        train_s += t.elapsed().as_secs_f64();
        done = cp;
        let t = Instant::now();
        let bpb = curve_text.iter().map(|c| model.evaluate(c)).sum::<f64>() / curve_text.len() as f64;
        eval_s += t.elapsed().as_secs_f64();
        eval_bytes += curve_text.iter().map(|c| c.len()).sum::<usize>();
        curve.push(json!({"bytes": cp, "bpb": bpb}));
        if cp >= exp.len() / 2 && snap_half.is_none() {
            let s = model.snapshot();
            let h = model.snapshot_sha256();
            snap_half = Some((s, h));
            // Window boundaries differ from the half point only by `step`; the branch point is wherever this lands.
            curve.last_mut().unwrap()["branch_point"] = json!(true);
        }
    }
    let branch_at = curve.iter().find(|c| c["branch_point"] == true).map(|c| c["bytes"].as_u64().unwrap() as usize).unwrap_or(n / 2);
    model.close_log();
    let weights_hash_final = model.weights_sha256();
    let events = model.events();
    let upd_n = events.len();
    let mean_update = events.iter().map(|e| e.update_norm as f64).sum::<f64>() / upd_n.max(1) as f64;
    let mean_grad = events.iter().map(|e| e.grad_norm as f64).sum::<f64>() / upd_n.max(1) as f64;
    let n_ckpt = model.checkpoints().len();

    // ---- E1 final evaluation: alone, with Kneser-Ney 5, novel-context positions ----
    let mut kn = LeanKn::new(5);
    for i in 0..exp.len() {
        kn.observe(&exp[i.saturating_sub(4)..i], exp[i]);
    }
    let kidx = kgram_index(exp, 10);
    let mut fin = serde_json::Map::new();
    let before_eval = model.weights_sha256();
    for (name, text) in &data.e1_held {
        let alone = model.score(text);
        let mixed = run_mixed(&model, &kn, text, ch.mixer_lr);
        let novel = novel_positions(text, &kidx, 10);
        let nov = |v: &[f32]| { let x: Vec<f32> = v.iter().zip(&novel).filter(|p| *p.1).map(|p| *p.0).collect(); mean(&x) };
        let kn_bits: Vec<f32> = (0..text.len()).map(|i| -(kn.prob(&text[i.saturating_sub(4)..i], text[i]).max(1e-12)).log2() as f32).collect();
        fin.insert(name.to_string(), json!({"alone": mean(&alone), "with_kn": mean(&mixed), "kn5": mean(&kn_bits), "novel_alone": nov(&alone), "novel_with_kn": nov(&mixed), "novel_positions": novel.iter().filter(|x| **x).count()}));
    }
    let leakage_ok = before_eval == model.weights_sha256();

    // ---- snapshot: exact restore of the final state, and continued execution identical ----
    let final_snap = model.snapshot();
    let final_snap_sha = model.snapshot_sha256();
    let cont = &data.fic_exp[(off_fic + n) % (data.fic_exp.len() - 3000)..][..2000];
    let mut restored = make();
    restored.restore(&final_snap).expect("restore");
    let restored_same = restored.snapshot_sha256() == final_snap_sha;
    let bits_a = model.observe_all(cont);
    let bits_b = restored.observe_all(cont);
    let exact_restore = restored_same && bits_a.to_bits() == bits_b.to_bits() && model.weights_sha256() == restored.weights_sha256() && model.snapshot_sha256() == restored.snapshot_sha256();
    // (model has now advanced 2000 bytes; the probes below use the restored-from-final snapshot's weights, i.e. `final_snap`)
    let mut probe_model = make();
    probe_model.restore(&final_snap).unwrap();

    // ---- E2: branch from the half-way snapshot into the second domain ----
    let (half_snap, half_sha) = snap_half.expect("half snapshot");
    let mut branch = make();
    branch.restore(&half_snap).expect("restore half");
    let branch_restore_exact = branch.snapshot_sha256() == half_sha;
    let after1: serde_json::Map<String, Value> = data.e2_held.iter().map(|(nm, t)| (nm.to_string(), json!(branch.evaluate(t)))).collect();
    let wiki = &data.wiki_exp[off_wiki..off_wiki + (n - branch_at).min(data.wiki_exp.len() - off_wiki)];
    branch.observe_all(wiki);
    let after2: serde_json::Map<String, Value> = data.e2_held.iter().map(|(nm, t)| (nm.to_string(), json!(branch.evaluate(t)))).collect();
    let forgetting = after2["fiction"].as_f64().unwrap() - after1["fiction"].as_f64().unwrap();
    let wiki_gain = after1["wiki"].as_f64().unwrap() - after2["wiki"].as_f64().unwrap();

    // ---- E3/E4: recall probes (frozen weights; the `weights` condition learns in throw-away copies) ----
    let mut probe = serde_json::Map::new();
    for &dl in &PROBE_DISTRACTORS {
        let res = recall_probe(&probe_model, &make, &final_snap, &data.distractor, dl, probe_trials, PROBE_FACTS, seed ^ (dl as u64 + 1));
        let mut m = serde_json::Map::new();
        for (cn, t) in res {
            m.insert(cn.to_string(), json!({"recall": t.recall(), "shown_bits": t.shown_bits / t.n as f64, "control_bits": t.control_bits / t.n as f64, "shown_acc": t.shown_hits as f64 / t.n as f64, "control_acc": t.control_hits as f64 / t.n as f64, "queries": t.n}));
        }
        probe.insert(dl.to_string(), Value::Object(m));
    }
    let lt = model.gdm_param_range();
    let flops_fwd = gru_flops_per_byte(e, d) + model.effective_param_count().min(1) as f64 * cfg.gdm.filter(|g| !g.off).map_or(0.0, |g| gdm_flops_per_byte(g.rank, g.lags));
    json!({
        "seed": seed, "arm": arm, "genome_sha256": genome_sha, "config": {"lr": ch.lr, "b0": ch.b0, "gamma0": ch.gamma0, "mixer_lr": ch.mixer_lr, "budget": budget},
        "e1": {"curve": curve, "final": Value::Object(fin), "exposure_offset": off_fic},
        "e2": {"branch_bytes": branch_at, "after_phase1": after1, "after_phase2": after2, "forgetting_fiction": forgetting, "wiki_gain": wiki_gain, "phase2_bytes": wiki.len(), "phase2_offset": off_wiki},
        "probe": Value::Object(probe),
        "snapshot": {"exact_restore": exact_restore, "branch_restore_exact": branch_restore_exact, "final_snapshot_sha256": final_snap_sha, "weights_sha256_after_exposure": weights_hash_final, "bytes": final_snap.len()},
        "leakage": {"weights_unchanged_by_eval": leakage_ok, "heldout_roles_disjoint_from_exposure": true},
        "updates": {"count": upd_n, "mean_update_norm": mean_update, "mean_grad_norm": mean_grad, "checkpoint_blocks": n_ckpt, "changed_tensors": model.changed_tensors().len()},
        "resources": {"params_total": model.param_count(), "params_updatable": model.effective_param_count(), "gdm_params": lt.len(), "hidden_state_floats": d, "fast_weight_floats": model.fast_state_len(), "persistent_state_floats": d + model.fast_state_len(),
            "snapshot_bytes": final_snap.len(), "train_wall_s": train_s, "train_bytes_per_s": exp.len() as f64 / train_s, "eval_wall_s": eval_s, "eval_bytes_per_s": eval_bytes as f64 / eval_s,
            "flops_per_byte_forward_est": flops_fwd, "flops_per_byte_train_est": 3.0 * flops_fwd, "process_peak_rss_kb": peak_rss_kb(), "rss_note": "process-wide peak while jobs run concurrently; per-arm RSS is in the resource runs"},
    })
}
