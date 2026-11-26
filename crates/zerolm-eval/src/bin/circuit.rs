//! `zerolm-circuit --root DATA [--exposure 1000000] [--heldout 100000] [--seeds 9201,9202,9203] [--out DIR]`
//!
//! Experiment N2 (`docs/experimental-design.md`): the trainable compiled circuit inside the online GRU.
//! Arms (dev role only, equal exposure): A plain GRU; A+ GRU with parameters matched to F; D compiled circuit,
//! frozen; F compiled circuit as initial value, then trained; E same circuit architecture, random tables
//! (this is also the scrambled-weights control X1: same shapes, same row norms, no alignment).
//! Every arm is also reported mixed with Kneser-Ney 5 (X + KN) and with the match layer on top (X + KN + match),
//! the same hybrids as arms G and H. Init scalars (beta, s0) and the circuit learning-rate multiplier are chosen
//! per arm (F and E) from one shared grid on a probe-role slice; the hidden role is never read.
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_eval::bptt::{Bptt, Circuit, CircuitInit, Config};
use zerolm_eval::headroom::{paired_ci, MatchModel};
use zerolm_eval::hybrid::{ctx_of, Mixer};
use zerolm_eval::ngram_lean::LeanKn;
use zerolm_eval::representation::{kgram_index, novel_positions};

type Curve = Vec<(usize, f64)>;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

const SCALARS: [(f32, f32); 3] = [(8.0, 20.0), (4.0, 4.0), (2.0, 0.0)];
const LR_MULT: [f32; 2] = [1.0, 10.0];
const MIX_LR: [f64; 4] = [0.01, 0.03, 0.1, 0.3];
const CURVE: [usize; 5] = [100_000, 200_000, 400_000, 700_000, 1_000_000];

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64
}

/// Bits of (model alone, model + KN mixture, mixture + match layer) per position.
fn run_hybrid(model: &Bptt, kn: &LeanKn, text: &[u8], lr: f64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut alone, mut mixed, mut full) = (vec![], vec![], vec![]);
    let mut mixer = Mixer::new(lr);
    let mut mm = MatchModel::new();
    let bits = |p: f64| -p.max(1e-12).log2() as f32;
    let q_of = |pp: f64| if pp < 0.1 { 0 } else if pp < 0.4 { 1 } else if pp < 0.8 { 2 } else { 3 };
    model.score_probs(text, |i, probs| {
        let hist = &text[i.saturating_sub(4)..i];
        let y = text[i];
        let ctx = ctx_of(probs.iter().cloned().fold(0.0, f32::max));
        let (pa, pk) = (probs[y as usize] as f64, kn.prob(hist, y));
        let pg = mixer.mix(ctx, &[pa, pk]);
        alone.push(bits(pa));
        mixed.push(bits(pg));
        let mut q = 0;
        let ph = match mm.predict() {
            Some((pred, len)) => {
                q = q_of(mixer.mix(ctx, &[probs[pred as usize] as f64, kn.prob(hist, pred)]));
                let c = mm.confidence(len, q);
                (1.0 - c) * pg + if pred == y { c } else { 0.0 }
            }
            None => pg,
        };
        full.push(bits(ph));
        mixer.update(ctx, &[pa, pk]);
        mm.observe(y, q);
    });
    (alone, mixed, full)
}

fn circuit(init: CircuitInit, (beta, s0): (f32, f32), lr_mult: f32, frozen: bool) -> Circuit {
    Circuit { beta, s0, init, lr_mult, frozen, ..Circuit::compiled() }
}

/// Bytes of exposure at which a curve (checkpoint, bpb) first reaches `target`, interpolated; None if never.
fn bytes_to_target(curve: &[(usize, f64)], target: f64) -> Option<f64> {
    for w in curve.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if y1 <= target {
            if y0 <= target { return Some(x0 as f64); }
            return Some(x0 as f64 + (y0 - target) / (y0 - y1) * (x1 - x0) as f64);
        }
    }
    curve.first().filter(|c| c.1 <= target).map(|c| c.0 as f64)
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let manifest_text = std::fs::read_to_string("research/corpus/manifest.json").expect("manifest");
    let m = Manifest::parse(&manifest_text).expect("manifest");
    assert!(corpus::verify(&m, &root).is_empty());
    let exp_n: usize = arg("--exposure").map(|v| v.parse().unwrap()).unwrap_or(1_000_000);
    let held_n: usize = arg("--heldout").map(|v| v.parse().unwrap()).unwrap_or(100_000);
    let seeds: Vec<u64> = arg("--seeds").map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or_else(|| vec![9201, 9202, 9203]);
    let sel_n: usize = arg("--select-bytes").map(|v| v.parse().unwrap()).unwrap_or(150_000);
    let mut exposure = Vec::new();
    for id in &m.exposure_order {
        exposure.extend(corpus::source_role_stream(&m, &root, id, corpus::Role::Exposure).unwrap());
        if exposure.len() >= exp_n { break; }
    }
    exposure.truncate(exp_n);
    let slice = |id: &str, role, n: usize| {
        let mut v = corpus::source_role_stream(&m, &root, id, role).unwrap();
        v.truncate(n);
        v
    };
    let probe = slice("gutenberg-books", corpus::Role::Probe, 50_000);
    let held: Vec<(&str, Vec<u8>)> = vec![("fiction", slice("gutenberg-books", corpus::Role::Dev, held_n)), ("nonfiction", slice("gutenberg-nonfiction", corpus::Role::Dev, held_n))];
    let mut kn = LeanKn::new(5);
    for i in 0..exposure.len() {
        kn.observe(&exposure[i.saturating_sub(4)..i], exposure[i]);
    }
    let kidx = kgram_index(&exposure, 10);
    let novel: Vec<Vec<bool>> = held.iter().map(|(_, t)| novel_positions(t, &kidx, 10)).collect();
    let base_cfg = Config { lr: 0.001, ..Config::default() };

    // Per-arm selection of (beta, s0, lr multiplier) on the probe slice, first seed, short exposure.
    let mut selection = Vec::new();
    let mut chosen = std::collections::HashMap::new();
    for (name, init) in [("F", CircuitInit::Compiled), ("E", CircuitInit::Random)] {
        let mut best = (f64::INFINITY, SCALARS[0], 1.0f32);
        for &sc in &SCALARS {
            for &mult in &LR_MULT {
                let mut g = Bptt::with_budget(seeds[0], Config { circuit: Some(circuit(init, sc, mult, false)), ..base_cfg }, 20_000);
                g.observe_all(&exposure[..sel_n.min(exposure.len())]);
                let b = g.evaluate(&probe);
                println!("select {name} beta {} s0 {} lr_mult {mult}: probe BPB {b:.4}", sc.0, sc.1);
                selection.push(json!({"arm": name, "beta": sc.0, "s0": sc.1, "lr_mult": mult, "probe_bpb": b}));
                if b < best.0 { best = (b, sc, mult); }
            }
        }
        chosen.insert(name, (best.1, best.2));
        println!("chosen {name}: beta {} s0 {} lr_mult {}", (best.1).0, (best.1).1, best.2);
    }
    let (fsc, fm) = chosen["F"];
    let (esc, em) = chosen["E"];
    let f_cfg = Config { circuit: Some(circuit(CircuitInit::Compiled, fsc, fm, false)), ..base_cfg };
    let d_cfg = Config { circuit: Some(circuit(CircuitInit::Compiled, fsc, fm, true)), ..base_cfg };
    let e_cfg = Config { circuit: Some(circuit(CircuitInit::Random, esc, em, false)), ..base_cfg };
    let f_params = Bptt::with_budget(seeds[0], f_cfg, 20_000).param_count();
    let plus_budget = (20_000..80_000).step_by(1000).find(|&b| Bptt::with_budget(seeds[0], base_cfg, b).param_count() >= f_params).expect("budget");
    let a_params = Bptt::with_budget(seeds[0], base_cfg, 20_000).param_count();
    let ap_params = Bptt::with_budget(seeds[0], base_cfg, plus_budget).param_count();
    println!("parameters: A {a_params}, A+ {ap_params} (budget {plus_budget}), F/D/E {f_params}");

    // Mixer rate: one choice for every arm, on the probe slice with the plain GRU trained on the selection exposure.
    let mut g0 = Bptt::with_budget(seeds[0], base_cfg, 20_000);
    g0.observe_all(&exposure[..sel_n.min(exposure.len())]);
    let msel: Vec<f64> = MIX_LR.iter().map(|&lr| mean(&run_hybrid(&g0, &kn, &probe, lr).1)).collect();
    let mlr = MIX_LR[(0..MIX_LR.len()).min_by(|&i, &j| msel[i].partial_cmp(&msel[j]).unwrap()).unwrap()];
    println!("mixer lr grid {MIX_LR:?} probe BPB {msel:?} -> {mlr}");
    drop(g0);

    let arm_cfgs: Vec<(&str, Config, usize)> = vec![("A", base_cfg, 20_000), ("A+", base_cfg, plus_budget), ("D", d_cfg, 20_000), ("F", f_cfg, 20_000), ("E", e_cfg, 20_000)];
    // Held-out text for the learning curve: the first 30,000 bytes of each role.
    let curve_text: Vec<u8> = held.iter().flat_map(|(_, t)| t[..30_000.min(t.len())].to_vec()).collect();
    struct Out { seed: u64, arm: &'static str, role: &'static str, alone: Vec<f32>, mixed: Vec<f32>, full: Vec<f32> }
    let mut outs: Vec<Out> = Vec::new();
    let mut curves: Vec<(u64, &str, Curve)> = Vec::new();
    let mut train_secs: Vec<Value> = Vec::new();
    for &seed in &seeds {
        for &(name, cfg, budget) in &arm_cfgs {
            let t = Instant::now();
            let mut g = Bptt::with_budget(seed, cfg, budget);
            let mut curve = Vec::new();
            let mut done = 0;
            for &cp in CURVE.iter().filter(|c| **c <= exposure.len()) {
                g.observe_all(&exposure[done..cp]);
                done = cp;
                curve.push((cp, g.evaluate(&curve_text)));
            }
            let secs = t.elapsed().as_secs_f64();
            println!("seed {seed} {name}: trained {} bytes in {secs:.0}s (incl. curve evals), curve {:?}", exposure.len(), curve.iter().map(|c| format!("{:.3}", c.1)).collect::<Vec<_>>());
            train_secs.push(json!({"seed": seed, "arm": name, "wall_s_incl_curve_evals": secs, "params": g.param_count()}));
            for (role, text) in &held {
                let (alone, mixed, full) = run_hybrid(&g, &kn, text, mlr);
                println!("   {role}: alone {:.3}  +KN {:.3}  +KN+match {:.3}", mean(&alone), mean(&mixed), mean(&full));
                outs.push(Out { seed, arm: name, role, alone, mixed, full });
            }
            curves.push((seed, name, curve));
        }
    }

    // Paired contrasts over seeds, overall and on novel-context positions.
    let pick = |seed: u64, arm: &str, role: &str| outs.iter().find(|o| o.seed == seed && o.arm == arm && o.role == role).unwrap();
    type Sel = fn(&Out) -> &Vec<f32>;
    let (alone, mixed, full): (Sel, Sel, Sel) = (|o| &o.alone, |o| &o.mixed, |o| &o.full);
    let contrasts: Vec<(&str, &str, Sel, &str, Sel)> = vec![
        ("F vs E (compiled init over random init, same architecture)", "E", alone, "F", alone),
        ("F vs A (circuit over plain GRU)", "A", alone, "F", alone),
        ("F vs A+ (circuit over parameter-matched GRU)", "A+", alone, "F", alone),
        ("F vs D (training the compiled circuit over freezing it)", "D", alone, "F", alone),
        ("D vs A (frozen compiled circuit over plain GRU)", "A", alone, "D", alone),
        ("F+KN vs A+KN = G (circuit inside the hybrid)", "A", mixed, "F", mixed),
        ("F+KN+match vs A+KN+match = H", "A", full, "F", full),
        ("E+KN vs A+KN = G (random circuit inside the hybrid)", "A", mixed, "E", mixed),
    ];
    let mut crows = Vec::new();
    for (ri, (role, _)) in held.iter().enumerate() {
        for scope in ["all", "novel-context"] {
            for (name, ba, sa, bb, sb) in &contrasts {
                let mut gains = Vec::new();
                for &seed in &seeds {
                    let (x, y) = (sa(pick(seed, ba, role)), sb(pick(seed, bb, role)));
                    let (x, y): (Vec<f32>, Vec<f32>) = if scope == "all" { (x.clone(), y.clone()) } else { (x.iter().zip(&novel[ri]).filter(|p| *p.1).map(|p| *p.0).collect(), y.iter().zip(&novel[ri]).filter(|p| *p.1).map(|p| *p.0).collect()) };
                    gains.push(paired_ci(&x, &y, 10_000, 500, 20261129).0);
                }
                let n = gains.len() as f64;
                let mu = gains.iter().sum::<f64>() / n;
                let sd = (gains.iter().map(|g| (g - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
                println!("{role:<10} {scope:<13} {name:<62} gain {mu:+.4} seeds {gains:.4?}");
                crows.push(json!({"role": role, "scope": scope, "contrast": name, "mean_gain_bpb": mu, "sd_over_seeds": sd, "seed_gains": gains, "novel_positions": novel[ri].iter().filter(|x| **x).count()}));
            }
        }
    }
    // Bytes to target on the learning curve.
    let mut btt = Vec::new();
    for &seed in &seeds {
        let get = |arm: &str| curves.iter().find(|c| c.0 == seed && c.1 == arm).unwrap().2.clone();
        for (tname, tarm) in [("A final", "A"), ("E final", "E")] {
            let target = get(tarm).last().unwrap().1;
            for arm in ["A", "A+", "D", "F", "E"] {
                btt.push(json!({"seed": seed, "target": tname, "target_bpb": target, "arm": arm, "bytes": bytes_to_target(&get(arm), target)}));
            }
        }
    }
    let arm_rows: Vec<Value> = outs.iter().map(|o| json!({"seed": o.seed, "arm": o.arm, "role": o.role, "bpb_alone": mean(&o.alone), "bpb_with_kn": mean(&o.mixed), "bpb_with_kn_and_match": mean(&o.full)})).collect();
    let curve_rows: Vec<Value> = curves.iter().map(|c| json!({"seed": c.0, "arm": c.1, "curve": c.2.iter().map(|p| json!({"exposure_bytes": p.0, "bpb": p.1})).collect::<Vec<_>>()})).collect();
    let summary = json!({"report": "circuit", "split_role": "dev", "exposure_bytes": exposure.len(), "exposure_sha256": corpus::sha256_hex(&exposure), "manifest_sha256": corpus::sha256_hex(manifest_text.as_bytes()),
        "heldout_bytes_per_role": held_n, "seeds": seeds, "params": {"A": a_params, "A+": ap_params, "F_D_E": f_params},
        "selection": selection, "chosen": {"F": {"beta": fsc.0, "s0": fsc.1, "lr_mult": fm}, "E": {"beta": esc.0, "s0": esc.1, "lr_mult": em}}, "mixer_lr": mlr, "mixer_lr_probe_bpb": msel,
        "arms": arm_rows, "contrasts": crows, "curves": curve_rows, "bytes_to_target": btt, "train_wall": train_secs,
        "scope_note": "Dev role, one exposure budget, GRU seeds differ; count and match models are deterministic. Equal exposure; training wall time is reported, not equalised. The circuit is a host-side reference, not yet compiled onto the neural ISA, so no mainline claim follows. The hidden role is unread."});
    let dir = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/circuit".into()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
}
