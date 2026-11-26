//! `zerolm-hybrid --root DATA [--exposure 1000000] [--heldout 200000] [--seeds 9201,9202,9203] [--out DIR]`
//!
//! Experiment N1 (`docs/experimental-design.md`): the bars. Arms on the development role, equal exposure:
//! A online GRU (truncated BPTT), B Kneser-Ney 5, Bm B + match model, G GRU + B mixture, H G + match model.
//! The mixer learning rate is chosen once on a probe-role slice from a grid shared by all mixing arms.
//! The hidden role is never read.
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;
use zerolm_data::{corpus, manifest::Manifest};
use zerolm_eval::bptt::{Bptt, Config};
use zerolm_eval::headroom::{paired_ci, MatchModel};
use zerolm_eval::hybrid::{ctx_of, Mixer};
use zerolm_eval::ngram_lean::LeanKn;

type Pick = fn(&Arms) -> (&Vec<f32>, &Vec<f32>);

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

const LR_GRID: [f64; 4] = [0.003, 0.01, 0.03, 0.1];

/// Per-position bits of the five arms on `text`.
struct Arms {
    a: Vec<f32>,
    b: Vec<f32>,
    bm: Vec<f32>,
    g: Vec<f32>,
    h: Vec<f32>,
}

fn run_arms(gru: &Bptt, kn: &LeanKn, text: &[u8], lr: f64) -> Arms {
    let mut out = Arms { a: vec![], b: vec![], bm: vec![], g: vec![], h: vec![] };
    let mut mixer = Mixer::new(lr);
    let (mut mm_h, mut mm_k) = (MatchModel::new(), MatchModel::new());
    let bits = |p: f64| -p.max(1e-12).log2() as f32;
    let q_of = |pp: f64| if pp < 0.1 { 0 } else if pp < 0.4 { 1 } else if pp < 0.8 { 2 } else { 3 };
    gru.score_probs(text, |i, probs| {
        let hist = &text[i.saturating_sub(4)..i];
        let y = text[i];
        let ctx = ctx_of(probs.iter().cloned().fold(0.0, f32::max));
        let (pa, pk) = (probs[y as usize] as f64, kn.prob(hist, y));
        let pg = mixer.mix(ctx, &[pa, pk]);
        out.a.push(bits(pa));
        out.b.push(bits(pk));
        out.g.push(bits(pg));
        // H: the match layer on top of the mixture.
        let mut qh = 0;
        let ph = match mm_h.predict() {
            Some((pred, len)) => {
                qh = q_of(mixer.mix(ctx, &[probs[pred as usize] as f64, kn.prob(hist, pred)]));
                let c = mm_h.confidence(len, qh);
                (1.0 - c) * pg + if pred == y { c } else { 0.0 }
            }
            None => pg,
        };
        out.h.push(bits(ph));
        // Bm: the match layer on the count model alone.
        let mut qk = 0;
        let pbm = match mm_k.predict() {
            Some((pred, len)) => {
                qk = q_of(kn.prob(hist, pred));
                let c = mm_k.confidence(len, qk);
                (1.0 - c) * pk + if pred == y { c } else { 0.0 }
            }
            None => pk,
        };
        out.bm.push(bits(pbm));
        mixer.update(ctx, &[pa, pk]);
        mm_h.observe(y, qh);
        mm_k.observe(y, qk);
    });
    out
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64
}

fn main() {
    let root = PathBuf::from(arg("--root").expect("--root DIR"));
    let manifest_text = std::fs::read_to_string("research/corpus/manifest.json").expect("manifest");
    let m = Manifest::parse(&manifest_text).expect("manifest");
    assert!(corpus::verify(&m, &root).is_empty());
    let exp_n: usize = arg("--exposure").map(|v| v.parse().unwrap()).unwrap_or(1_000_000);
    let held_n: usize = arg("--heldout").map(|v| v.parse().unwrap()).unwrap_or(200_000);
    let seeds: Vec<u64> = arg("--seeds").map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or_else(|| vec![9201, 9202, 9203]);
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
    let probe = slice("gutenberg-books", corpus::Role::Probe, 100_000);
    let held: Vec<(&str, Vec<u8>)> = vec![("fiction", slice("gutenberg-books", corpus::Role::Dev, held_n)), ("nonfiction", slice("gutenberg-nonfiction", corpus::Role::Dev, held_n))];
    let mut kn = LeanKn::new(5);
    for i in 0..exposure.len() {
        kn.observe(&exposure[i.saturating_sub(4)..i], exposure[i]);
    }
    let cfg = Config { lr: 0.001, ..Config::default() };
    let train = |seed: u64| {
        let t = Instant::now();
        let mut g = Bptt::with_budget(seed, cfg, 20_000);
        g.observe_all(&exposure);
        (g, t.elapsed().as_secs_f64())
    };
    // Mixer learning rate: chosen on the probe slice with the first seed's GRU, G only.
    let (g0, _) = train(seeds[0]);
    let mut first = Some(g0);
    let sel: Vec<f64> = LR_GRID.iter().map(|&lr| mean(&run_arms(first.as_ref().unwrap(), &kn, &probe, lr).g)).collect();
    let best = (0..LR_GRID.len()).min_by(|&i, &j| sel[i].partial_cmp(&sel[j]).unwrap()).unwrap();
    let lr = LR_GRID[best];
    println!("mixer lr grid {LR_GRID:?} probe BPB {sel:?} -> {lr}");
    let mut rows: Vec<Value> = Vec::new();
    let mut per: Vec<(u64, &str, Arms, f64, f64)> = Vec::new();
    for (si, &seed) in seeds.iter().enumerate() {
        let (g, train_s) = match first.take() { Some(g) => (g, f64::NAN), None => train(seed) };
        println!("seed {seed}: GRU trained on {} bytes ({} s)", exposure.len(), if si == 0 { "see first run".to_string() } else { format!("{train_s:.0}") });
        for (role, text) in &held {
            let t = Instant::now();
            let arms = run_arms(&g, &kn, text, lr);
            let wall = t.elapsed().as_secs_f64();
            println!("seed {seed} {role}: A {:.3} B {:.3} Bm {:.3} G {:.3} H {:.3} ({wall:.0}s)", mean(&arms.a), mean(&arms.b), mean(&arms.bm), mean(&arms.g), mean(&arms.h));
            per.push((seed, role, arms, wall, text.len() as f64));
        }
    }
    let contrasts: [(&str, Pick); 5] = [
        ("G vs A (hybrid over GRU alone)", |a| (&a.a, &a.g)),
        ("G vs B (hybrid over Kneser-Ney 5)", |a| (&a.b, &a.g)),
        ("H vs G (match layer over hybrid)", |a| (&a.g, &a.h)),
        ("H vs Bm (GRU adds over count+match)", |a| (&a.bm, &a.h)),
        ("Bm vs B (match over counts)", |a| (&a.b, &a.bm)),
    ];
    let mut summary_rows = Vec::new();
    for (role, _) in &held {
        for (name, f) in contrasts.iter() {
            let mut gains = Vec::new();
            for (seed, r, arms, _, _) in per.iter().filter(|p| p.1 == *role) {
                let (base, other) = f(arms);
                let (d, lo, hi) = paired_ci(base, other, 10_000, 1000, 20261128);
                rows.push(json!({"seed": seed, "role": r, "contrast": name, "gain_bpb": d, "gain_ci95": [lo, hi]}));
                gains.push(d);
            }
            let n = gains.len() as f64;
            let mu = gains.iter().sum::<f64>() / n;
            let sd = (gains.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
            let dz = if sd > 1e-9 { json!(mu / sd) } else { Value::Null }; // seeds vary only the GRU init; a contrast with no seed variance has no d_z
            println!("{role:<10} {name:<40} gain {mu:+.4} over seeds {gains:.4?} d_z {dz}");
            summary_rows.push(json!({"role": role, "contrast": name, "mean_gain_bpb": mu, "sd_over_seeds": sd, "seed_gains": gains, "cohens_dz": dz}));
        }
    }
    let arm_rows: Vec<Value> = per.iter().map(|(seed, role, a, wall, n)| json!({"seed": seed, "role": role, "bpb": {"A": mean(&a.a), "B": mean(&a.b), "Bm": mean(&a.bm), "G": mean(&a.g), "H": mean(&a.h)}, "score_wall_s": wall, "bytes_per_s": n / wall})).collect();
    let summary = json!({"report": "hybrid-bars", "split_role": "dev", "exposure_bytes": exposure.len(), "exposure_sha256": corpus::sha256_hex(&exposure), "manifest_sha256": corpus::sha256_hex(manifest_text.as_bytes()),
        "heldout_bytes_per_role": held_n, "seeds": seeds, "gru_params": 20000, "gru_lr": 0.001, "mixer_lr_grid": LR_GRID, "mixer_lr_probe_bpb": sel, "mixer_lr": lr, "arms": arm_rows, "paired_per_seed": rows, "contrasts": summary_rows,
        "scope_note": "Dev role only, one exposure budget, GRU seeds differ and the count model is deterministic. The mixer and match confidences adapt prequentially on the held-out stream. Equal exposure only; equal-compute accounting is not reported here. Not a ZeroLM-specific result: these are the controls."});
    let dir = PathBuf::from(arg("--out").unwrap_or_else(|| "reports/hybrid".into()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
}
