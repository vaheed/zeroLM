use zerolm_eval::*;
use zerolm_engine::Model;
use zerolm_nn::Kind;

fn quick_results() -> Vec<CandidateResult> {
    let cfg = Config::quick();
    let mut r: Vec<_> = Kind::ALL.iter().map(|&k| evaluate(k, &cfg)).collect();
    score_all(&mut r);
    r
}

#[test]
fn zeroing_the_recurrent_path_damages_memory_dependent_tasks() {
    for r in quick_results() {
        for a in &r.ablations {
            assert!(a.horizon_intact >= 1, "{}: intact model has no memory at lag 1", r.kind);
            assert_eq!(a.horizon_severed, 0, "{}: severed model still remembers", r.kind);
        }
    }
}

#[test]
fn shuffling_the_recurrent_weights_changes_behaviour() {
    for r in quick_results() {
        for a in &r.ablations {
            assert!(a.shuffle_state_divergence > 1e-3, "{}: {}", r.kind, a.shuffle_state_divergence);
        }
    }
}

#[test]
fn all_candidates_are_stable_with_constant_state() {
    for r in quick_results() {
        assert!(r.stable, "{}", r.kind);
        for s in &r.stability {
            assert_eq!(s.state_len_start, s.state_len_end);
            assert!((s.bits_per_byte_untrained - 8.0).abs() < 0.3, "{}: {}", r.kind, s.bits_per_byte_untrained);
        }
    }
}

#[test]
fn parameter_budgets_are_matched() {
    for r in quick_results() {
        let dev = (r.cell_params as f64 - 4_000.0).abs() / 4_000.0;
        assert!(dev < 0.05, "{}: {}", r.kind, r.cell_params);
    }
}

#[test]
fn hybrid_is_never_selected_and_selection_is_deterministic() {
    let a = quick_results();
    let b = quick_results();
    let (sa, sb) = (select(&a).unwrap(), select(&b).unwrap());
    assert_eq!(sa.kind, sb.kind);
    assert_ne!(sa.kind, "hybrid");
    assert_eq!(sa.score, sb.score);
}

#[test]
fn horizon_is_deterministic_per_seed() {
    let m = Model::new(Kind::Gru, 1, EMBED_DIM, 4_000);
    assert_eq!(sensitivity_by_lag(&m, 1), sensitivity_by_lag(&m, 1));
}

#[test]
fn horizon_function_stops_at_first_drop() {
    let mut s = vec![1.0f32; LAGS.len()];
    s[3] = 0.0;
    assert_eq!(horizon(&s), LAGS[2]);
    assert_eq!(horizon(&vec![0.0; LAGS.len()]), 0);
}
