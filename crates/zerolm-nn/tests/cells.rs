use zerolm_nn::*;
use zerolm_tensor::*;

const E: usize = 16;
const BUDGET: usize = 20_000;

fn inputs(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| (0..E).map(|_| r.uniform(1.0)).collect()).collect()
}

fn weights_bytes(c: &dyn Cell) -> Vec<u8> {
    c.groups().iter().flat_map(|(_, g)| g.iter().flat_map(|v| v.to_bits().to_le_bytes())).collect()
}

#[test]
fn equal_parameter_budgets_within_five_percent() {
    for k in Kind::ALL {
        let c = build_with_budget(k, E, BUDGET, 1);
        let p = c.param_count();
        let dev = (p as f64 - BUDGET as f64).abs() / BUDGET as f64;
        assert!(dev < 0.05, "{k:?}: {p} params, deviation {dev}");
    }
}

#[test]
fn declared_param_count_matches_groups() {
    for k in Kind::ALL {
        let c = build(k, E, 24, 1);
        let n: usize = c.groups().iter().map(|(_, g)| g.len()).sum();
        assert_eq!(n, c.param_count(), "{k:?}");
        assert_eq!(n, param_count_for(k, E, 24), "{k:?} closed form");
    }
}

#[test]
fn same_seed_and_spec_give_byte_identical_weights() {
    for k in Kind::ALL {
        let a = build_with_budget(k, E, BUDGET, 7);
        let b = build_with_budget(k, E, BUDGET, 7);
        let c = build_with_budget(k, E, BUDGET, 8);
        assert_eq!(weights_bytes(a.as_ref()), weights_bytes(b.as_ref()), "{k:?}");
        assert_ne!(weights_bytes(a.as_ref()), weights_bytes(c.as_ref()), "{k:?}");
    }
}

#[test]
fn chunked_processing_matches_token_by_token_exactly() {
    let xs = inputs(5, 200);
    for k in Kind::ALL {
        let cell = build(k, E, 20, 3);
        let mut whole = cell.init_state();
        for x in &xs {
            cell.step(&mut whole, x, None);
        }
        // Resume across arbitrary chunk boundaries through serialization.
        let mut st = cell.init_state();
        for chunk in xs.chunks(37) {
            let bytes = state::encode(k.name(), &st);
            st = state::decode(k.name(), cell.state_len(), &bytes).unwrap();
            for x in chunk {
                cell.step(&mut st, x, None);
            }
        }
        assert_eq!(whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), st.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{k:?}");
    }
}

#[test]
fn ssm_parallel_scan_matches_sequential() {
    let cell = DiagSsm::new(E, 32, 9);
    let xs = inputs(11, 500);
    let mut seq = cell.init_state();
    for x in &xs {
        cell.step(&mut seq, x, None);
    }
    for chunk in [1, 7, 64, 500] {
        let scan = cell.scan_chunked(&cell.init_state(), &xs, chunk);
        assert!(max_abs_diff(&seq, &scan) <= 1e-5, "chunk {chunk}");
    }
}

#[test]
fn state_size_is_fixed_and_values_stay_bounded_on_long_streams() {
    let xs = inputs(21, 5_000);
    for k in Kind::ALL {
        let cell = build_with_budget(k, E, BUDGET, 2);
        let mut st = cell.init_state();
        let len = st.len();
        for x in &xs {
            cell.step(&mut st, x, None);
        }
        assert_eq!(st.len(), len, "{k:?} state grew");
        assert_eq!(len, cell.state_len());
        assert!(all_finite(&st), "{k:?} non-finite");
        assert!(max_abs(&st) <= 16.0, "{k:?} unbounded: {}", max_abs(&st));
    }
}

#[test]
fn state_decode_rejects_corruption_and_mismatch() {
    let st = vec![1.0f32, 2.0, 3.0];
    let bytes = state::encode("gru", &st);
    assert_eq!(state::decode("gru", 3, &bytes).unwrap(), st);
    let mut bad = bytes.clone();
    bad[8] ^= 1;
    assert_eq!(state::decode("gru", 3, &bad), Err(state::StateError::BadChecksum));
    assert!(matches!(state::decode("ssm", 3, &bytes), Err(state::StateError::WrongCell { .. })));
    assert!(matches!(state::decode("gru", 4, &bytes), Err(state::StateError::WrongLength { .. })));
    assert_eq!(state::decode("gru", 3, &bytes[..6]), Err(state::StateError::Truncated));
}

#[test]
fn trace_hooks_fire_with_declared_count() {
    let xs = inputs(1, 20);
    for k in Kind::ALL {
        let cell = build(k, E, 16, 1);
        let mut st = cell.init_state();
        let mut tr = VecTrace::default();
        for x in &xs {
            cell.step(&mut st, x, Some(&mut tr));
        }
        let per_step = tr.0.len() as f64 / xs.len() as f64;
        assert!(per_step > 0.0 && per_step <= cell.trace_points() as f64, "{k:?}: {per_step}");
        assert!(tr.0.iter().all(|(_, v)| all_finite(v)));
    }
}

#[test]
fn severed_recurrence_makes_output_independent_of_history() {
    let xs = inputs(4, 30);
    for k in Kind::ALL {
        let mut cell = build(k, E, 20, 1);
        cell.sever_recurrence();
        let mut a = cell.init_state();
        let mut b = cell.init_state();
        // different histories, same last input
        for x in &xs[..10] {
            cell.step(&mut a, x, None);
        }
        for x in &xs[10..20] {
            cell.step(&mut b, x, None);
        }
        cell.step(&mut a, &xs[25], None);
        cell.step(&mut b, &xs[25], None);
        let d = cell.hidden_dim();
        assert!(max_abs_diff(&a[..d], &b[..d]) <= 1e-6, "{k:?}");
    }
}

#[test]
fn shuffled_recurrence_changes_dynamics() {
    let xs = inputs(4, 50);
    for k in Kind::ALL {
        let intact = build(k, E, 20, 1);
        let mut shuf = build(k, E, 20, 1);
        shuf.shuffle_recurrence(99);
        let (mut a, mut b) = (intact.init_state(), shuf.init_state());
        for x in &xs {
            intact.step(&mut a, x, None);
            shuf.step(&mut b, x, None);
        }
        let d = intact.hidden_dim();
        assert!(max_abs_diff(&a[..d], &b[..d]) > 1e-3, "{k:?}: shuffle had no effect");
    }
}
