use llm_nature_experiential::broadcast::{apply_broadcast, expand_rg_to_n, rg_avg_pool};
use llm_nature_experiential::ledger::{ReplayRow, TraceRow};
use llm_nature_experiential::memory::MemoryState;
use llm_nature_experiential::policy::choose_action;
use llm_nature_experiential::sensory::sensory_from_flat_col;

#[test]
fn layers_exist_and_link() {
    let _ = rg_avg_pool(&ndarray::Array1::from_vec(vec![1.0, 2.0]), 0);
    let _ = expand_rg_to_n(&ndarray::Array1::from_vec(vec![0.1]), 4, 1);
    let _ = apply_broadcast(
        &ndarray::Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]),
        &ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
        1.0,
    );

    let s = sensory_from_flat_col(vec![0.2, 0.6, 0.1, 0.1], 1.2, 0.0);
    assert_eq!(s.lik_mod.len(), 4);

    let mem = MemoryState::new(8);
    let feat = mem.features(0);
    let a = choose_action(&ndarray::Array1::from_vec(vec![0.4, 0.2, 0.2, 0.2]), &feat);
    assert!(a.sniff_strength > 0.0);

    let _phantom: Option<(TraceRow, ReplayRow)> = None;
}
