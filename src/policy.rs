use ndarray::Array1;

use crate::adapter::normalize;
use crate::sensory::ActionParams;

const EPS: f64 = 1e-9;

// Minimal interface the policy needs from memory.
// Stable policy <- memory boundary.
pub trait MemoryStats {
    fn mem_ignite_rate(&self) -> f64;
    fn mem_mean_dg_broadcast(&self) -> f64;
}

fn entropy(q: &Array1<f64>) -> f64 {
    let q = q.mapv(|x| x.max(EPS));
    let q = &q / (q.sum().max(EPS));
    -q.iter().map(|&x| x * x.ln()).sum::<f64>()
}

// Policy: choose action parameters from belief state + memory stats + task vector.
// Signature must match callsites in sniff_loop.rs / sniff_run.rs / tests.
pub fn choose_action<M: MemoryStats>(
    q_state: &Array1<f64>,
    mem: &M,
    task: &Array1<f64>,
) -> ActionParams {
    let qn = normalize(q_state);
    let h = entropy(&qn);
    let h_norm = h / ((qn.len() as f64).ln().max(EPS));

    // Simple task influence: average positive mass in task over the belief support.
    let l = qn.len().min(task.len()).max(1);
    let task_mean = task.iter().take(l).map(|x| x.max(0.0)).sum::<f64>() / (l as f64);

    let ignite = mem.mem_ignite_rate();
    let dg = mem.mem_mean_dg_broadcast();

    // Heuristic controls (bounded):
    // - higher uncertainty => more sniff
    // - higher ignite => slightly less sniff / more touch (already aligned)
    // - higher dg => damp sniff a bit (broadcast already doing work)
    // - task_mean => increase sniff (task demands discriminability)
    let base_sniff = 1.0;
    let base_touch = 0.25;

    let sniff = base_sniff + 1.25 * h_norm + 0.50 * (1.0 - ignite) + 0.35 * task_mean - 0.15 * dg;

    let touch = base_touch + 0.75 * (1.0 - h_norm) + 0.25 * ignite + 0.10 * (1.0 - task_mean);

    ActionParams {
        sniff_strength: sniff.max(0.1).min(3.0),
        touch_pressure: touch.max(0.0).min(3.0),
    }
}
