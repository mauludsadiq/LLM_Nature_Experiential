use ndarray::Array1;

use crate::adapter::normalize;
use crate::memory::MemoryFeatures;
use crate::sensory::ActionParams;

const EPS: f64 = 1e-9;

fn entropy(q: &Array1<f64>) -> f64 {
    let q = q.mapv(|x| x.max(EPS));
    let q = &q / (q.sum().max(EPS));
    -q.iter().map(|&x| x * x.ln()).sum::<f64>()
}

pub fn choose_action(q_state: &Array1<f64>, mem: &MemoryFeatures) -> ActionParams {
    let qn = normalize(q_state);
    let h = entropy(&qn);
    let h_norm = h / ((qn.len() as f64).ln().max(EPS));

    let base = 1.0;
    let sniff = base + 1.25 * h_norm + 0.50 * (1.0 - mem.ignite_rate);
    let touch = 0.25 + 0.75 * (1.0 - h_norm) + 0.25 * mem.ignite_rate;

    ActionParams {
        sniff_strength: sniff.max(0.1).min(3.0),
        touch_pressure: touch.max(0.0).min(3.0),
    }
}
