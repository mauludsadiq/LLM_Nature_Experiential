use ndarray::Array2;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Params {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub c_crit: f64,
    pub delta: f64,
}

pub fn efficiency(e: f64, p: f64, k: f64) -> f64 {
    (e + p) / (k + 1e-9)
}

pub fn coherence(p: &Array2<f64>) -> f64 {
    let n = p.nrows();
    if n == 0 { return 0.0; }
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            let a = p.row(i);
            let b = p.row(j);
            let dot = a.dot(&b);
            let na = a.dot(&a).sqrt();
            let nb = b.dot(&b).sqrt();
            sum += dot / ((na * nb) + 1e-9);
        }
    }
    sum / ((n * n) as f64)
}
