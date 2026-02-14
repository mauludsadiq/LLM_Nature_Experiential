use ndarray::Array1;
use serde::{Deserialize, Serialize};

const EPS: f64 = 1e-9;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionParams {
    pub sniff_strength: f64,
    pub touch_pressure: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensoryOut {
    pub lik_raw: Vec<f64>,
    pub lik_mod: Vec<f64>,
    pub temperature: f64,
}

pub fn modulate_likelihood(
    lik: &Array1<f64>,
    sniff_strength: f64,
    touch_pressure: f64,
) -> (Array1<f64>, f64) {
    let t0 = 1.0f64;
    let k_touch = 0.75f64;

    let denom = (sniff_strength.max(0.0) + k_touch * touch_pressure.max(0.0)).max(EPS);
    let mut temp = t0 / denom;

    let t_min = 0.25f64;
    let t_max = 4.0f64;
    if temp < t_min {
        temp = t_min;
    }
    if temp > t_max {
        temp = t_max;
    }

    let inv_t = 1.0f64 / temp;
    let mut v = lik.mapv(|x| x.max(EPS).powf(inv_t));
    let z = v.sum().max(EPS);
    v.mapv_inplace(|x| x / z);
    (v, temp)
}

pub fn sensory_from_flat_col(
    a_flat_col: Vec<f64>,
    sniff_strength: f64,
    touch_pressure: f64,
) -> SensoryOut {
    let lik_raw = Array1::from(a_flat_col.clone());
    let (lik_mod, temperature) = modulate_likelihood(&lik_raw, sniff_strength, touch_pressure);
    SensoryOut {
        lik_raw: a_flat_col,
        lik_mod: lik_mod.to_vec(),
        temperature,
    }
}
