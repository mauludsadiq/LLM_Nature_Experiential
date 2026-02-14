const EPS: f64 = 1e-9;

pub fn ravel_multi_index(o: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for (i, dim) in shape.iter().rev().enumerate() {
        let oi = o[o.len() - 1 - i];
        idx += oi * stride;
        stride *= *dim;
    }
    idx
}

pub fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x > 1.0 {
        return 1.0;
    }
    x
}

pub fn safe_ln_n(n: usize) -> f64 {
    (n as f64).ln().max(EPS)
}
