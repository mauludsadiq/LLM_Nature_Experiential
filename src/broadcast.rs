use ndarray::Array1;

const EPS: f64 = 1e-9;

pub fn rg_avg_pool(v: &Array1<f64>, rg_level: usize) -> Array1<f64> {
    let k = 1usize << rg_level;
    if k <= 1 {
        return v.clone();
    }
    let n = (v.len() / k) * k;
    if n == 0 {
        return Array1::from_vec(vec![0.0]);
    }
    let mut out = Vec::with_capacity(n / k);
    let vv = v.slice(ndarray::s![0..n]).to_vec();
    for chunk in vv.chunks_exact(k) {
        let m = chunk.iter().sum::<f64>() / (k as f64);
        out.push(m);
    }
    Array1::from_vec(out)
}

pub fn expand_rg_to_n(b_rg: &Array1<f64>, n: usize, rg_level: usize) -> Array1<f64> {
    if b_rg.len() == 0 {
        return Array1::from_vec(vec![0.0; n]);
    }
    let k = 1usize << rg_level;
    let mut out: Vec<f64> = Vec::with_capacity(n);
    for &v in b_rg.iter() {
        for _ in 0..k {
            out.push(v);
        }
    }
    out.truncate(n);
    if out.len() < n {
        out.extend(std::iter::repeat(0.0).take(n - out.len()));
    }
    Array1::from_vec(out)
}

pub fn apply_broadcast(q: &Array1<f64>, b_expanded: &Array1<f64>, lambda: f64) -> Array1<f64> {
    let mut logits = q.mapv(|x| x.max(EPS).ln());
    for i in 0..q.len() {
        logits[i] += lambda * b_expanded[i];
    }
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex = logits.mapv(|x| (x - m).exp());
    &ex / (ex.sum().max(EPS))
}
