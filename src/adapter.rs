use ndarray::Array1;

pub fn normalize(v: &Array1<f64>) -> Array1<f64> {
    let s: f64 = v.sum();
    v.mapv(|x| (x.max(1e-12)) / s.max(1e-12))
}

pub fn bayes_update(prior: &Array1<f64>, likelihood_col: &Array1<f64>) -> Array1<f64> {
    let post = prior * likelihood_col;
    normalize(&post)
}
