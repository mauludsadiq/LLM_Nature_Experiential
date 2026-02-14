use ndarray::{array, Array2, Axis};
use llm_nature_experiential::adapter::{bayes_update, normalize};
use llm_nature_experiential::ignition::{coherence, efficiency};

fn main() {
    let prior = normalize(&array![0.4, 0.2, 0.2, 0.2]);
    let likelihood = normalize(&array![0.2, 0.6, 0.1, 0.1]);

    let post = bayes_update(&prior, &likelihood);
    println!("Posterior belief: {:?}", post);

    let e = 0.30;
    let p = 0.25;
    let k = 0.90;
    let eta = efficiency(e, p, k);
    println!("Toy efficiency eta: {:.6}", eta);

    let precisions = Array2::from_shape_vec(
        (2, 4),
        vec![
            0.1, 0.7, 0.1, 0.1,
            0.2, 0.6, 0.1, 0.1
        ]
    ).unwrap();
    let coh = coherence(&precisions);
    println!("Toy coherence: {:.6}", coh);

    let _ = precisions.mean_axis(Axis(0)).unwrap();
}
