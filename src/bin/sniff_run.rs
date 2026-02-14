use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use ndarray::{Array1, Array2};

use llm_nature_experiential::adapter::{normalize, bayes_update};
use llm_nature_experiential::ignition::{Params, coherence, efficiency};

const EPS: f64 = 1e-9;

#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
struct SniffEvent {
    t: u64,
    o: Vec<usize>,
    A_shape: Vec<usize>,
    A_flat_col: Vec<f64>,
    p_prior: Vec<f64>,
    q_before: Vec<f64>,
    task_vec: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SniffStepOut {
    t: u64,
    o_idx: usize,
    u_t: f64,
    g_before: f64,
    g_after_local: f64,
    g_after_broadcast: f64,
    d_g_local: f64,
    d_g_broadcast: f64,
    ignited: bool,
    ignite_reason: String,
    theta: f64,
    survivors_n: usize,
    coherence: f64,
    survivor_levels: Vec<u8>,
    q_after: Vec<f64>,
    broadcast: Vec<f64>,
    dx: Vec<f64>,
    e: f64,
    p: f64,
    k: f64,
    eta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ReplayRow {
    t: u64,
    o_idx: usize,
    survivors_n: usize,
    survivor_levels: Vec<u8>,
    broadcast: Vec<f64>,
    b_expanded: Vec<f64>,
    q_after: Vec<f64>,
    q_broadcast: Vec<f64>,
    ignited: bool,
    ignite_reason: String,
    g_before: f64,
    g_after_local: f64,
    g_after_broadcast: f64,
    d_g_local: f64,
    d_g_broadcast: f64,
}

#[derive(Clone, Debug)]
struct Msg {
    // keep for later replay/audit (and to kill warnings later by logging)
    level: u8,
    dx: Array1<f64>,
    prec: Array1<f64>,
    e: f64,
    p: f64,
    k: f64,
}

fn ravel_multi_index(o: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for (i, dim) in shape.iter().rev().enumerate() {
        let oi = o[o.len() - 1 - i];
        idx += oi * stride;
        stride *= *dim;
    }
    idx
}

fn entropy(q: &Array1<f64>) -> f64 {
    let q = q.mapv(|x| x.max(EPS));
    let q = &q / (q.sum().max(EPS));
    -q.iter().map(|&x| x * x.ln()).sum::<f64>()
}

fn kl(q: &Array1<f64>, p: &Array1<f64>) -> f64 {
    let q = q.mapv(|x| x.max(EPS));
    let p = p.mapv(|x| x.max(EPS));
    let q = &q / (q.sum().max(EPS));
    let p = &p / (p.sum().max(EPS));
    q.iter()
        .zip(p.iter())
        .map(|(&qi, &pi)| qi * (qi.ln() - pi.ln()))
        .sum()
}

// G = KL(q||p) - Eq log p(o|s) ; likelihood_col is p(o|s) already selected
fn vfe(q: &Array1<f64>, p_prior: &Array1<f64>, likelihood_col: &Array1<f64>) -> f64 {
    let qn = normalize(q);
    let pn = normalize(p_prior);
    let lik = likelihood_col.mapv(|x| x.max(EPS));
    let eloglik = qn
        .iter()
        .zip(lik.iter())
        .map(|(&qi, &li)| qi * li.ln())
        .sum::<f64>();
    kl(&qn, &pn) - eloglik
}

fn rg_avg_pool(v: &Array1<f64>, rg_level: usize) -> Array1<f64> {
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

fn precision_from_dx(dx: &Array1<f64>, task: &Array1<f64>, level: u8) -> Array1<f64> {
    let mut prec = dx.mapv(|x| x.abs());
    if level == 1 {
        let l = prec.len().min(task.len());
        for i in 0..l {
            prec[i] += 0.1 * task[i];
        }
    }
    let norm = prec.iter().map(|x| x * x).sum::<f64>().sqrt().max(EPS);
    prec.mapv(|x| x / norm)
}

fn msg_metrics(q_before: &Array1<f64>, q_after: &Array1<f64>, task: &Array1<f64>, level: u8) -> Msg {
    let qb = normalize(q_before);
    let qa = normalize(q_after);
    let dx = &qa - &qb;

    let e = kl(&qa, &qb);
    let p = entropy(&qb) - entropy(&qa);
    let l2 = dx.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nnz = dx.iter().filter(|&&x| x.abs() > 1e-6).count() as f64;
    let k = l2 + 0.5 * (nnz / (dx.len() as f64 + EPS));

    let prec = precision_from_dx(&dx, task, level);

    Msg { level, dx, prec, e, p, k }
}

fn task_biased_belief(q: &Array1<f64>, task: &Array1<f64>, gain: f64) -> Array1<f64> {
    let mut logits = q.mapv(|x| x.max(EPS).ln());
    let l = logits.len().min(task.len());
    for i in 0..l {
        logits[i] += gain * task[i];
    }
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex = logits.mapv(|x| (x - m).exp());
    &ex / (ex.sum().max(EPS))
}

fn expand_rg_to_n(b_rg: &Array1<f64>, n: usize, rg_level: usize) -> Array1<f64> {
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

fn apply_broadcast(q: &Array1<f64>, b_expanded: &Array1<f64>, lambda: f64) -> Array1<f64> {
    let mut logits = q.mapv(|x| x.max(EPS).ln());
    for i in 0..q.len() {
        logits[i] += lambda * b_expanded[i];
    }
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex = logits.mapv(|x| (x - m).exp());
    &ex / (ex.sum().max(EPS))
}

fn main() -> anyhow::Result<()> {
    // harness defaults; later: load from config
    let params = Params {
        alpha: 0.10,
        beta: 0.25,
        gamma: 0.80,
        c_crit: 0.70,
        delta: 0.05,
    };

    let rg_level = 1usize; // avg_pool by 2
    let rg_cost = 0.1f64;  // explicit complexity increment after RG
    let lambda_broadcast = 1.0f64;

    let in_path = "data/sniff_events.ndjson";
    let out_path = "out/trace.ndjson";
    std::fs::create_dir_all("out")?;

    let fin = BufReader::new(File::open(in_path)?);
    let mut fout = File::create(out_path)?;
    let mut freplay = File::create("out/replay.ndjson")?;

    for line in fin.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let ev: SniffEvent = serde_json::from_str(&line)?;

        let n = ev.p_prior.len();
        if ev.A_shape.is_empty() || ev.A_shape[0] != n {
            anyhow::bail!("A_shape mismatch: expected first dim {}, got {:?}", n, ev.A_shape);
        }

        let o_shape = &ev.A_shape[1..];
        let o_idx = ravel_multi_index(&ev.o, o_shape);

        let p_prior = Array1::from(ev.p_prior);
        let q_before = Array1::from(ev.q_before);
        let task = Array1::from(ev.task_vec);

        let lik_col = Array1::from(ev.A_flat_col);
        if lik_col.len() != n {
            anyhow::bail!("A_flat_col len mismatch: got {}, expected {}", lik_col.len(), n);
        }

        let u_t = entropy(&normalize(&q_before)) / ((n as f64).ln().max(EPS));
        let g_before = vfe(&q_before, &p_prior, &lik_col);

        // level0: sensory posterior
        let q_after = bayes_update(&normalize(&p_prior), &normalize(&lik_col));
        let g_after_local = vfe(&q_after, &p_prior, &lik_col);

        // level1: task interpretation (small bias)
        let q_task = task_biased_belief(&q_after, &task, 0.05);

        // two messages
        let m0 = msg_metrics(&q_before, &q_after, &task, 0);
        let m1 = msg_metrics(&q_after, &q_task, &task, 1);

        let theta = params.alpha + params.beta * u_t;

        // survivors with RG check
        let mut survivors: Vec<Msg> = Vec::new();
        for m in [m0, m1] {
            let eta = efficiency(m.e, m.p, m.k);
            if eta < theta {
                continue;
            }
            let dx_rg = rg_avg_pool(&m.dx, rg_level);
            let k_rg = m.k + rg_cost;
            let eta_rg = efficiency(m.e, m.p, k_rg);
            if eta_rg >= params.gamma * theta {
                survivors.push(Msg { dx: dx_rg, ..m });
            }
        }

        let survivors_n = survivors.len();
        let survivor_levels: Vec<u8> = survivors.iter().map(|m| m.level).collect();

        // coherence across survivor precision vectors
        let coh = if survivors_n == 0 {
            0.0
        } else {
            let d = survivors[0].prec.len();
            let mut mat: Vec<f64> = Vec::with_capacity(survivors_n * d);
            for m in &survivors {
                mat.extend_from_slice(m.prec.as_slice().unwrap());
            }
            let precisions = Array2::from_shape_vec((survivors_n, d), mat).unwrap();
            coherence(&precisions)
        };

        // broadcast = softmax(eta) weighted sum of survivor dx (RG-space)
        let broadcast = if survivors_n == 0 {
            Array1::from_vec(vec![])
        } else {
            let mut etas: Vec<f64> = survivors.iter().map(|m| efficiency(m.e, m.p, m.k)).collect();
            let max_eta = etas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for x in etas.iter_mut() {
                *x = (*x - max_eta).exp();
            }
            let z = etas.iter().sum::<f64>().max(EPS);

            let mut b = Array1::from_vec(vec![0.0; survivors[0].dx.len()]);
            for (w_raw, m) in etas.iter().zip(survivors.iter()) {
                let w = *w_raw / z;
                b = b + &(m.dx.mapv(|x| x * w));
            }
            b
        };

        // apply broadcast to q_after -> q_broadcast
        let b_expanded = expand_rg_to_n(&broadcast, n, rg_level);
        let q_broadcast = apply_broadcast(&q_after, &b_expanded, lambda_broadcast);
        let g_after_broadcast = vfe(&q_broadcast, &p_prior, &lik_col);

        // ΔG: positive means improvement (drop in G)
        let d_g_local = g_before - g_after_local;
        let d_g_broadcast = g_before - g_after_broadcast;

        // ignition decided on broadcasted improvement
        let ignite_reason = if survivors_n == 0 {
            "no_survivors"
        } else if coh < params.c_crit {
            "coherence_fail"
        } else if d_g_broadcast < params.delta {
            "deltaG_fail"
        } else {
            "ignite"
        }
        .to_string();

        let ignited = ignite_reason == "ignite";

        // trace convenience: record first survivor's metrics if any, else zeros
        let (dx0, e0, p0, k0, eta0) = if survivors_n == 0 {
            (Array1::from_vec(vec![]), 0.0, 0.0, 0.0, 0.0)
        } else {
            let m = &survivors[0];
            let eta0 = efficiency(m.e, m.p, m.k);
            (m.dx.clone(), m.e, m.p, m.k, eta0)
        };

        let out = SniffStepOut {
            t: ev.t,
            o_idx,
            u_t,
            g_before,
            g_after_local,
            g_after_broadcast,
            d_g_local,
            d_g_broadcast,
            ignited,
            ignite_reason: ignite_reason.clone(),
            theta,
            survivors_n,
            coherence: coh,
            survivor_levels: survivor_levels.clone(),
            q_after: q_after.to_vec(),
            broadcast: broadcast.to_vec(),
            dx: dx0.to_vec(),
            e: e0,
            p: p0,
            k: k0,
            eta: eta0,
        };

        let j = serde_json::to_string(&out)?;
        writeln!(fout, "{}", j)?;

        let replay = ReplayRow {
            t: ev.t,
            o_idx,
            survivors_n,
            survivor_levels: survivor_levels.clone(),
            broadcast: broadcast.to_vec(),
            b_expanded: b_expanded.to_vec(),
            q_after: q_after.to_vec(),
            q_broadcast: q_broadcast.to_vec(),
            ignited,
            ignite_reason: ignite_reason.clone(),
            g_before,
            g_after_local,
            g_after_broadcast,
            d_g_local,
            d_g_broadcast,
        };
        let rj = serde_json::to_string(&replay)?;
        writeln!(freplay, "{}", rj)?;
    }

    println!("Wrote {}", out_path);
    Ok(())
}
