use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};

use ndarray::{Array1, Array2};

use llm_nature_experiential::adapter::{bayes_update, normalize};
use llm_nature_experiential::broadcast::{apply_broadcast, expand_rg_to_n, rg_avg_pool};
use llm_nature_experiential::ignition::{coherence, efficiency, Params};
use llm_nature_experiential::ledger::{ndjson_write_row, ReplayRow, TraceRow};
use llm_nature_experiential::memory::{MemoryRow, MemoryState};
use llm_nature_experiential::policy::choose_action;
use llm_nature_experiential::sensory::sensory_from_flat_col;
use llm_nature_experiential::util::{ravel_multi_index, safe_ln_n};

const EPS: f64 = 1e-9;

#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
struct StreamEvent {
    t: u64,
    o: Vec<usize>,
    A_shape: Vec<usize>,
    A_flat_col: Vec<f64>,
    p_prior: Vec<f64>,
    task_vec: Vec<f64>,
    q0: Option<Vec<f64>>,
    sniff_strength: Option<f64>,
    touch_pressure: Option<f64>,
}

#[derive(Clone, Debug)]
struct Msg {
    level: u8,
    dx: Array1<f64>,
    prec: Array1<f64>,
    e: f64,
    p: f64,
    k: f64,
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

// G = KL(q||p) - Eq log p(o|s)
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

fn msg_metrics(
    q_before: &Array1<f64>,
    q_after: &Array1<f64>,
    task: &Array1<f64>,
    level: u8,
) -> Msg {
    let qb = normalize(q_before);
    let qa = normalize(q_after);
    let dx = &qa - &qb;

    let e = kl(&qa, &qb);
    let p = entropy(&qb) - entropy(&qa);
    let l2 = dx.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nnz = dx.iter().filter(|&&x| x.abs() > 1e-6).count() as f64;
    let k = l2 + 0.5 * (nnz / (dx.len() as f64 + EPS));

    let prec = precision_from_dx(&dx, task, level);
    Msg {
        level,
        dx,
        prec,
        e,
        p,
        k,
    }
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

fn main() -> anyhow::Result<()> {
    let params = Params {
        alpha: 0.10,
        beta: 0.25,
        gamma: 0.80,
        c_crit: 0.70,
        delta: 0.05,
    };

    let rg_level = 1usize;
    let rg_cost = 0.1f64;
    let lambda_broadcast = 1.0f64;

    let in_path = "data/sniff_stream.ndjson";
    std::fs::create_dir_all("out")?;
    let mut ftrace = File::create("out/trace_loop.ndjson")?;
    let mut freplay = File::create("out/replay_loop.ndjson")?;

    let fin = BufReader::new(File::open(in_path)?);

    let mut q_state: Option<Array1<f64>> = None;
    let mut mem = MemoryState::new(64);

    for line in fin.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let ev: StreamEvent = serde_json::from_str(&line)?;

        let n = ev.p_prior.len();
        if ev.A_shape.is_empty() || ev.A_shape[0] != n {
            anyhow::bail!(
                "A_shape mismatch: expected first dim {}, got {:?}",
                n,
                ev.A_shape
            );
        }

        let o_shape = &ev.A_shape[1..];
        let o_idx = ravel_multi_index(&ev.o, o_shape);

        let p_prior = Array1::from(ev.p_prior);
        let task = Array1::from(ev.task_vec);

        if q_state.is_none() {
            let q0 = ev.q0.clone().unwrap_or_else(|| p_prior.to_vec());
            q_state = Some(Array1::from(q0));
        }
        let q_before = q_state.as_ref().unwrap().clone();

        let mem_feat_pre = mem.features(ev.t);

        let (sniff_strength, touch_pressure, action_source) =
            match (ev.sniff_strength, ev.touch_pressure) {
                (Some(s), Some(tp)) => (s, tp, "event".to_string()),
                _ => {
                    let a = choose_action(&q_before, &mem_feat_pre);
                    (a.sniff_strength, a.touch_pressure, "policy".to_string())
                }
            };

        let sensory = sensory_from_flat_col(ev.A_flat_col, sniff_strength, touch_pressure);
        let lik_col = Array1::from(sensory.lik_mod.clone());

        let u_t = entropy(&normalize(&q_before)) / safe_ln_n(n);
        let g_before = vfe(&q_before, &p_prior, &lik_col);

        let q_after = bayes_update(&normalize(&p_prior), &normalize(&lik_col));
        let g_after_local = vfe(&q_after, &p_prior, &lik_col);

        let q_task = task_biased_belief(&q_after, &task, 0.05);

        let m0 = msg_metrics(&q_before, &q_after, &task, 0);
        let m1 = msg_metrics(&q_after, &q_task, &task, 1);

        let theta = params.alpha + params.beta * u_t;

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

        let broadcast = if survivors_n == 0 {
            Array1::from_vec(vec![])
        } else {
            let mut etas: Vec<f64> = survivors
                .iter()
                .map(|m| efficiency(m.e, m.p, m.k))
                .collect();
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

        let b_expanded = expand_rg_to_n(&broadcast, n, rg_level);
        let q_broadcast = apply_broadcast(&q_after, &b_expanded, lambda_broadcast);
        let g_after_broadcast = vfe(&q_broadcast, &p_prior, &lik_col);

        let d_g_local = g_before - g_after_local;
        let d_g_broadcast = g_before - g_after_broadcast;

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

        let q_next = if ignited {
            q_broadcast.clone()
        } else {
            q_after.clone()
        };
        q_state = Some(q_next.clone());

        let (dx0, e0, p0, k0, eta0) = if survivors_n == 0 {
            (Array1::from_vec(vec![]), 0.0, 0.0, 0.0, 0.0)
        } else {
            let m = &survivors[0];
            let eta0 = efficiency(m.e, m.p, m.k);
            (m.dx.clone(), m.e, m.p, m.k, eta0)
        };

        let trace = TraceRow {
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
        ndjson_write_row(&mut ftrace, &trace)?;

        mem.push(MemoryRow {
            t: ev.t,
            ignited,
            d_g_broadcast,
            temperature: sensory.temperature,
            sniff_strength,
            touch_pressure,
        });
        let mem_feat_post = mem.features(ev.t);

        let replay = ReplayRow {
            t: ev.t,
            o_idx,
            sniff_strength,
            touch_pressure,
            action_source,
            temperature: sensory.temperature,
            q_before: q_before.to_vec(),
            q_after: q_after.to_vec(),
            q_broadcast: q_broadcast.to_vec(),
            q_next: q_next.to_vec(),
            survivors_n,
            survivor_levels,
            broadcast: broadcast.to_vec(),
            b_expanded: b_expanded.to_vec(),
            ignited,
            ignite_reason,
            g_before,
            g_after_local,
            g_after_broadcast,
            d_g_local,
            d_g_broadcast,
            mem_window_len: mem_feat_post.window_len,
            mem_ignite_rate: mem_feat_post.ignite_rate,
            mem_mean_d_g_broadcast: mem_feat_post.mean_d_g_broadcast,
        };
        ndjson_write_row(&mut freplay, &replay)?;
    }

    println!("Wrote out/trace_loop.ndjson and out/replay_loop.ndjson");
    Ok(())
}
