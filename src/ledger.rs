use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Result as IoResult, Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRow {
    pub t: u64,
    pub o_idx: usize,
    pub u_t: f64,
    pub g_before: f64,
    pub g_after_local: f64,
    pub g_after_broadcast: f64,
    pub d_g_local: f64,
    pub d_g_broadcast: f64,
    pub ignited: bool,
    pub ignite_reason: String,
    pub theta: f64,
    pub survivors_n: usize,
    pub coherence: f64,
    pub survivor_levels: Vec<u8>,
    pub q_after: Vec<f64>,
    pub broadcast: Vec<f64>,
    pub dx: Vec<f64>,
    pub e: f64,
    pub p: f64,
    pub k: f64,
    pub eta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRow {
    pub t: u64,
    pub o_idx: usize,

    pub sniff_strength: f64,
    pub touch_pressure: f64,
    pub action_source: String,

    pub temperature: f64,

    pub q_before: Vec<f64>,
    pub q_after: Vec<f64>,
    pub q_broadcast: Vec<f64>,
    pub q_next: Vec<f64>,

    pub survivors_n: usize,
    pub survivor_levels: Vec<u8>,
    pub broadcast: Vec<f64>,
    pub b_expanded: Vec<f64>,

    pub ignited: bool,
    pub ignite_reason: String,

    pub g_before: f64,
    pub g_after_local: f64,
    pub g_after_broadcast: f64,
    pub d_g_local: f64,
    pub d_g_broadcast: f64,

    pub mem_window_len: usize,
    pub mem_ignite_rate: f64,
    pub mem_mean_d_g_broadcast: f64,
}

pub fn ndjson_write_row<T: Serialize>(f: &mut File, row: &T) -> IoResult<()> {
    let s = serde_json::to_string(row).unwrap();
    writeln!(f, "{}", s)
}
