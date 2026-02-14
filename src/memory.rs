use serde::{Deserialize, Serialize};

const EPS: f64 = 1e-9;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryFeatures {
    pub t: u64,
    pub window_len: usize,
    pub ignite_rate: f64,
    pub mean_d_g_broadcast: f64,
    pub mean_temperature: f64,
    pub mean_sniff_strength: f64,
    pub mean_touch_pressure: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryRow {
    pub t: u64,
    pub ignited: bool,
    pub d_g_broadcast: f64,
    pub temperature: f64,
    pub sniff_strength: f64,
    pub touch_pressure: f64,
}

#[derive(Clone, Debug)]
pub struct MemoryState {
    pub window_max: usize,
    pub rows: Vec<MemoryRow>,
}

impl MemoryState {
    pub fn new(window_max: usize) -> Self {
        Self {
            window_max,
            rows: Vec::new(),
        }
    }

    pub fn push(&mut self, row: MemoryRow) {
        self.rows.push(row);
        if self.rows.len() > self.window_max {
            let overflow = self.rows.len() - self.window_max;
            self.rows.drain(0..overflow);
        }
    }

    pub fn features(&self, t: u64) -> MemoryFeatures {
        let n = self.rows.len().max(1) as f64;
        let ignite_rate = self.rows.iter().filter(|r| r.ignited).count() as f64 / n;

        let mean_d_g_broadcast = self.rows.iter().map(|r| r.d_g_broadcast).sum::<f64>() / n;
        let mean_temperature = self.rows.iter().map(|r| r.temperature).sum::<f64>() / n;
        let mean_sniff_strength = self.rows.iter().map(|r| r.sniff_strength).sum::<f64>() / n;
        let mean_touch_pressure = self.rows.iter().map(|r| r.touch_pressure).sum::<f64>() / n;

        MemoryFeatures {
            t,
            window_len: self.rows.len(),
            ignite_rate,
            mean_d_g_broadcast,
            mean_temperature,
            mean_sniff_strength,
            mean_touch_pressure,
        }
    }

    pub fn decay_hint(&self) -> f64 {
        let n = self.rows.len() as f64;
        1.0 / (1.0 + (n / 16.0).max(EPS))
    }
}

use crate::policy::MemoryStats;

impl MemoryState {
    pub fn mem_window_len(&self) -> usize {
        self.rows.len()
    }

    pub fn mem_ignite_rate(&self) -> f64 {
        let n = self.rows.len().max(1) as f64;
        self.rows.iter().filter(|r| r.ignited).count() as f64 / n
    }

    pub fn mem_mean_dg_broadcast(&self) -> f64 {
        let n = self.rows.len().max(1) as f64;
        self.rows.iter().map(|r| r.d_g_broadcast).sum::<f64>() / n
    }
}

impl MemoryStats for MemoryState {
    fn mem_ignite_rate(&self) -> f64 {
        self.mem_ignite_rate()
    }

    fn mem_mean_dg_broadcast(&self) -> f64 {
        self.mem_mean_dg_broadcast()
    }
}

impl crate::policy::MemoryStats for MemoryFeatures {
    fn mem_ignite_rate(&self) -> f64 {
        self.ignite_rate
    }

    fn mem_mean_dg_broadcast(&self) -> f64 {
        self.mean_d_g_broadcast
    }
}
