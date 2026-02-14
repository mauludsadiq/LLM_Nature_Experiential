
# LLM Nature Experiential

Deterministic perception + ignition kernel scaffold in Rust.

## Purpose
Implements:
- Bayesian sensory update over hidden causes
- Message efficiency & coherence metrics
- Ignition condition (phase transition trigger)

## Build
```bash
cargo build --release
cargo run
```

## Future Data
`data/` will store empirical olfactory & tactile sensor logs.
Each timestep should contain:
- observation index
- likelihood column
- prior belief
- resulting posterior

## Architecture
src/
- adapter.rs: Bayesian update & normalization
- ignition.rs: efficiency + coherence
- main.rs: minimal demo

Designed to integrate later with real sensor likelihoods.
