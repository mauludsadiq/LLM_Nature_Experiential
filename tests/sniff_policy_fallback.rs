use std::process::Command;

fn run(bin: &str) {
    let out = Command::new("cargo")
        .args(["run", "-q", "--bin", bin])
        .output()
        .expect("cargo run failed");
    assert!(out.status.success(), "nonzero exit status");
}

const CANONICAL: &str = r#"{"t":0,"o":[1,2],"A_shape":[4,3,5],"A_flat_col":[0.2,0.6,0.1,0.1],"p_prior":[0.4,0.2,0.2,0.2],"task_vec":[0.0,1.0,0.0,0.0],"q0":[0.35,0.22,0.25,0.18],"sniff_strength":1.2,"touch_pressure":0.0}
{"t":1,"o":[1,2],"A_shape":[4,3,5],"A_flat_col":[0.25,0.55,0.10,0.10],"p_prior":[0.4,0.2,0.2,0.2],"task_vec":[0.0,1.0,0.0,0.0],"sniff_strength":0.6,"touch_pressure":0.3}
"#;

const POLICY_FIRE: &str = r#"{"t":0,"o":[1,2],"A_shape":[4,3,5],"A_flat_col":[0.2,0.6,0.1,0.1],"p_prior":[0.4,0.2,0.2,0.2],"task_vec":[0.0,1.0,0.0,0.0],"q0":[0.35,0.22,0.25,0.18],"sniff_strength":null,"touch_pressure":null}
{"t":1,"o":[1,2],"A_shape":[4,3,5],"A_flat_col":[0.25,0.55,0.10,0.10],"p_prior":[0.4,0.2,0.2,0.2],"task_vec":[0.0,1.0,0.0,0.0],"sniff_strength":0.6,"touch_pressure":0.3}
"#;

#[test]
fn policy_fallback_fixture_behaves() {
    let path = "data/sniff_stream.ndjson";
    let original = std::fs::read_to_string(path).unwrap_or_default();

    std::fs::write(path, CANONICAL).unwrap();
    run("sniff_loop");
    let replay = std::fs::read_to_string("out/replay_loop.ndjson").unwrap_or_default();
    assert!(
        !replay.contains(r#""action_source":"policy""#),
        "unexpected policy action in canonical fixture"
    );

    std::fs::write(path, POLICY_FIRE).unwrap();
    run("sniff_loop");
    let replay2 = std::fs::read_to_string("out/replay_loop.ndjson").unwrap_or_default();
    assert!(
        replay2.contains(r#""action_source":"policy""#),
        "expected policy action not found in policy-fire fixture"
    );

    std::fs::write(path, original).unwrap();
}
