use std::process::Command;

fn run(bin: &str) -> String {
    let out = Command::new("cargo")
        .args(["run", "-q", "--bin", bin])
        .output()
        .expect("cargo run failed");
    assert!(out.status.success(), "nonzero exit status");
    String::from_utf8_lossy(&out.stdout).to_string()
}

#[test]
fn policy_fallback_fixture_behaves() {
    // 1) event fixture: no policy actions expected
    std::fs::copy("data/sniff_stream.ndjson", "data/sniff_stream.ndjson.tmp").unwrap();
    run("sniff_loop");
    let replay = std::fs::read_to_string("out/replay_loop.ndjson").unwrap_or_default();
    assert!(
        !replay.contains(r#""action_source":"policy""#),
        "unexpected policy action in canonical fixture"
    );

    // 2) policy-fire fixture: policy actions expected
    std::fs::copy(
        "data/sniff_stream_policy_fire.ndjson",
        "data/sniff_stream.ndjson",
    )
    .unwrap();
    run("sniff_loop");
    let replay2 = std::fs::read_to_string("out/replay_loop.ndjson").unwrap_or_default();
    assert!(
        replay2.contains(r#""action_source":"policy""#),
        "expected policy action not found in policy-fire fixture"
    );

    // restore
    std::fs::copy("data/sniff_stream.ndjson.tmp", "data/sniff_stream.ndjson").unwrap();
    let _ = std::fs::remove_file("data/sniff_stream.ndjson.tmp");
}
