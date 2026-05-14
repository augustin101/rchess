// Integration tests for the rchess-uci binary.
//
// Run:
//   cargo test --test uci_pipe
//
// The tests spawn the actual binary so they require a prior `cargo build`.
// Cargo builds binaries automatically before running integration tests, so
// plain `cargo test --test uci_pipe` is enough.

use std::io::Write;
use std::process::{Command, Stdio};

fn uci_bin() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rchess-uci"));
    cmd.arg("--no-nnue");
    cmd
}

// ── Sanity: normal handshake ──────────────────────────────────────────────────

#[test]
fn uci_handshake() {
    let mut child = uci_bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn rchess-uci");

    child.stdin.as_mut().unwrap().write_all(b"uci\nquit\n").unwrap();

    let output = child.wait_with_output().unwrap();
    let stdout  = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("uciok"),     "expected uciok in: {stdout}");
    assert!(stdout.contains("id name"),   "expected id name in: {stdout}");
    assert!(output.status.success(),      "process exited with: {}", output.status);
}

#[test]
fn isready_response() {
    let mut child = uci_bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn rchess-uci");

    child.stdin.as_mut().unwrap().write_all(b"isready\nquit\n").unwrap();

    let output = child.wait_with_output().unwrap();
    let stdout  = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("readyok"), "expected readyok in: {stdout}");
    assert!(output.status.success());
}

#[test]
fn bestmove_starting_position() {
    let mut child = uci_bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn rchess-uci");

    child.stdin.as_mut().unwrap()
        .write_all(b"position startpos\ngo movetime 100\n")
        .unwrap();
    drop(child.stdin.take());

    let output = child.wait_with_output().unwrap();
    let stdout  = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("bestmove"), "expected bestmove in: {stdout}");
}

// ── Broken-pipe resilience ────────────────────────────────────────────────────

/// Close the read end of the pipe right after spawning, then send "uci".
/// The engine will get EPIPE when it tries to write "id name …" / "uciok".
/// It must exit cleanly (not panic — Rust panic exit code is 101).
#[test]
fn no_panic_on_broken_pipe_during_uci() {
    let mut child = uci_bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn rchess-uci");

    // Close the read end before the engine has a chance to write anything.
    drop(child.stdout.take());

    child.stdin.as_mut().unwrap().write_all(b"uci\nquit\n").unwrap();
    drop(child.stdin.take());

    let status = child.wait().unwrap();
    assert_ne!(status.code(), Some(101), "process panicked (exit 101)");
}

/// Same but during a search: the engine is computing and will try to write
/// "bestmove …".  Closing stdout before it finishes must not cause a panic.
#[test]
fn no_panic_on_broken_pipe_during_search() {
    let mut child = uci_bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn rchess-uci");

    // Send the search command while stdout is still open so the engine starts.
    child.stdin.as_mut().unwrap()
        .write_all(b"position startpos\ngo movetime 200\n")
        .unwrap();
    drop(child.stdin.take());

    // Now close the read end mid-search.
    drop(child.stdout.take());

    let status = child.wait().unwrap();
    assert_ne!(status.code(), Some(101), "process panicked (exit 101)");
}
