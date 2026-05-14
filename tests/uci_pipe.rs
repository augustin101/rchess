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

// ── Time management: losing position must not blow the budget ─────────────────

/// Regression test: in a completely losing position, aspiration fail-lows used
/// to call tm.extend() on every retry, compounding geometrically and pushing the
/// hard deadline to ~30 s even with only 10 s on the clock.
///
/// The position is White to move after 82 half-moves from startpos; wtime=10549.
/// The formula gives a hard limit of ~1.2 s. We allow 6 s as a generous ceiling
/// (5× the hard limit) to avoid flakiness on slow CI while still catching the
/// 30-second regression.
#[test]
fn time_budget_respected_in_losing_position() {
    const MOVES: &str = concat!(
        "b1c3 d7d5 d2d4 g8f6 a2a3 c7c5 d4c5 d5d4 c3b1 e7e5 ",
        "e2e3 f8c5 g1f3 b8c6 c2c3 e8g8 e3d4 e5d4 f1e2 d4d3 ",
        "d1d3 d8d3 e2d3 f8e8 d3e2 c8f5 b2b4 c5b6 h2h3 f6d5 ",
        "e1f1 f5b1 a1b1 e8e2 f1e2 d5c3 e2d3 c3b1 c1b2 a8d8 ",
        "d3e2 d8e8 e2d1 b1a3 b2a3 b6f2 d1c2 e8e3 c2b2 a7a6 ",
        "h1d1 f2g3 d1d2 h7h5 d2d5 f7f6 b2a2 e3e2 a2b3 e2g2 ",
        "d5h5 g2f2 f3e1 c6d4 b3c4 d4f5 e1d3 f2c2 c4d5 g7g6 ",
        "h5f5 g6f5 b4b5 c2c3 h3h4 c3d3 d5e6 d3a3 h4h5 f5f4 ",
        "e6f5 f4f3"
    );

    let input = format!(
        "position startpos moves {MOVES}\ngo wtime 10549 btime 20420 winc 1000 binc 1000\n"
    );

    let mut child = uci_bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn rchess-uci");

    child.stdin.as_mut().unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    drop(child.stdin.take());

    let deadline = std::time::Duration::from_secs(6);
    let t0 = std::time::Instant::now();

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let out = child.wait_with_output().unwrap();
        tx.send(out).unwrap();
    });

    let output = rx.recv_timeout(deadline).unwrap_or_else(|_| {
        panic!(
            "engine did not respond within {}s — time management bug in losing position",
            deadline.as_secs()
        )
    });
    handle.join().ok();

    let elapsed = t0.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("bestmove"),
        "expected bestmove in output: {stdout}"
    );
    assert!(
        elapsed < deadline,
        "engine took {:.1}s — exceeded {}s limit",
        elapsed.as_secs_f32(),
        deadline.as_secs()
    );
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
