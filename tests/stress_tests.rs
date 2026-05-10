// Engine stress tests — always #[ignore].
//
// Run the whole family (debug — fast, useful for CI):
//   cargo test --test stress_tests -- --include-ignored
//
// Run in release for meaningful wall-clock timings:
//   cargo test --release --test stress_tests -- --include-ignored --nocapture
//
// Run with the name filter to select just this family:
//   cargo test stress -- --include-ignored
//
// These tests measure wall-clock search time so you can detect performance
// regressions.  They are deliberately separate from the perft ignored tests so
// you can run each family independently.

use rchess::core::board::Board;
use rchess::engine::alpha_beta::AlphaBetaEngine;
use rchess::engine::Engine;

const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE_FEN: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
// A busy middlegame position to exercise the engine in a typical game situation.
const MIDDLEGAME_FEN: &str =
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9";

fn timed_search(fen: &str, depth: u32) -> (std::time::Duration, Option<rchess::core::moves::Move>) {
    let board = Board::from_fen(fen).expect("valid FEN");
    let mut engine = AlphaBetaEngine::with_depth(depth);
    let t0 = std::time::Instant::now();
    let mv = engine.choose_move(&board);
    (t0.elapsed(), mv)
}

/// Starting position at depth 6.
/// A clean baseline: highly symmetric, lots of pruning potential.
#[test]
#[ignore = "stress: run with `cargo test --test stress_tests -- --include-ignored`"]
fn stress_engine_starting_depth6() {
    let (elapsed, mv) = timed_search(STARTING_FEN, 6);
    eprintln!("start/d6  → {:?}  ({:.2?})", mv, elapsed);
    assert!(mv.is_some(), "engine returned no move");
    assert!(elapsed.as_millis() < 120, "search too slow: {:.2?}", elapsed);
}

/// Kiwipete at depth 5.
/// This position has many legal moves and is a common engine benchmark.
#[test]
#[ignore = "stress: run with `cargo test --test stress_tests -- --include-ignored`"]
fn stress_engine_kiwipete_depth5() {
    let (elapsed, mv) = timed_search(KIWIPETE_FEN, 5);
    eprintln!("kiwipete/d5  → {:?}  ({:.2?})", mv, elapsed);
    assert!(mv.is_some(), "engine returned no move");
    assert!(elapsed.as_millis() < 1000, "search too slow: {:.2?}", elapsed);
}

/// Middlegame at depth 7.
/// Tests deeper searches where LMR, null-move pruning, and the TT matter most.
#[test]
#[ignore = "stress: run with `cargo test --test stress_tests -- --include-ignored`"]
fn stress_engine_middlegame_depth7() {
    let (elapsed, mv) = timed_search(MIDDLEGAME_FEN, 7);
    eprintln!("middlegame/d7  → {:?}  ({:.2?})", mv, elapsed);
    assert!(mv.is_some(), "engine returned no move");
    assert!(elapsed.as_millis() < 200, "search too slow: {:.2?}", elapsed);
}
