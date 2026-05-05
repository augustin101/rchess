// Perft integration tests against known-good node counts.
//
// Fast depths run automatically. Slow depths (≥5 from start, ≥4 from Kiwipete)
// are #[ignore]d and can be run with:
//   cargo test -- --include-ignored

use rchess::core::board::Board;
use rchess::core::perft::perft;

fn run(fen: &str, depth: u32, expected: u64) {
    let mut board = Board::from_fen(fen).unwrap_or_else(|e| {
        panic!("bad FEN \"{fen}\": {e}")
    });
    let got = perft(&mut board, depth);
    assert_eq!(
        got, expected,
        "perft({depth}) for \"{fen}\"\n  got {got}, expected {expected}"
    );
}

// ── Position 1: starting position ────────────────────────────────────────────

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[test] fn perft_start_d1() { run(START, 1,        20); }
#[test] fn perft_start_d2() { run(START, 2,       400); }
#[test] fn perft_start_d3() { run(START, 3,      8_902); }
#[test] fn perft_start_d4() { run(START, 4,    197_281); }
#[test]
#[ignore]
fn perft_start_d5() { run(START, 5,  4_865_609); }
#[test]
#[ignore]
fn perft_start_d6() { run(START, 6, 119_060_324); }

// ── Position 2: Kiwipete ──────────────────────────────────────────────────────

const KIWIPETE: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

#[test] fn perft_kiwipete_d1() { run(KIWIPETE, 1,        48); }
#[test] fn perft_kiwipete_d2() { run(KIWIPETE, 2,     2_039); }
#[test] fn perft_kiwipete_d3() { run(KIWIPETE, 3,    97_862); }
#[test]
#[ignore]
fn perft_kiwipete_d4() { run(KIWIPETE, 4, 4_085_603); }
#[test]
#[ignore]
fn perft_kiwipete_d5() { run(KIWIPETE, 5, 193_690_690); }

// ── Position 3: promotions and en passant ─────────────────────────────────────

const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

#[test] fn perft_pos3_d1() { run(POS3, 1,        14); }
#[test] fn perft_pos3_d2() { run(POS3, 2,       191); }
#[test] fn perft_pos3_d3() { run(POS3, 3,     2_812); }
#[test] fn perft_pos3_d4() { run(POS3, 4,    43_238); }
#[test]
#[ignore]
fn perft_pos3_d5() { run(POS3, 5,   674_624); }
#[test]
#[ignore]
fn perft_pos3_d6() { run(POS3, 6, 11_030_083); }
