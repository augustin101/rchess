use super::board::Board;
use super::movegen::generate_legal;

/// Count leaf nodes at `depth` from `board`.
///
/// Depth 0 returns 1 (the position itself). Depth 1 uses bulk-counting:
/// generate legal moves and return the count directly, avoiding a round of
/// make/unmake at the leaves.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let moves = generate_legal(board);
    if depth == 1 { return moves.len() as u64; }

    let mut total = 0u64;
    for &mv in &moves {
        let state = board.make_move(mv);
        total += perft(board, depth - 1);
        board.unmake_move(mv, state);
    }
    total
}

/// Like `perft` but prints each root move with its subtree node count.
pub fn perft_divide(board: &mut Board, depth: u32) -> u64 {
    let moves = generate_legal(board);
    let mut total = 0u64;
    for &mv in &moves {
        let state = board.make_move(mv);
        let nodes = perft(board, depth.saturating_sub(1));
        board.unmake_move(mv, state);
        println!("{mv}: {nodes}");
        total += nodes;
    }
    println!("\nTotal: {total}");
    total
}
