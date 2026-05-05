use crate::core::board::Board;
use crate::core::movegen::generate_legal;
use crate::core::moves::{Move, MoveList};
use crate::core::types::Color;
use super::Engine;
use super::eval::{static_eval, CHECKMATE_SCORE};

const NEG_INF: i32 = -(CHECKMATE_SCORE + 1);
const POS_INF: i32 =   CHECKMATE_SCORE + 1;

pub struct AlphaBetaEngine {
    depth: u32,
    name:  String,
}

impl AlphaBetaEngine {
    /// `depth` is the number of half-moves (plies) to search.
    /// Minimum effective depth is 1; values below that are clamped.
    pub fn new(depth: u32) -> Self {
        let depth = depth.max(1);
        AlphaBetaEngine { depth, name: format!("Alpha-Beta (d={depth})") }
    }
}

impl Engine for AlphaBetaEngine {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let mut b = board.clone();
        let legal = generate_legal(&b);
        if legal.is_empty() { return None; }

        let mut best_move = None;
        let mut alpha = NEG_INF;

        for mv in order_moves(&b, &legal) {
            let state = b.make_move(mv);
            // Score from the opponent's perspective, then negate for ours.
            let score = -negamax(&mut b, self.depth - 1, -POS_INF, -alpha);
            b.unmake_move(mv, state);

            if score > alpha {
                alpha = score;
                best_move = Some(mv);
            }
        }

        best_move
    }

    fn name(&self) -> &str { self.name.as_str() }
}

/// Negamax with alpha-beta pruning.
/// Returns the score of `board` from the current side's perspective.
fn negamax(board: &mut Board, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    let legal = generate_legal(board);

    // Terminal: checkmate or stalemate.
    if legal.is_empty() {
        return if board.is_in_check() { -CHECKMATE_SCORE } else { 0 };
    }

    // Leaf node: static evaluation.
    if depth == 0 {
        let raw = static_eval(board);
        return if board.side_to_move == Color::White { raw } else { -raw };
    }

    let mut best = NEG_INF;
    for mv in order_moves(board, &legal) {
        let state = board.make_move(mv);
        let score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmake_move(mv, state);

        if score > best { best = score; }
        if score > alpha { alpha = score; }
        if alpha >= beta { break; } // Beta cutoff
    }
    best
}

/// Put captures and en-passant before quiet moves.
/// Better move ordering → more alpha-beta cutoffs → faster search.
fn order_moves(board: &Board, legal: &MoveList) -> Vec<Move> {
    let mut moves: Vec<Move> = legal.as_slice().to_vec();
    moves.sort_unstable_by_key(|mv| {
        if board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant() { 0i32 } else { 1i32 }
    });
    moves
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;
    use crate::core::movegen::generate_legal;

    fn board(fen: &str) -> Board { Board::from_fen(fen).unwrap() }

    #[test]
    fn returns_a_move_from_starting_position() {
        let b = Board::starting_position();
        let mut engine = AlphaBetaEngine::new(3);
        assert!(engine.choose_move(&b).is_some());
    }

    #[test]
    fn returns_none_when_already_mated() {
        // Fool's mate: it is White's turn and they are mated.
        let b = board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        let mut engine = AlphaBetaEngine::new(3);
        assert!(engine.choose_move(&b).is_none());
    }

    #[test]
    fn finds_checkmate_in_one() {
        // Fool's mate setup after 1.f3 e5 2.g4: Black to move, Qd8-h4# is available.
        // Depth 2 is required: the search must see the reply position
        // (White in check with no moves) one level below the leaf.
        let b = board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2");
        let mut engine = AlphaBetaEngine::new(2);
        let mv = engine.choose_move(&b).expect("engine must return a move");

        let mut b2 = b.clone();
        b2.make_move(mv);
        assert!(b2.is_in_check(),             "engine's move must give check");
        assert!(generate_legal(&b2).is_empty(), "engine's move must be checkmate");
    }
}
