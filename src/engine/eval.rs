use crate::core::board::Board;
use crate::core::movegen::generate_legal;
use crate::core::types::{Color, PieceType, Square};

/// Score returned (in absolute value) when the side to move is checkmated.
/// Large enough to dominate any material evaluation.
pub const CHECKMATE_SCORE: i32 = 30_000;

// ── Material values (centipawns) ──────────────────────────────────────────────

const PAWN_VAL:   i32 = 100;
const KNIGHT_VAL: i32 = 320;
const BISHOP_VAL: i32 = 330;
const ROOK_VAL:   i32 = 500;
const QUEEN_VAL:  i32 = 900;
// King: no material value (always present, cancels out); only PST is used.

// ── Piece-Square Tables ───────────────────────────────────────────────────────
// All tables are from White's perspective, indexed by Square (a1=0 … h8=63).
// For Black pieces, mirror vertically: pst[sq ^ 56].
// Values are positional bonuses in centipawns.

#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
//  a    b    c    d    e    f    g    h
    0,   0,   0,   0,   0,   0,   0,   0,  // rank 1 – unreachable for pawns
    5,  10,  10, -20, -20,  10,  10,   5,  // rank 2 – discourage d/e-pawn block
    5,  -5, -10,   0,   0, -10,  -5,   5,  // rank 3
    0,   0,   0,  20,  20,   0,   0,   0,  // rank 4 – centre advance
    5,   5,  10,  25,  25,  10,   5,   5,  // rank 5
   10,  10,  20,  30,  30,  20,  10,  10,  // rank 6
   50,  50,  50,  50,  50,  50,  50,  50,  // rank 7 – about to promote
    0,   0,   0,   0,   0,   0,   0,   0,  // rank 8
];

#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
  -50, -40, -30, -30, -30, -30, -40, -50,  // rank 1
  -40, -20,   0,   5,   5,   0, -20, -40,  // rank 2
  -30,   5,  10,  15,  15,  10,   5, -30,  // rank 3
  -30,   0,  15,  20,  20,  15,   0, -30,  // rank 4
  -30,   5,  15,  20,  20,  15,   5, -30,  // rank 5
  -30,   0,  10,  15,  15,  10,   0, -30,  // rank 6
  -40, -20,   0,   0,   0,   0, -20, -40,  // rank 7
  -50, -40, -30, -30, -30, -30, -40, -50,  // rank 8
];

#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
  -20, -10, -10, -10, -10, -10, -10, -20,  // rank 1
  -10,   5,   0,   0,   0,   0,   5, -10,  // rank 2
  -10,  10,  10,  10,  10,  10,  10, -10,  // rank 3
  -10,   0,  10,  10,  10,  10,   0, -10,  // rank 4
  -10,   5,   5,  10,  10,   5,   5, -10,  // rank 5
  -10,   0,   5,  10,  10,   5,   0, -10,  // rank 6
  -10,   0,   0,   0,   0,   0,   0, -10,  // rank 7
  -20, -10, -10, -10, -10, -10, -10, -20,  // rank 8
];

#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
    0,   0,   0,   5,   5,   0,   0,   0,  // rank 1 – slight d/e bonus (open file)
   -5,   0,   0,   0,   0,   0,   0,  -5,  // rank 2
   -5,   0,   0,   0,   0,   0,   0,  -5,  // rank 3
   -5,   0,   0,   0,   0,   0,   0,  -5,  // rank 4
   -5,   0,   0,   0,   0,   0,   0,  -5,  // rank 5
   -5,   0,   0,   0,   0,   0,   0,  -5,  // rank 6
    5,  10,  10,  10,  10,  10,  10,   5,  // rank 7 – 7th rank is powerful
    0,   0,   0,   0,   0,   0,   0,   0,  // rank 8
];

#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
  -20, -10, -10,  -5,  -5, -10, -10, -20,  // rank 1
  -10,   0,   5,   0,   0,   0,   0, -10,  // rank 2
  -10,   5,   5,   5,   5,   5,   0, -10,  // rank 3
    0,   0,   5,   5,   5,   5,   0,  -5,  // rank 4
   -5,   0,   5,   5,   5,   5,   0,  -5,  // rank 5
  -10,   0,   5,   5,   5,   5,   0, -10,  // rank 6
  -10,   0,   0,   0,   0,   0,   0, -10,  // rank 7
  -20, -10, -10,  -5,  -5, -10, -10, -20,  // rank 8
];

#[rustfmt::skip]
const KING_PST: [i32; 64] = [
   20,  30,  10,   0,   0,  10,  30,  20,  // rank 1 – castled king rewarded
   20,  20,   0,   0,   0,   0,  20,  20,  // rank 2
  -10, -20, -20, -20, -20, -20, -20, -10,  // rank 3
  -20, -30, -30, -40, -40, -30, -30, -20,  // rank 4
  -30, -40, -40, -50, -50, -40, -40, -30,  // rank 5
  -30, -40, -40, -50, -50, -40, -40, -30,  // rank 6
  -30, -40, -40, -50, -50, -40, -40, -30,  // rank 7
  -30, -40, -40, -50, -50, -40, -40, -30,  // rank 8
];

#[inline]
fn piece_table(pt: PieceType) -> (i32, &'static [i32; 64]) {
    match pt {
        PieceType::Pawn   => (PAWN_VAL,   &PAWN_PST),
        PieceType::Knight => (KNIGHT_VAL, &KNIGHT_PST),
        PieceType::Bishop => (BISHOP_VAL, &BISHOP_PST),
        PieceType::Rook   => (ROOK_VAL,   &ROOK_PST),
        PieceType::Queen  => (QUEEN_VAL,  &QUEEN_PST),
        PieceType::King   => (0,          &KING_PST),
    }
}

/// Static evaluation of `board` in centipawns from White's perspective.
///
/// - Positive  → White is better
/// - Negative  → Black is better
/// - `0`       → balanced or stalemate
/// - `±CHECKMATE_SCORE` → the side to move is mated
///
/// Accounts for: material, piece-square tables, check, and checkmate/stalemate.
/// Material + PST + check penalty, always from White's perspective.
/// Does **not** detect checkmate or stalemate — use inside a search that
/// already handles terminal positions, to avoid a redundant `generate_legal`.
pub fn static_eval(board: &Board) -> i32 {
    let mut score = 0i32;
    for sq in 0u8..64 {
        let Some(piece) = board.piece_at(Square(sq)) else { continue };
        let (val, pst) = piece_table(piece.piece_type);
        let idx = if piece.color == Color::White { sq as usize } else { sq as usize ^ 56 };
        let contribution = val + pst[idx];
        if piece.color == Color::White { score += contribution } else { score -= contribution }
    }
    if board.is_in_check() {
        score += if board.side_to_move == Color::White { -50 } else { 50 };
    }
    score
}

/// Full evaluation of `board` in centipawns from White's perspective.
///
/// - Positive  → White is better
/// - Negative  → Black is better
/// - `0`       → balanced or stalemate
/// - `±CHECKMATE_SCORE` → the side to move is mated
pub fn evaluate(board: &Board) -> i32 {
    let legal = generate_legal(board);
    if legal.is_empty() {
        return if board.is_in_check() {
            if board.side_to_move == Color::White { -CHECKMATE_SCORE } else { CHECKMATE_SCORE }
        } else {
            0
        };
    }
    static_eval(board)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;

    fn board(fen: &str) -> Board { Board::from_fen(fen).unwrap() }

    #[test]
    fn starting_position_is_balanced() {
        // Perfectly symmetric position: all PST contributions cancel, no check.
        assert_eq!(evaluate(&Board::starting_position()), 0);
    }

    #[test]
    fn material_advantage_is_positive_for_white() {
        // Starting position minus Black queen → White is up ~900 cp.
        let b = board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(evaluate(&b) > 800);
    }

    #[test]
    fn material_advantage_is_negative_for_black() {
        // Starting position minus White queen → Black is up ~900 cp.
        let b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
        assert!(evaluate(&b) < -800);
    }

    #[test]
    fn checkmate_returns_extreme_score() {
        // Fool's mate (1.f3 e5 2.g4 Qh4#): it is White's turn and White is mated.
        let b = board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        assert_eq!(evaluate(&b), -CHECKMATE_SCORE);
    }

    #[test]
    fn stalemate_returns_zero() {
        // Classic stalemate: Black king on a8, White queen on b6, White king on a6.
        // Black to move has no legal moves and is not in check.
        let b = board("k7/8/KQ6/8/8/8/8/8 b - - 0 1");
        assert_eq!(evaluate(&b), 0);
    }
}
