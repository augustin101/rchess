use crate::core::board::Board;
use crate::core::moves::Move;

pub mod alpha_beta;
pub mod eval;
pub mod nnue;
pub mod opening_book;
pub mod random;
mod pst;

pub trait Engine {
    /// Pick a move for the current side to move. Returns `None` if there are
    /// no legal moves (checkmate or stalemate — caller should have checked).
    fn choose_move(&mut self, board: &Board) -> Option<Move>;

    fn name(&self) -> &str;

    /// True if the most recent `choose_move` call was answered by an opening
    /// book rather than the engine's own search.  Defaults to false.
    fn last_was_book(&self) -> bool { false }

    /// Score (centipawns, from the engine's own side's perspective) produced
    /// by the most recent `choose_move` search.  Returns `None` for engines
    /// that don't search (random mover, opening book moves).
    fn last_score(&self) -> Option<i32> { None }
}
