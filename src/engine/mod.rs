use crate::core::board::Board;
use crate::core::moves::Move;

pub mod alpha_beta;
pub mod eval;
pub mod opening_book;
pub mod random;

pub trait Engine {
    /// Pick a move for the current side to move. Returns `None` if there are
    /// no legal moves (checkmate or stalemate — caller should have checked).
    fn choose_move(&mut self, board: &Board) -> Option<Move>;

    fn name(&self) -> &str;
}
