use crate::core::board::Board;
use crate::core::movegen::generate_legal;
use crate::core::moves::Move;
use crate::utils::Xorshift64;
use super::Engine;

pub struct RandomEngine {
    rng: Xorshift64,
}

impl RandomEngine {
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs() << 17))
            .unwrap_or(0xdeadbeef);
        RandomEngine { rng: Xorshift64::new(seed) }
    }
}

impl Engine for RandomEngine {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let moves = generate_legal(board);
        if moves.is_empty() { return None; }
        Some(moves.as_slice()[self.rng.next_usize() % moves.len()])
    }

    fn name(&self) -> &str { "RandomEngine" }
}
