use crate::core::board::Board;
use crate::core::movegen::generate_legal;
use crate::core::moves::Move;
use super::Engine;

pub struct RandomEngine {
    rng: u64,
}

impl RandomEngine {
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs() << 17))
            .unwrap_or(0xdeadbeef);
        RandomEngine { rng: seed | 1 }
    }

    fn next_usize(&mut self) -> usize {
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng as usize
    }
}

impl Engine for RandomEngine {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let moves = generate_legal(board);
        if moves.is_empty() {
            return None;
        }
        let idx = self.next_usize() % moves.len();
        Some(moves.as_slice()[idx])
    }

    fn name(&self) -> &str {
        "RandomEngine"
    }
}
