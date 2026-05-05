use std::sync::OnceLock;
use crate::utils::Xorshift64;

pub struct ZobristKeys {
    /// `piece_sq[color][piece_type][square]`
    pub piece_sq:   [[[u64; 64]; 6]; 2],
    /// XOR into hash when it is Black's turn to move.
    pub side:       u64,
    /// Indexed by the 4-bit castling-rights value (0–15).
    /// `castling[0] == 0` by convention (no rights → zero contribution).
    pub castling:   [u64; 16],
    /// Indexed by file (0–7); XOR in when en-passant is available on that file.
    pub en_passant: [u64; 8],
}

impl ZobristKeys {
    fn generate() -> Self {
        let mut rng = Xorshift64::new(0x246C_CB38_BCEF_3C01_u64);
        let mut next = || rng.next_u64();

        let mut piece_sq = [[[0u64; 64]; 6]; 2];
        for color in &mut piece_sq {
            for pt in color.iter_mut() {
                for sq in pt.iter_mut() {
                    *sq = next();
                }
            }
        }

        let side = next();

        let mut castling = [0u64; 16];
        for entry in castling[1..].iter_mut() {
            *entry = next();
        }

        let mut en_passant = [0u64; 8];
        for entry in &mut en_passant {
            *entry = next();
        }

        ZobristKeys { piece_sq, side, castling, en_passant }
    }
}

static KEYS: OnceLock<ZobristKeys> = OnceLock::new();

#[inline]
pub fn zobrist_keys() -> &'static ZobristKeys {
    KEYS.get_or_init(ZobristKeys::generate)
}
