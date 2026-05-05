use super::types::Square;

pub type Bitboard = u64;

pub const EMPTY: Bitboard = 0;
pub const FULL:  Bitboard = u64::MAX;

// ── File masks ────────────────────────────────────────────────────────────────

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_B: Bitboard = FILE_A << 1;
pub const FILE_C: Bitboard = FILE_A << 2;
pub const FILE_D: Bitboard = FILE_A << 3;
pub const FILE_E: Bitboard = FILE_A << 4;
pub const FILE_F: Bitboard = FILE_A << 5;
pub const FILE_G: Bitboard = FILE_A << 6;
pub const FILE_H: Bitboard = FILE_A << 7;

pub const FILES: [Bitboard; 8] = [
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H,
];

// ── Rank masks ────────────────────────────────────────────────────────────────

pub const RANK_1: Bitboard = 0xFF;
pub const RANK_2: Bitboard = RANK_1 << 8;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_4: Bitboard = RANK_1 << 24;
pub const RANK_5: Bitboard = RANK_1 << 32;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_7: Bitboard = RANK_1 << 48;
pub const RANK_8: Bitboard = RANK_1 << 56;

pub const RANKS: [Bitboard; 8] = [
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8,
];

// ── Bit utilities ─────────────────────────────────────────────────────────────

#[inline]
pub const fn square_bb(sq: Square) -> Bitboard { 1u64 << sq.0 }

#[inline]
pub fn set_bit(bb: &mut Bitboard, sq: Square)   { *bb |=  square_bb(sq); }

#[inline]
pub fn clear_bit(bb: &mut Bitboard, sq: Square) { *bb &= !square_bb(sq); }

#[inline]
pub fn test_bit(bb: Bitboard, sq: Square) -> bool { bb & square_bb(sq) != 0 }

/// Index of the least-significant set bit.
#[inline]
pub fn lsb(bb: Bitboard) -> Square {
    debug_assert_ne!(bb, 0);
    Square(bb.trailing_zeros() as u8)
}

/// Remove and return the least-significant set bit.
#[inline]
pub fn pop_lsb(bb: &mut Bitboard) -> Square {
    let sq = lsb(*bb);
    *bb &= *bb - 1;
    sq
}

#[inline]
pub fn popcount(bb: Bitboard) -> u32 { bb.count_ones() }

/// Returns true if two or more bits are set.
#[inline]
pub fn more_than_one(bb: Bitboard) -> bool { bb != 0 && bb & bb.wrapping_sub(1) != 0 }

// ── Directional shifts ────────────────────────────────────────────────────────
// File-boundary guards prevent wrap-around. Used by pawn and attack generation.

#[inline] pub const fn north(bb: Bitboard) -> Bitboard { bb << 8 }
#[inline] pub const fn south(bb: Bitboard) -> Bitboard { bb >> 8 }
#[inline] pub const fn east (bb: Bitboard) -> Bitboard { (bb & !FILE_H) << 1 }
#[inline] pub const fn west (bb: Bitboard) -> Bitboard { (bb & !FILE_A) >> 1 }
#[inline] pub const fn north_east(bb: Bitboard) -> Bitboard { (bb & !FILE_H) << 9 }
#[inline] pub const fn north_west(bb: Bitboard) -> Bitboard { (bb & !FILE_A) << 7 }
#[inline] pub const fn south_east(bb: Bitboard) -> Bitboard { (bb & !FILE_H) >> 7 }
#[inline] pub const fn south_west(bb: Bitboard) -> Bitboard { (bb & !FILE_A) >> 9 }

// ── Bit iterator ──────────────────────────────────────────────────────────────

/// Iterator over all squares with a set bit, LSB first.
pub struct BitIter(pub Bitboard);

impl Iterator for BitIter {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 { None } else { Some(pop_lsb(&mut self.0)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = popcount(self.0) as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitIter {}

#[inline]
pub fn iter_bits(bb: Bitboard) -> BitIter { BitIter(bb) }

// ── Debug helper ──────────────────────────────────────────────────────────────

/// Print a bitboard as an 8×8 grid with rank 8 at the top.
pub fn print_bb(bb: Bitboard) {
    for rank in (0..8u8).rev() {
        print!("{} ", rank + 1);
        for file in 0..8u8 {
            print!("{} ", if test_bit(bb, Square::new(file, rank)) { '1' } else { '.' });
        }
        println!();
    }
    println!("  a b c d e f g h");
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_bb_roundtrip() {
        for i in 0u8..64 {
            let sq = Square(i);
            assert_eq!(lsb(square_bb(sq)), sq);
        }
    }

    #[test]
    fn pop_lsb_drains_correctly() {
        let mut bb = FILE_A;
        let mut count = 0;
        while bb != 0 {
            let sq = pop_lsb(&mut bb);
            assert_eq!(sq.file(), 0);
            count += 1;
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn directional_shifts_no_wrap() {
        // Shifting east from FILE_H should produce nothing
        assert_eq!(east(FILE_H), EMPTY);
        assert_eq!(west(FILE_A), EMPTY);
        // Shifting the full FILE_A east one step gives FILE_B
        assert_eq!(east(FILE_A), FILE_B);
    }

    #[test]
    fn iter_bits_exact_size() {
        let bb = RANK_1;
        let iter = iter_bits(bb);
        assert_eq!(iter.len(), 8);
        let squares: Vec<_> = iter_bits(bb).collect();
        assert_eq!(squares.len(), 8);
        // All squares on rank 1 (rank index 0)
        for sq in squares {
            assert_eq!(sq.rank(), 0);
        }
    }
}
