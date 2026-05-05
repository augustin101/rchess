//! Named square constants for all 64 squares.
//!
//! Import with `use rchess::core::squares::*;` (or `use super::squares::*;`
//! from within the `core` module) for expressive square literals in tests and
//! engine code.

use super::types::Square;

// ── Rank 1 ────────────────────────────────────────────────────────────────────
pub const A1: Square = Square::new(0, 0);
pub const B1: Square = Square::new(1, 0);
pub const C1: Square = Square::new(2, 0);
pub const D1: Square = Square::new(3, 0);
pub const E1: Square = Square::new(4, 0);
pub const F1: Square = Square::new(5, 0);
pub const G1: Square = Square::new(6, 0);
pub const H1: Square = Square::new(7, 0);

// ── Rank 2 ────────────────────────────────────────────────────────────────────
pub const A2: Square = Square::new(0, 1);
pub const B2: Square = Square::new(1, 1);
pub const C2: Square = Square::new(2, 1);
pub const D2: Square = Square::new(3, 1);
pub const E2: Square = Square::new(4, 1);
pub const F2: Square = Square::new(5, 1);
pub const G2: Square = Square::new(6, 1);
pub const H2: Square = Square::new(7, 1);

// ── Rank 3 ────────────────────────────────────────────────────────────────────
pub const A3: Square = Square::new(0, 2);
pub const B3: Square = Square::new(1, 2);
pub const C3: Square = Square::new(2, 2);
pub const D3: Square = Square::new(3, 2);
pub const E3: Square = Square::new(4, 2);
pub const F3: Square = Square::new(5, 2);
pub const G3: Square = Square::new(6, 2);
pub const H3: Square = Square::new(7, 2);

// ── Rank 4 ────────────────────────────────────────────────────────────────────
pub const A4: Square = Square::new(0, 3);
pub const B4: Square = Square::new(1, 3);
pub const C4: Square = Square::new(2, 3);
pub const D4: Square = Square::new(3, 3);
pub const E4: Square = Square::new(4, 3);
pub const F4: Square = Square::new(5, 3);
pub const G4: Square = Square::new(6, 3);
pub const H4: Square = Square::new(7, 3);

// ── Rank 5 ────────────────────────────────────────────────────────────────────
pub const A5: Square = Square::new(0, 4);
pub const B5: Square = Square::new(1, 4);
pub const C5: Square = Square::new(2, 4);
pub const D5: Square = Square::new(3, 4);
pub const E5: Square = Square::new(4, 4);
pub const F5: Square = Square::new(5, 4);
pub const G5: Square = Square::new(6, 4);
pub const H5: Square = Square::new(7, 4);

// ── Rank 6 ────────────────────────────────────────────────────────────────────
pub const A6: Square = Square::new(0, 5);
pub const B6: Square = Square::new(1, 5);
pub const C6: Square = Square::new(2, 5);
pub const D6: Square = Square::new(3, 5);
pub const E6: Square = Square::new(4, 5);
pub const F6: Square = Square::new(5, 5);
pub const G6: Square = Square::new(6, 5);
pub const H6: Square = Square::new(7, 5);

// ── Rank 7 ────────────────────────────────────────────────────────────────────
pub const A7: Square = Square::new(0, 6);
pub const B7: Square = Square::new(1, 6);
pub const C7: Square = Square::new(2, 6);
pub const D7: Square = Square::new(3, 6);
pub const E7: Square = Square::new(4, 6);
pub const F7: Square = Square::new(5, 6);
pub const G7: Square = Square::new(6, 6);
pub const H7: Square = Square::new(7, 6);

// ── Rank 8 ────────────────────────────────────────────────────────────────────
pub const A8: Square = Square::new(0, 7);
pub const B8: Square = Square::new(1, 7);
pub const C8: Square = Square::new(2, 7);
pub const D8: Square = Square::new(3, 7);
pub const E8: Square = Square::new(4, 7);
pub const F8: Square = Square::new(5, 7);
pub const G8: Square = Square::new(6, 7);
pub const H8: Square = Square::new(7, 7);
