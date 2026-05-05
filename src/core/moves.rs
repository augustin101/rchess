use std::fmt;
use super::types::{PieceType, Square};

// ── Move flags ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum MoveFlag {
    Normal    = 0,
    Promo     = 1,
    EnPassant = 2,
    Castling  = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum PromoKind {
    Knight = 0,
    Bishop = 1,
    Rook   = 2,
    Queen  = 3,
}

impl PromoKind {
    pub fn piece_type(self) -> PieceType {
        match self {
            PromoKind::Knight => PieceType::Knight,
            PromoKind::Bishop => PieceType::Bishop,
            PromoKind::Rook   => PieceType::Rook,
            PromoKind::Queen  => PieceType::Queen,
        }
    }
}

// ── Move ──────────────────────────────────────────────────────────────────────
//
// Bit layout (u16):
//   bits  0– 5 : from square  (0–63)
//   bits  6–11 : to square    (0–63)
//   bits 12–13 : promo piece  (Knight=0, Bishop=1, Rook=2, Queen=3)
//   bits 14–15 : flag         (Normal=0, Promo=1, EnPassant=2, Castling=3)

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Move = Move(0);

    #[inline]
    pub fn new(from: Square, to: Square, flag: MoveFlag, promo: PromoKind) -> Move {
        Move(
            (from.0 as u16)
            | ((to.0 as u16) << 6)
            | ((promo as u16) << 12)
            | ((flag  as u16) << 14),
        )
    }

    #[inline]
    pub fn normal(from: Square, to: Square) -> Move {
        Move::new(from, to, MoveFlag::Normal, PromoKind::Knight)
    }

    #[inline]
    pub fn promo(from: Square, to: Square, kind: PromoKind) -> Move {
        Move::new(from, to, MoveFlag::Promo, kind)
    }

    #[inline]
    pub fn en_passant(from: Square, to: Square) -> Move {
        Move::new(from, to, MoveFlag::EnPassant, PromoKind::Knight)
    }

    #[inline]
    pub fn castling(from: Square, to: Square) -> Move {
        Move::new(from, to, MoveFlag::Castling, PromoKind::Knight)
    }

    #[inline]
    pub fn from_sq(self) -> Square { Square(( self.0        & 0x3F) as u8) }

    #[inline]
    pub fn to_sq(self) -> Square   { Square(((self.0 >>  6) & 0x3F) as u8) }

    #[inline]
    pub fn promo_kind(self) -> PromoKind {
        match (self.0 >> 12) & 3 {
            0 => PromoKind::Knight,
            1 => PromoKind::Bishop,
            2 => PromoKind::Rook,
            _ => PromoKind::Queen,
        }
    }

    #[inline]
    pub fn flag(self) -> MoveFlag {
        match (self.0 >> 14) & 3 {
            0 => MoveFlag::Normal,
            1 => MoveFlag::Promo,
            2 => MoveFlag::EnPassant,
            _ => MoveFlag::Castling,
        }
    }

    #[inline] pub fn is_null(self)        -> bool { self.0 == 0 }
    #[inline] pub fn is_promo(self)       -> bool { self.flag() == MoveFlag::Promo }
    #[inline] pub fn is_en_passant(self)  -> bool { self.flag() == MoveFlag::EnPassant }
    #[inline] pub fn is_castling(self)    -> bool { self.flag() == MoveFlag::Castling }

    #[inline]
    pub fn promo_piece_type(self) -> PieceType {
        self.promo_kind().piece_type()
    }
}

/// UCI format: "e2e4", "e7e8q"
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from_sq(), self.to_sq())?;
        if self.is_promo() {
            let c = match self.promo_kind() {
                PromoKind::Knight => 'n',
                PromoKind::Bishop => 'b',
                PromoKind::Rook   => 'r',
                PromoKind::Queen  => 'q',
            };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

// ── MoveList ──────────────────────────────────────────────────────────────────

pub struct MoveList {
    moves: [Move; 256],
    len:   usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> Self {
        MoveList { moves: [Move::NULL; 256], len: 0 }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < 256, "MoveList overflow");
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline] pub fn len(&self) -> usize   { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub fn as_slice(&self) -> &[Move] { &self.moves[..self.len] }
}

impl Default for MoveList {
    fn default() -> Self { Self::new() }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;
    fn into_iter(self) -> Self::IntoIter { self.as_slice().iter() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::squares::*;

    #[test]
    fn move_encoding_roundtrip() {
        let from =  E2;
        let to   = E4;
        let mv   = Move::normal(from, to);
        assert_eq!(mv.from_sq(), from);
        assert_eq!(mv.to_sq(),   to);
        assert_eq!(mv.flag(),    MoveFlag::Normal);
    }

    #[test]
    fn promo_encoding() {
        let from = E7;
        let to   = E8;
        let mv   = Move::promo(from, to, PromoKind::Queen);
        assert_eq!(mv.from_sq(),         from);
        assert_eq!(mv.to_sq(),           to);
        assert!(mv.is_promo());
        assert_eq!(mv.promo_piece_type(), PieceType::Queen);
        assert_eq!(mv.to_string(),       "e7e8q");
    }

    #[test]
    fn move_list_push_len() {
        let mut list = MoveList::new();
        for i in 0u8..20 {
            list.push(Move::normal(Square(i), Square(i + 1)));
        }
        assert_eq!(list.len(), 20);
        assert_eq!(list.as_slice().len(), 20);
    }
}
