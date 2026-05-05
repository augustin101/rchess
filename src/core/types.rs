pub const NUM_COLORS: usize = 2;
pub const NUM_PIECE_TYPES: usize = 6;

// ── Color ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(usize)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Color::White => write!(f, "White"),
            Color::Black => write!(f, "Black"),
        }
    }
}

// ── PieceType ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(usize)]
pub enum PieceType {
    Pawn   = 0,
    Knight = 1,
    Bishop = 2,
    Rook   = 3,
    Queen  = 4,
    King   = 5,
}

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    pub fn to_char(self, color: Color) -> char {
        let c = match self {
            PieceType::Pawn   => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook   => 'r',
            PieceType::Queen  => 'q',
            PieceType::King   => 'k',
        };
        if color == Color::White { c.to_ascii_uppercase() } else { c }
    }

    pub fn from_char(c: char) -> Option<(Color, PieceType)> {
        let color = if c.is_uppercase() { Color::White } else { Color::Black };
        let pt = match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _   => return None,
        };
        Some((color, pt))
    }
}

// ── Piece ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece {
    pub color:      Color,
    pub piece_type: PieceType,
}

impl Piece {
    #[inline]
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { color, piece_type }
    }

    #[inline]
    pub fn to_char(self) -> char {
        self.piece_type.to_char(self.color)
    }
}

// ── Square ────────────────────────────────────────────────────────────────────

/// Square index: a1 = 0, b1 = 1, …, h1 = 7, a2 = 8, …, h8 = 63.
/// Bit n in a Bitboard corresponds to Square(n).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Square(pub u8);

impl Square {
    #[inline]
    pub fn new(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8);
        Square(rank * 8 + file)
    }

    #[inline]
    pub fn file(self) -> u8 { self.0 & 7 }

    #[inline]
    pub fn rank(self) -> u8 { self.0 >> 3 }

    pub fn all() -> impl Iterator<Item = Square> {
        (0u8..64).map(Square)
    }
}

impl std::str::FromStr for Square {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        let b = s.as_bytes();
        if b.len() != 2 { return Err(()); }
        let file = b[0].wrapping_sub(b'a');
        let rank  = b[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 { Ok(Square::new(file, rank)) } else { Err(()) }
    }
}

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", (b'a' + self.file()) as char, (b'1' + self.rank()) as char)
    }
}

// ── CastlingRights ────────────────────────────────────────────────────────────

/// Castling availability packed into 4 bits.
///   Bit 0: White kingside  (K)
///   Bit 1: White queenside (Q)
///   Bit 2: Black kingside  (k)
///   Bit 3: Black queenside (q)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const WHITE_KINGSIDE:  u8 = 0b0001;
    pub const WHITE_QUEENSIDE: u8 = 0b0010;
    pub const BLACK_KINGSIDE:  u8 = 0b0100;
    pub const BLACK_QUEENSIDE: u8 = 0b1000;

    pub const fn none() -> Self { CastlingRights(0) }
    pub const fn all()  -> Self { CastlingRights(0b1111) }

    #[inline] pub fn has(self, flag: u8) -> bool  { self.0 & flag != 0 }
    #[inline] pub fn set(&mut self, flag: u8)     { self.0 |=  flag; }
    #[inline] pub fn remove(&mut self, flag: u8)  { self.0 &= !flag; }

    pub fn kingside(self, color: Color) -> bool {
        match color {
            Color::White => self.has(Self::WHITE_KINGSIDE),
            Color::Black => self.has(Self::BLACK_KINGSIDE),
        }
    }

    pub fn queenside(self, color: Color) -> bool {
        match color {
            Color::White => self.has(Self::WHITE_QUEENSIDE),
            Color::Black => self.has(Self::BLACK_QUEENSIDE),
        }
    }
}

impl std::fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            write!(f, "-")
        } else {
            if self.has(Self::WHITE_KINGSIDE)  { write!(f, "K")?; }
            if self.has(Self::WHITE_QUEENSIDE) { write!(f, "Q")?; }
            if self.has(Self::BLACK_KINGSIDE)  { write!(f, "k")?; }
            if self.has(Self::BLACK_QUEENSIDE) { write!(f, "q")?; }
            Ok(())
        }
    }
}
