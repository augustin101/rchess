use std::fmt;

use super::bitboard::*;
use super::moves::{Move, MoveFlag};
use super::types::*;
use super::zobrist::zobrist_keys;

pub const STARTING_FEN: &str =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ── FEN error ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    WrongFieldCount,
    InvalidPiecePlacement(String),
    InvalidSideToMove,
    InvalidCastling,
    InvalidEnPassant,
    InvalidHalfMoveClock,
    InvalidFullMoveNumber,
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongFieldCount =>
                write!(f, "FEN must have exactly 6 space-separated fields"),
            Self::InvalidPiecePlacement(s) =>
                write!(f, "invalid piece placement: {s}"),
            Self::InvalidSideToMove =>
                write!(f, "side to move must be 'w' or 'b'"),
            Self::InvalidCastling =>
                write!(f, "invalid castling availability field"),
            Self::InvalidEnPassant =>
                write!(f, "invalid en-passant square"),
            Self::InvalidHalfMoveClock =>
                write!(f, "invalid half-move clock"),
            Self::InvalidFullMoveNumber =>
                write!(f, "invalid full-move number"),
        }
    }
}

impl std::error::Error for FenError {}

// ── Board ─────────────────────────────────────────────────────────────────────

/// Hybrid board representation.
///
/// **Bit-centric view** — `pieces[color][piece_type]` bitboards plus
/// pre-computed occupancy unions — supports fast bitwise algorithms (move
/// generation, attack masks, magic bitboards).
///
/// **Board-centric view** — `mailbox[square]` gives an O(1) piece lookup by
/// square index, convenient for move legality, display, and FEN serialisation.
///
/// Both views are always kept in sync through `put_piece` / `remove_piece`.
#[derive(Clone, Debug)]
pub struct Board {
    // ── Bit-centric ───────────────────────────────────────────────────────────
    /// `pieces[color][piece_type]` — one bitboard per (color, piece) pair.
    pub pieces: [[Bitboard; NUM_PIECE_TYPES]; NUM_COLORS],
    /// Union of all squares occupied by each color.
    pub occupancy: [Bitboard; NUM_COLORS],
    /// Union of all occupied squares (both colors).
    pub all_occupancy: Bitboard,

    // ── Board-centric (mailbox) ───────────────────────────────────────────────
    /// Direct square → piece mapping for O(1) lookup.
    pub mailbox: [Option<Piece>; 64],

    // ── UCI / game state ──────────────────────────────────────────────────────
    pub side_to_move:     Color,
    pub castling_rights:  CastlingRights,
    pub en_passant:       Option<Square>,
    pub half_move_clock:  u32,
    pub full_move_number: u32,

    /// Incrementally-updated Zobrist hash of the current position.
    pub hash: u64,
}

impl Board {
    // ── Constructors ──────────────────────────────────────────────────────────

    pub fn empty() -> Self {
        Board {
            pieces:           [[EMPTY; NUM_PIECE_TYPES]; NUM_COLORS],
            occupancy:        [EMPTY; NUM_COLORS],
            all_occupancy:    EMPTY,
            mailbox:          [None; 64],
            side_to_move:     Color::White,
            castling_rights:  CastlingRights::none(),
            en_passant:       None,
            half_move_clock:  0,
            full_move_number: 1,
            hash:             0,
        }
    }

    pub fn starting_position() -> Self {
        Self::from_fen(STARTING_FEN).expect("starting FEN is always valid")
    }

    // ── Internal mutation (keeps both views in sync) ──────────────────────────

    pub fn put_piece(&mut self, piece: Piece, sq: Square) {
        let (c, pt) = (piece.color as usize, piece.piece_type as usize);
        set_bit(&mut self.pieces[c][pt], sq);
        set_bit(&mut self.occupancy[c],  sq);
        set_bit(&mut self.all_occupancy, sq);
        self.mailbox[sq.0 as usize] = Some(piece);
        self.hash ^= zobrist_keys().piece_sq[c][pt][sq.0 as usize];
    }

    pub fn remove_piece(&mut self, sq: Square) {
        if let Some(piece) = self.mailbox[sq.0 as usize].take() {
            let (c, pt) = (piece.color as usize, piece.piece_type as usize);
            clear_bit(&mut self.pieces[c][pt], sq);
            clear_bit(&mut self.occupancy[c],  sq);
            clear_bit(&mut self.all_occupancy, sq);
            self.hash ^= zobrist_keys().piece_sq[c][pt][sq.0 as usize];
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// O(1) piece lookup via mailbox.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.0 as usize]
    }

    /// Bitboard for a specific (color, piece type) combination.
    #[inline]
    pub fn piece_bb(&self, color: Color, pt: PieceType) -> Bitboard {
        self.pieces[color as usize][pt as usize]
    }

    // ── FEN ───────────────────────────────────────────────────────────────────

    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(FenError::WrongFieldCount);
        }

        let mut board = Board::empty();

        // 1. Piece placement — FEN lists rank 8 first, rank 1 last
        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError::InvalidPiecePlacement(
                "expected 8 rank sections separated by '/'".into(),
            ));
        }
        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - rank_idx as u8;
            let mut file: u8 = 0;

            for ch in rank_str.chars() {
                if let Some(skip) = ch.to_digit(10) {
                    file += skip as u8;
                } else {
                    let (color, pt) = PieceType::from_char(ch).ok_or_else(|| {
                        FenError::InvalidPiecePlacement(format!("unknown piece '{ch}'"))
                    })?;
                    if file >= 8 {
                        return Err(FenError::InvalidPiecePlacement(
                            format!("file overflow on rank {}", rank + 1),
                        ));
                    }
                    board.put_piece(Piece::new(color, pt), Square::new(file, rank));
                    file += 1;
                }
            }
            if file != 8 {
                return Err(FenError::InvalidPiecePlacement(format!(
                    "rank {} has {file} files, expected 8",
                    rank + 1,
                )));
            }
        }

        // 2. Side to move
        board.side_to_move = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _   => return Err(FenError::InvalidSideToMove),
        };

        // 3. Castling rights
        if fields[2] != "-" {
            for ch in fields[2].chars() {
                match ch {
                    'K' => board.castling_rights.set(CastlingRights::WHITE_KINGSIDE),
                    'Q' => board.castling_rights.set(CastlingRights::WHITE_QUEENSIDE),
                    'k' => board.castling_rights.set(CastlingRights::BLACK_KINGSIDE),
                    'q' => board.castling_rights.set(CastlingRights::BLACK_QUEENSIDE),
                    _   => return Err(FenError::InvalidCastling),
                }
            }
        }

        // 4. En-passant target square
        board.en_passant = if fields[3] == "-" {
            None
        } else {
            Some(fields[3].parse::<Square>().map_err(|_| FenError::InvalidEnPassant)?)
        };

        // 5. Half-move clock
        board.half_move_clock = fields[4]
            .parse()
            .map_err(|_| FenError::InvalidHalfMoveClock)?;

        // 6. Full-move number
        board.full_move_number = fields[5]
            .parse()
            .map_err(|_| FenError::InvalidFullMoveNumber)?;

        // 7. Finish Zobrist hash (pieces already XOR'd in via put_piece)
        let keys = zobrist_keys();
        if board.side_to_move == Color::Black {
            board.hash ^= keys.side;
        }
        board.hash ^= keys.castling[board.castling_rights.0 as usize];
        if let Some(sq) = board.en_passant {
            board.hash ^= keys.en_passant[sq.file() as usize];
        }

        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(90);

        // 1. Piece placement
        for rank in (0..8u8).rev() {
            let mut empty_run: u8 = 0;
            for file in 0..8u8 {
                match self.piece_at(Square::new(file, rank)) {
                    None => empty_run += 1,
                    Some(piece) => {
                        if empty_run > 0 {
                            fen.push((b'0' + empty_run) as char);
                            empty_run = 0;
                        }
                        fen.push(piece.to_char());
                    }
                }
            }
            if empty_run > 0 { fen.push((b'0' + empty_run) as char); }
            if rank > 0 { fen.push('/'); }
        }

        // 2. Side to move
        fen.push(' ');
        fen.push(match self.side_to_move { Color::White => 'w', Color::Black => 'b' });

        // 3. Castling
        fen.push_str(&format!(" {}", self.castling_rights));

        // 4. En-passant
        fen.push(' ');
        match self.en_passant {
            None     => fen.push('-'),
            Some(sq) => fen.push_str(&sq.to_string()),
        }

        // 5 & 6. Clocks
        fen.push_str(&format!(" {} {}", self.half_move_clock, self.full_move_number));

        fen
    }
}

// ── Castling rights update mask ───────────────────────────────────────────────
//
// Index by `from` and `to` of any move, AND both values into castling_rights.0.
// Clears the correct bits for king moves, rook moves, and rook captures with no
// branching.
pub const CASTLING_RIGHTS_MASK: [u8; 64] = {
    let mut m = [0xFFu8; 64];
    m[0]  = 0xFD; // a1 → clear WHITE_QUEENSIDE
    m[4]  = 0xFC; // e1 → clear both white
    m[7]  = 0xFE; // h1 → clear WHITE_KINGSIDE
    m[56] = 0xF7; // a8 → clear BLACK_QUEENSIDE
    m[60] = 0xF3; // e8 → clear both black
    m[63] = 0xFB; // h8 → clear BLACK_KINGSIDE
    m
};

// ── Irreversible state snapshot (needed for unmake_move) ──────────────────────

#[derive(Clone, Copy, Debug)]
pub struct IrreversibleState {
    pub captured:        Option<Piece>,
    pub en_passant:      Option<Square>,
    pub castling_rights: CastlingRights,
    pub half_move_clock: u32,
    pub hash:            u64,
}

impl Board {
    // ── Make / Unmake ─────────────────────────────────────────────────────────

    /// Apply `mv` to the board and return the state needed to undo it.
    pub fn make_move(&mut self, mv: Move) -> IrreversibleState {
        let from = mv.from_sq();
        let to   = mv.to_sq();
        let us   = self.side_to_move;

        let state = IrreversibleState {
            captured:        self.piece_at(to),
            en_passant:      self.en_passant,
            castling_rights: self.castling_rights,
            half_move_clock: self.half_move_clock,
            hash:            self.hash,
        };

        let moving = self.piece_at(from)
            .expect("make_move: no piece on from-square");
        let is_pawn    = moving.piece_type == PieceType::Pawn;
        let is_capture = state.captured.is_some();

        self.half_move_clock = if is_pawn || is_capture { 0 }
                               else { self.half_move_clock + 1 };

        // Zobrist: remove old ep and castling contributions before the move.
        let keys = zobrist_keys();
        if let Some(sq) = state.en_passant {
            self.hash ^= keys.en_passant[sq.file() as usize];
        }
        self.hash ^= keys.castling[state.castling_rights.0 as usize];

        self.en_passant = None;

        match mv.flag() {
            MoveFlag::Normal => {
                if is_capture { self.remove_piece(to); }
                self.remove_piece(from);
                self.put_piece(moving, to);
                // Double pawn push → set en-passant target square
                if is_pawn {
                    let (fr, tr) = (from.rank(), to.rank());
                    if (us == Color::White && fr == 1 && tr == 3)
                    || (us == Color::Black && fr == 6 && tr == 4) {
                        let ep_rank = if us == Color::White { 2 } else { 5 };
                        self.en_passant = Some(Square::new(from.file(), ep_rank));
                    }
                }
            }
            MoveFlag::Promo => {
                if is_capture { self.remove_piece(to); }
                self.remove_piece(from);
                self.put_piece(Piece::new(us, mv.promo_piece_type()), to);
            }
            MoveFlag::EnPassant => {
                // Captured pawn is on the same rank as `from`, same file as `to`
                let cap_sq = Square::new(to.file(), from.rank());
                self.remove_piece(cap_sq);
                self.remove_piece(from);
                self.put_piece(moving, to);
            }
            MoveFlag::Castling => {
                let (rook_from, rook_to) = castling_rook_squares(to);
                self.remove_piece(from);
                self.remove_piece(rook_from);
                self.put_piece(moving, to);
                self.put_piece(Piece::new(us, PieceType::Rook), rook_to);
            }
        }

        // Update castling rights for any king/rook move or rook capture
        self.castling_rights.0 &= CASTLING_RIGHTS_MASK[from.0 as usize]
                                & CASTLING_RIGHTS_MASK[to.0 as usize];

        // Zobrist: add new ep, castling, and flip side-to-move.
        if let Some(sq) = self.en_passant {
            self.hash ^= keys.en_passant[sq.file() as usize];
        }
        self.hash ^= keys.castling[self.castling_rights.0 as usize];
        self.hash ^= keys.side;

        if us == Color::Black { self.full_move_number += 1; }
        self.side_to_move = us.flip();

        state
    }

    /// Undo a move previously applied with `make_move`.
    /// `mv` and `state` must be exactly those from the matching `make_move` call.
    pub fn unmake_move(&mut self, mv: Move, state: IrreversibleState) {
        let from = mv.from_sq();
        let to   = mv.to_sq();

        // Restore the side that made the move
        self.side_to_move = self.side_to_move.flip();
        let us = self.side_to_move;

        if us == Color::Black { self.full_move_number -= 1; }

        self.en_passant      = state.en_passant;
        self.castling_rights = state.castling_rights;
        self.half_move_clock = state.half_move_clock;

        match mv.flag() {
            MoveFlag::Normal => {
                let moved = self.piece_at(to).expect("unmake Normal: no piece at to");
                self.remove_piece(to);
                self.put_piece(moved, from);
                if let Some(cap) = state.captured { self.put_piece(cap, to); }
            }
            MoveFlag::Promo => {
                self.remove_piece(to);
                self.put_piece(Piece::new(us, PieceType::Pawn), from);
                if let Some(cap) = state.captured { self.put_piece(cap, to); }
            }
            MoveFlag::EnPassant => {
                self.remove_piece(to);
                self.put_piece(Piece::new(us, PieceType::Pawn), from);
                let cap_sq = Square::new(to.file(), from.rank());
                self.put_piece(Piece::new(us.flip(), PieceType::Pawn), cap_sq);
            }
            MoveFlag::Castling => {
                let (rook_from, rook_to) = castling_rook_squares(to);
                self.remove_piece(to);
                self.put_piece(Piece::new(us, PieceType::King), from);
                self.remove_piece(rook_to);
                self.put_piece(Piece::new(us, PieceType::Rook), rook_from);
            }
        }

        // Restore pre-move hash; the put/remove calls above XOR'd piece keys
        // that are not needed since we're restoring the saved snapshot.
        self.hash = state.hash;
    }

    // ── Zobrist ───────────────────────────────────────────────────────────────

    /// Recompute the Zobrist hash from scratch. Used to verify incremental
    /// updates in tests; not needed in the hot path.
    pub fn compute_hash(&self) -> u64 {
        let keys = zobrist_keys();
        let mut h = 0u64;
        for sq in 0u8..64 {
            if let Some(p) = self.mailbox[sq as usize] {
                h ^= keys.piece_sq[p.color as usize][p.piece_type as usize][sq as usize];
            }
        }
        if self.side_to_move == Color::Black { h ^= keys.side; }
        h ^= keys.castling[self.castling_rights.0 as usize];
        if let Some(sq) = self.en_passant {
            h ^= keys.en_passant[sq.file() as usize];
        }
        h
    }

    // ── Attack queries ────────────────────────────────────────────────────────

    /// Returns true if `sq` is attacked by any piece of `by_color`.
    pub fn is_attacked_by(&self, sq: Square, by: Color) -> bool {
        use super::attacks;
        let them = by as usize;
        let occ  = self.all_occupancy;

        // Pawns — reverse-attack trick: use the opposite color's attack pattern
        if attacks::pawn_attacks(by.flip(), sq)
            & self.pieces[them][PieceType::Pawn as usize] != 0
        { return true; }

        if attacks::knight_attacks(sq)
            & self.pieces[them][PieceType::Knight as usize] != 0
        { return true; }

        if attacks::king_attacks(sq)
            & self.pieces[them][PieceType::King as usize] != 0
        { return true; }

        let diag = attacks::bishop_attacks(sq, occ);
        if diag & (self.pieces[them][PieceType::Bishop as usize]
                 | self.pieces[them][PieceType::Queen  as usize]) != 0
        { return true; }

        let ortho = attacks::rook_attacks(sq, occ);
        if ortho & (self.pieces[them][PieceType::Rook  as usize]
                  | self.pieces[them][PieceType::Queen as usize]) != 0
        { return true; }

        false
    }

    /// Returns true if the side to move's king is currently in check.
    #[inline]
    pub fn is_in_check(&self) -> bool {
        let king_bb = self.piece_bb(self.side_to_move, PieceType::King);
        if king_bb == EMPTY { return false; }
        self.is_attacked_by(lsb(king_bb), self.side_to_move.flip())
    }
}

/// Returns (rook_from, rook_to) given the king's castling destination square.
fn castling_rook_squares(king_to: Square) -> (Square, Square) {
    match king_to.0 {
        6  => (Square(7),  Square(5)),  // White kingside:  H1 → F1
        2  => (Square(0),  Square(3)),  // White queenside: A1 → D1
        62 => (Square(63), Square(61)), // Black kingside:  H8 → F8
        58 => (Square(56), Square(59)), // Black queenside: A8 → D8
        _  => panic!("castling_rook_squares: invalid king destination {}", king_to.0),
    }
}

// ── Terminal display ──────────────────────────────────────────────────────────

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  +---+---+---+---+---+---+---+---+")?;
        for rank in (0..8u8).rev() {
            write!(f, "{} |", rank + 1)?;
            for file in 0..8u8 {
                let sym = match self.piece_at(Square::new(file, rank)) {
                    Some(p) => p.to_char(),
                    None    => '.',
                };
                write!(f, " {sym} |")?;
            }
            writeln!(f)?;
            writeln!(f, "  +---+---+---+---+---+---+---+---+")?;
        }
        writeln!(f, "    a   b   c   d   e   f   g   h")?;
        writeln!(f)?;
        writeln!(f, "  Side to move : {}", self.side_to_move)?;
        writeln!(f, "  Castling     : {}", self.castling_rights)?;
        write!(f,   "  En passant   : ")?;
        match self.en_passant {
            None     => write!(f, "-")?,
            Some(sq) => write!(f, "{sq}")?,
        }
        write!(f, "\n  Clocks       : half={} full={}",
               self.half_move_clock, self.full_move_number)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::squares::*;
    use super::super::moves::Move;

    #[test]
    fn starting_fen_round_trip() {
        let board = Board::starting_position();
        assert_eq!(board.to_fen(), STARTING_FEN);
    }

    #[test]
    fn piece_counts_starting_position() {
        let b = Board::starting_position();
        assert_eq!(popcount(b.piece_bb(Color::White, PieceType::Pawn)),   8);
        assert_eq!(popcount(b.piece_bb(Color::Black, PieceType::Pawn)),   8);
        assert_eq!(popcount(b.piece_bb(Color::White, PieceType::King)),   1);
        assert_eq!(popcount(b.piece_bb(Color::Black, PieceType::Queen)),  1);
        assert_eq!(popcount(b.piece_bb(Color::White, PieceType::Rook)),   2);
        assert_eq!(popcount(b.piece_bb(Color::Black, PieceType::Knight)), 2);
        assert_eq!(popcount(b.all_occupancy), 32);
    }

    #[test]
    fn mailbox_starting_position() {
        let b = Board::starting_position();
        assert_eq!(b.piece_at(Square::new(4, 0)), // e1
            Some(Piece::new(Color::White, PieceType::King)));
        assert_eq!(b.piece_at(Square::new(3, 0)), // d1
            Some(Piece::new(Color::White, PieceType::Queen)));
        assert_eq!(b.piece_at(Square::new(4, 7)), // e8
            Some(Piece::new(Color::Black, PieceType::King)));
        assert_eq!(b.piece_at(Square::new(4, 3)), None); // e4 empty
    }

    #[test]
    fn occupancy_consistency() {
        let b = Board::starting_position();
        for c in [Color::White, Color::Black] {
            let expected = PieceType::ALL
                .iter()
                .fold(EMPTY, |acc, &pt| acc | b.piece_bb(c, pt));
            assert_eq!(b.occupancy[c as usize], expected,
                "occupancy mismatch for {c}");
        }
        assert_eq!(b.all_occupancy, b.occupancy[0] | b.occupancy[1]);
    }

    #[test]
    fn various_fen_round_trips() {
        let positions = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            "r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/2N2N2/PPP2PPP/R1BQKB1R b KQkq d3 0 4",
            "8/8/8/8/8/8/8/4K3 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ];
        for fen in &positions {
            let board = Board::from_fen(fen)
                .unwrap_or_else(|e| panic!("parse failed for {fen}: {e}"));
            assert_eq!(&board.to_fen(), fen, "round-trip failed for {fen}");
        }
    }

    #[test]
    fn invalid_fen_errors() {
        assert!(Board::from_fen("").is_err());
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -").is_err());
        assert!(Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1"
        ).is_err());
    }

    #[test]
    fn hash_matches_recomputed_from_fen() {
        for fen in &[
            STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "8/8/8/8/8/8/3K4/R6R w - - 0 1",
        ] {
            let b = Board::from_fen(fen).unwrap();
            assert_eq!(b.hash, b.compute_hash(), "hash mismatch for {fen}");
        }
    }

    #[test]
    fn hash_make_unmake_roundtrip() {
        let mut b = Board::starting_position();
        let h0 = b.hash;
        // e2e4 — normal double pawn push (sets en-passant square)
        let mv = Move::normal(E2, E4);
        let state = b.make_move(mv);
        assert_ne!(b.hash, h0, "hash must change after make_move");
        assert_eq!(b.hash, b.compute_hash(), "incremental hash diverged after make_move");
        b.unmake_move(mv, state);
        assert_eq!(b.hash, h0, "hash not restored after unmake_move");
    }

    #[test]
    fn hash_same_position_different_move_order() {
        // Four Knights: 1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6
        // Two independent knight manoeuvres played in opposite orders:
        //   White: Nf3-h4 and Nc3-d5   |  Black: Nc6-d4 and Nf6-h5
        // No pawn moves → no en-passant; no king/rook moves → castling rights
        // unchanged. Both sequences must reach exactly the same position.

        let fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4";

        let mut b1 = Board::from_fen(fen).unwrap();
        b1.make_move(Move::normal(F3, H4));
        b1.make_move(Move::normal(C6, D4));
        b1.make_move(Move::normal(C3, D5));
        b1.make_move(Move::normal(F6, H5));

        let mut b2 = Board::from_fen(fen).unwrap();
        b2.make_move(Move::normal(C3, D5));
        b2.make_move(Move::normal(F6, H5));
        b2.make_move(Move::normal(F3, H4));
        b2.make_move(Move::normal(C6, D4));

        assert_eq!(b1.to_fen(), b2.to_fen(), "positions must be identical");
        assert_eq!(b1.hash,     b2.hash,     "Zobrist hash must match for identical positions");
    }

    #[test]
    fn hash_different_positions_differ() {
        let b1 = Board::from_fen(STARTING_FEN).unwrap();
        let b2 = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        ).unwrap();
        assert_ne!(b1.hash, b2.hash);
    }

    #[test]
    fn put_remove_piece_sync() {
        let mut b = Board::empty();
        let sq = Square::new(4, 4); // e5
        let piece = Piece::new(Color::White, PieceType::Rook);

        b.put_piece(piece, sq);
        assert_eq!(b.piece_at(sq), Some(piece));
        assert!(test_bit(b.piece_bb(Color::White, PieceType::Rook), sq));
        assert!(test_bit(b.occupancy[Color::White as usize], sq));
        assert!(test_bit(b.all_occupancy, sq));

        b.remove_piece(sq);
        assert_eq!(b.piece_at(sq), None);
        assert!(!test_bit(b.piece_bb(Color::White, PieceType::Rook), sq));
        assert!(!test_bit(b.all_occupancy, sq));
    }
}
