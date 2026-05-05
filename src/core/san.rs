use super::board::Board;
use super::movegen::generate_legal;
use super::moves::{Move, PromoKind};
use super::types::PieceType;

/// Convert a move to Standard Algebraic Notation.
/// `board` must be the position **before** the move is applied.
pub fn move_to_san(board: &Board, mv: Move) -> String {
    let from = mv.from_sq();
    let to   = mv.to_sq();

    // ── Castling ──────────────────────────────────────────────────────────────
    if mv.is_castling() {
        let base = if to.file() > from.file() { "O-O" } else { "O-O-O" };
        let mut b2 = board.clone();
        b2.make_move(mv);
        return format!("{}{}", base, check_suffix(&b2));
    }

    let piece      = board.piece_at(from).expect("san: no piece on from-square");
    let is_capture = board.piece_at(to).is_some() || mv.is_en_passant();
    let mut san    = String::new();

    match piece.piece_type {
        // ── Pawn ──────────────────────────────────────────────────────────────
        PieceType::Pawn => {
            if is_capture {
                san.push((b'a' + from.file()) as char);
                san.push('x');
            }
            san.push((b'a' + to.file()) as char);
            san.push((b'1' + to.rank()) as char);
            if mv.is_promo() {
                san.push('=');
                san.push(promo_char(mv.promo_kind()));
            }
        }

        // ── Pieces ────────────────────────────────────────────────────────────
        pt => {
            san.push(piece_char(pt));

            // Disambiguation: other pieces of the same type that can also
            // reach `to` legally.
            let legal = generate_legal(board);
            let ambiguous: Vec<Move> = legal.as_slice().iter()
                .filter(|&&m| {
                    m != mv
                    && !m.is_castling()
                    && m.to_sq() == to
                    && board.piece_at(m.from_sq()).map(|p| p.piece_type) == Some(pt)
                })
                .copied()
                .collect();

            if !ambiguous.is_empty() {
                let same_file = ambiguous.iter().any(|m| m.from_sq().file() == from.file());
                let same_rank = ambiguous.iter().any(|m| m.from_sq().rank() == from.rank());

                if !same_file {
                    san.push((b'a' + from.file()) as char);
                } else if !same_rank {
                    san.push((b'1' + from.rank()) as char);
                } else {
                    san.push((b'a' + from.file()) as char);
                    san.push((b'1' + from.rank()) as char);
                }
            }

            if is_capture { san.push('x'); }
            san.push((b'a' + to.file()) as char);
            san.push((b'1' + to.rank()) as char);
        }
    }

    // ── Check / checkmate suffix ──────────────────────────────────────────────
    let mut b2 = board.clone();
    b2.make_move(mv);
    san.push_str(check_suffix(&b2));

    san
}

fn check_suffix(board: &Board) -> &'static str {
    if !board.is_in_check() { return ""; }
    if generate_legal(board).is_empty() { "#" } else { "+" }
}

fn piece_char(pt: PieceType) -> char {
    match pt {
        PieceType::Knight => 'N',
        PieceType::Bishop => 'B',
        PieceType::Rook   => 'R',
        PieceType::Queen  => 'Q',
        PieceType::King   => 'K',
        PieceType::Pawn   => unreachable!("pawn handled separately"),
    }
}

fn promo_char(pk: PromoKind) -> char {
    match pk {
        PromoKind::Knight => 'N',
        PromoKind::Bishop => 'B',
        PromoKind::Rook   => 'R',
        PromoKind::Queen  => 'Q',
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::board::Board;
    use super::super::movegen::generate_legal;

    fn board(fen: &str) -> Board { Board::from_fen(fen).unwrap() }

    fn san(fen: &str, uci: &str) -> String {
        let b = board(fen);
        let mv = generate_legal(&b).as_slice().iter()
            .find(|m| m.to_string() == uci)
            .copied()
            .expect("move not found");
        move_to_san(&b, mv)
    }

    #[test]
    fn pawn_push()       { assert_eq!(san("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"), "e4"); }

    #[test]
    fn knight_move()     { assert_eq!(san("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3"), "Nf3"); }

    #[test]
    fn pawn_capture()    { assert_eq!(san("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e4d5"), "exd5"); }

    #[test]
    fn castling_kingside() {
        assert_eq!(san("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"), "O-O");
    }

    #[test]
    fn castling_queenside() {
        assert_eq!(san("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"), "O-O-O");
    }

    #[test]
    fn promotion() {
        assert_eq!(san("8/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q"), "a8=Q");
    }

    #[test]
    fn disambiguation_file() {
        // Two rooks on a1 and h1, both can go to d1 → disambiguate by file
        let b = board("8/8/8/8/8/8/3K4/R6R w - - 0 1");
        let mv = generate_legal(&b).as_slice().iter()
            .find(|m| m.to_string() == "a1d1").copied().unwrap();
        let s = move_to_san(&b, mv);
        assert_eq!(s, "Rad1");
    }

    #[test]
    fn check_suffix() {
        // Scholar's mate setup — Qh5 gives check
        let b = board("rnbqkb1r/pppp1ppp/8/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 3");
        let mv = generate_legal(&b).as_slice().iter()
            .find(|m| m.to_string() == "h5f7").copied().unwrap();
        assert!(move_to_san(&b, mv).ends_with('#'));
    }
}
