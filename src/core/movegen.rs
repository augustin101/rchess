use super::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks};
use super::bitboard::*;
use super::board::Board;
use super::moves::{Move, MoveList, PromoKind};
use super::types::{CastlingRights, Color, PieceType, Square};

// ── Public API ────────────────────────────────────────────────────────────────

/// All pseudo-legal moves for the side to move.
/// Pseudo-legal = geometrically correct but may leave own king in check.
pub fn generate_pseudo_legal(board: &Board) -> MoveList {
    let us = board.side_to_move;
    let mut list = MoveList::new();
    gen_pawn_moves(board, us, &mut list);
    gen_piece_moves::<{ PieceType::Knight as usize }>(board, us, &mut list);
    gen_piece_moves::<{ PieceType::Bishop as usize }>(board, us, &mut list);
    gen_piece_moves::<{ PieceType::Rook   as usize }>(board, us, &mut list);
    gen_piece_moves::<{ PieceType::Queen  as usize }>(board, us, &mut list);
    gen_king_moves(board, us, &mut list);
    gen_castling(board, us, &mut list);
    list
}

/// All strictly legal moves: pseudo-legal filtered by make/unmake + check test.
pub fn generate_legal(board: &Board) -> MoveList {
    let pseudo = generate_pseudo_legal(board);
    let us     = board.side_to_move;
    let mut legal = MoveList::new();
    let mut b = board.clone();

    for &mv in &pseudo {
        let state = b.make_move(mv);
        // After make_move, side_to_move is flipped; the king we check is `us`.
        let king_bb = b.piece_bb(us, PieceType::King);
        let legal_move = king_bb != EMPTY
            && !b.is_attacked_by(lsb(king_bb), b.side_to_move);
        b.unmake_move(mv, state);
        if legal_move { legal.push(mv); }
    }

    legal
}

// ── Pawn moves ────────────────────────────────────────────────────────────────

fn gen_pawn_moves(board: &Board, us: Color, list: &mut MoveList) {
    let pawns     = board.piece_bb(us, PieceType::Pawn);
    let their_occ = board.occupancy[us.flip() as usize];
    let empty     = !board.all_occupancy;

    let promo_rank = match us {
        Color::White => RANK_8,
        Color::Black => RANK_1,
    };

    // ── Single pushes ─────────────────────────────────────────────────────────
    let step1 = if us == Color::White { pawns << 8 } else { pawns >> 8 } & empty;

    for to in iter_bits(step1 & !promo_rank) {
        let from = if us == Color::White { Square(to.0 - 8) } else { Square(to.0 + 8) };
        list.push(Move::normal(from, to));
    }
    for to in iter_bits(step1 & promo_rank) {
        let from = if us == Color::White { Square(to.0 - 8) } else { Square(to.0 + 8) };
        push_all_promos(from, to, list);
    }

    // ── Double pushes ─────────────────────────────────────────────────────────
    // Only from pawns whose single-push target landed on the third rank,
    // meaning they started on the second rank (the starting rank).
    let step2 = if us == Color::White {
        ((step1 & RANK_3) << 8) & empty
    } else {
        ((step1 & RANK_6) >> 8) & empty
    };
    for to in iter_bits(step2) {
        let from = if us == Color::White { Square(to.0 - 16) } else { Square(to.0 + 16) };
        list.push(Move::normal(from, to));
    }

    // ── Captures ──────────────────────────────────────────────────────────────
    let (east_cap, west_cap) = if us == Color::White {
        (((pawns & !FILE_H) << 9) & their_occ,
         ((pawns & !FILE_A) << 7) & their_occ)
    } else {
        (((pawns & !FILE_H) >> 7) & their_occ,
         ((pawns & !FILE_A) >> 9) & their_occ)
    };

    for to in iter_bits(east_cap) {
        let from = if us == Color::White { Square(to.0 - 9) } else { Square(to.0 + 7) };
        if square_bb(to) & promo_rank != 0 { push_all_promos(from, to, list); }
        else { list.push(Move::normal(from, to)); }
    }
    for to in iter_bits(west_cap) {
        let from = if us == Color::White { Square(to.0 - 7) } else { Square(to.0 + 9) };
        if square_bb(to) & promo_rank != 0 { push_all_promos(from, to, list); }
        else { list.push(Move::normal(from, to)); }
    }

    // ── En passant ────────────────────────────────────────────────────────────
    if let Some(ep_sq) = board.en_passant {
        // Reverse-attack: which of our pawns can reach ep_sq?
        let attackers = pawn_attacks(us.flip(), ep_sq) & pawns;
        for from in iter_bits(attackers) {
            list.push(Move::en_passant(from, ep_sq));
        }
    }
}

#[inline]
fn push_all_promos(from: Square, to: Square, list: &mut MoveList) {
    list.push(Move::promo(from, to, PromoKind::Queen));
    list.push(Move::promo(from, to, PromoKind::Rook));
    list.push(Move::promo(from, to, PromoKind::Bishop));
    list.push(Move::promo(from, to, PromoKind::Knight));
}

// ── Sliding and leaping pieces ────────────────────────────────────────────────

fn gen_piece_moves<const PT: usize>(board: &Board, us: Color, list: &mut MoveList) {
    let own_occ = board.occupancy[us as usize];
    let occ     = board.all_occupancy;
    let mut pieces = board.pieces[us as usize][PT];

    while pieces != EMPTY {
        let from = pop_lsb(&mut pieces);
        let targets = piece_attacks(PT, from, occ) & !own_occ;
        for to in iter_bits(targets) {
            list.push(Move::normal(from, to));
        }
    }
}

#[inline]
fn piece_attacks(pt: usize, sq: Square, occ: Bitboard) -> Bitboard {
    match pt {
        1 => knight_attacks(sq),
        2 => bishop_attacks(sq, occ),
        3 => rook_attacks(sq, occ),
        4 => queen_attacks(sq, occ),
        _ => 0,
    }
}

fn gen_king_moves(board: &Board, us: Color, list: &mut MoveList) {
    let own_occ = board.occupancy[us as usize];
    let king_bb = board.piece_bb(us, PieceType::King);
    if king_bb == EMPTY { return; }
    let from = lsb(king_bb);
    for to in iter_bits(king_attacks(from) & !own_occ) {
        list.push(Move::normal(from, to));
    }
}

// ── Castling ──────────────────────────────────────────────────────────────────
// We pre-verify that the king is not in check on any transit square, so these
// moves are already fully legal (the generate_legal filter still runs them, but
// they will always pass).

fn gen_castling(board: &Board, us: Color, list: &mut MoveList) {
    let cr  = board.castling_rights;
    let occ = board.all_occupancy;
    let them = us.flip();

    match us {
        Color::White => {
            // Kingside: e1(4) → g1(6); f1(5) and g1(6) must be empty
            if cr.has(CastlingRights::WHITE_KINGSIDE)
                && occ & 0x0000_0000_0000_0060 == 0
                && !board.is_attacked_by(Square(4), them)
                && !board.is_attacked_by(Square(5), them)
                && !board.is_attacked_by(Square(6), them)
            {
                list.push(Move::castling(Square(4), Square(6)));
            }
            // Queenside: e1(4) → c1(2); b1(1)/c1(2)/d1(3) must be empty
            if cr.has(CastlingRights::WHITE_QUEENSIDE)
                && occ & 0x0000_0000_0000_000E == 0
                && !board.is_attacked_by(Square(4), them)
                && !board.is_attacked_by(Square(3), them)
                && !board.is_attacked_by(Square(2), them)
            {
                list.push(Move::castling(Square(4), Square(2)));
            }
        }
        Color::Black => {
            // Kingside: e8(60) → g8(62); f8(61) and g8(62) must be empty
            if cr.has(CastlingRights::BLACK_KINGSIDE)
                && occ & 0x6000_0000_0000_0000 == 0
                && !board.is_attacked_by(Square(60), them)
                && !board.is_attacked_by(Square(61), them)
                && !board.is_attacked_by(Square(62), them)
            {
                list.push(Move::castling(Square(60), Square(62)));
            }
            // Queenside: e8(60) → c8(58); b8(57)/c8(58)/d8(59) must be empty
            if cr.has(CastlingRights::BLACK_QUEENSIDE)
                && occ & 0x0E00_0000_0000_0000 == 0
                && !board.is_attacked_by(Square(60), them)
                && !board.is_attacked_by(Square(59), them)
                && !board.is_attacked_by(Square(58), them)
            {
                list.push(Move::castling(Square(60), Square(58)));
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).expect("valid FEN")
    }

    #[test]
    fn start_has_20_legal_moves() {
        let b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(generate_legal(&b).len(), 20);
    }

    #[test]
    fn kiwipete_has_48_legal_moves() {
        let b = board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        assert_eq!(generate_legal(&b).len(), 48);
    }

    #[test]
    fn pos3_d1_has_14_moves() {
        let b = board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        assert_eq!(generate_legal(&b).len(), 14);
    }

    #[test]
    fn make_unmake_restores_fen() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut b = board(fen);
        let moves = generate_legal(&b);
        for &mv in &moves {
            let state = b.make_move(mv);
            b.unmake_move(mv, state);
            assert_eq!(b.to_fen(), fen, "make/unmake broke FEN for move {mv}");
        }
    }

    #[test]
    fn no_legal_moves_in_stalemate() {
        // Black king a8, white queen c7, white king b6 — black to move: stalemate
        // a7 attacked by queen (rank), b8 by queen (diagonal), b7 by queen+king
        let b = board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
        assert_eq!(generate_legal(&b).len(), 0);
    }

    #[test]
    fn promotion_count() {
        // White pawn on a7, black king far away, white king on h1
        let b = board("8/P7/8/8/8/8/8/7K w - - 0 1");
        let moves = generate_legal(&b);
        // 1 pawn → 4 promos; king has some moves
        let promos = moves.as_slice().iter().filter(|m| m.is_promo()).count();
        assert_eq!(promos, 4);
    }

    #[test]
    fn en_passant_move_generated() {
        // White pawn on e5, black just played d7-d5 (en passant on d6)
        let b = board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
        let moves = generate_legal(&b);
        let ep = moves.as_slice().iter().filter(|m| m.is_en_passant()).count();
        assert_eq!(ep, 1);
    }
}
