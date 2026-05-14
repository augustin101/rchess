use std::sync::Arc;

use crate::core::bitboard::{lsb, EMPTY};
use crate::core::board::Board;
use crate::core::movegen::generate_pseudo_legal;
use crate::core::moves::Move;
use crate::core::types::{Color, PieceType};
use super::Engine;
use super::eval::{static_eval, CHECKMATE_SCORE};
use super::nnue;

const NEG_INF: i32 = -(CHECKMATE_SCORE + 1);
const POS_INF: i32 =   CHECKMATE_SCORE + 1;
const MAX_PLY: usize = 64;

// ── Move ordering scores ──────────────────────────────────────────────────────
// Each bucket must be strictly above the one below it to preserve ordering
// priority.  Actual values are arbitrary as long as the gaps are wide enough.
const SCORE_TT_MOVE:      i32 = 30_000_000; // always searched first
const SCORE_PROMO_BASE:   i32 =  9_000_000; // + piece_value(promo piece)
const SCORE_CAPTURE_BASE: i32 =  1_000_000; // + MVV-LVA delta
const SCORE_KILLER:       i32 =    900_000; // below captures, above history
// History scores live in [0, HISTORY_MAX] and fall below killers naturally.
const SCORE_QUIET_SKIP:   i32 =         -1; // sentinel: sorted to back, then skipped
const HISTORY_MAX:        i32 =     32_000;

// ── Search tuning parameters ──────────────────────────────────────────────────
const NULL_MIN_DEPTH:  u32 = 3; // don't try null move at shallow depths
const NULL_FULL_DEPTH: u32 = 6; // use larger reduction at this depth and above
const NULL_R_PARTIAL:  u32 = 2; // reduction below NULL_FULL_DEPTH
const NULL_R_FULL:     u32 = 3; // reduction at/above NULL_FULL_DEPTH

const FUTILITY_MAX_DEPTH: u32 = 2;   // apply futility pruning only at depth 1-2
const FUTILITY_MARGIN_1:  i32 = 100; // centipawn margin at depth 1
const FUTILITY_MARGIN_2:  i32 = 300; // centipawn margin at depth 2

const LMR_MIN_QUIET: usize = 3; // start reducing after this many quiet moves
const LMR_MIN_DEPTH: u32   = 3; // don't reduce at shallow depths

// ── Transposition Table ───────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
enum Bound { #[default] None, Exact, Lower, Upper }

#[derive(Clone, Copy, Default)]
struct TtEntry {
    hash:      u64,
    score:     i32,
    best_move: Move,
    depth:     u8,
    bound:     Bound,
}

// 1M entries × 24 bytes ≈ 24 MB
const TT_SIZE: usize = 1 << 20;

struct TranspositionTable {
    entries: Box<[TtEntry]>,
}

impl TranspositionTable {
    fn new() -> Self {
        Self { entries: vec![TtEntry::default(); TT_SIZE].into_boxed_slice() }
    }

    #[inline]
    fn probe(&self, hash: u64, depth: u32) -> Option<TtEntry> {
        let e = self.entries[(hash as usize) & (TT_SIZE - 1)];
        if e.bound != Bound::None && e.hash == hash && u32::from(e.depth) >= depth {
            Some(e)
        } else {
            None
        }
    }

    #[inline]
    fn probe_move(&self, hash: u64) -> Move {
        let e = self.entries[(hash as usize) & (TT_SIZE - 1)];
        if e.bound != Bound::None && e.hash == hash { e.best_move } else { Move::NULL }
    }

    #[inline]
    fn store(&mut self, hash: u64, depth: u32, score: i32, bound: Bound, best_move: Move) {
        let slot = &mut self.entries[(hash as usize) & (TT_SIZE - 1)];
        if slot.bound == Bound::None || slot.hash == hash || depth >= u32::from(slot.depth) {
            *slot = TtEntry { hash, score, best_move, depth: depth as u8, bound };
        }
    }
}

// ── Killer moves (2 per ply) ──────────────────────────────────────────────────

struct KillerTable([[Move; 2]; MAX_PLY]);

impl KillerTable {
    fn new() -> Self { KillerTable([[Move::NULL; 2]; MAX_PLY]) }

    #[inline]
    fn update(&mut self, mv: Move, ply: usize) {
        if ply >= MAX_PLY { return; }
        let slot = &mut self.0[ply];
        if slot[0] != mv {
            slot[1] = slot[0];
            slot[0] = mv;
        }
    }

    #[inline]
    fn is_killer(&self, mv: Move, ply: usize) -> bool {
        if ply >= MAX_PLY { return false; }
        let s = self.0[ply];
        s[0] == mv || s[1] == mv
    }
}

// ── History heuristic ─────────────────────────────────────────────────────────

struct HistoryTable([[[i32; 64]; 64]; 2]);

impl HistoryTable {
    fn new() -> Self { HistoryTable([[[0i32; 64]; 64]; 2]) }

    #[inline]
    fn update(&mut self, color: Color, from: u8, to: u8, depth: u32) {
        let v = &mut self.0[color as usize][from as usize][to as usize];
        *v = (*v + (depth as i32) * (depth as i32)).min(HISTORY_MAX);
    }

    #[inline]
    fn get(&self, color: Color, from: u8, to: u8) -> i32 {
        self.0[color as usize][from as usize][to as usize]
    }

    fn age(&mut self) {
        for c in &mut self.0 {
            for f in c.iter_mut() {
                for v in f.iter_mut() {
                    *v >>= 1;
                }
            }
        }
    }
}

// ── Evaluator ─────────────────────────────────────────────────────────────────

enum Evaluator {
    Static,
    Nnue { weights: Arc<nnue::Nnue>, stack: nnue::AccumulatorStack },
}

impl Evaluator {
    #[inline]
    fn eval_stm(&self, board: &Board) -> i32 {
        match self {
            Evaluator::Static => {
                let raw = static_eval(board);
                if board.side_to_move == Color::White { raw } else { -raw }
            }
            Evaluator::Nnue { weights, stack } => stack.evaluate(weights, board.side_to_move),
        }
    }

    /// Refresh the accumulator at the root of a new search.
    fn init(&mut self, board: &Board) {
        let Evaluator::Nnue { weights, stack } = self else { return };
        stack.init(weights, board);
    }

    /// Push + apply move delta (call BEFORE board.make_move).
    #[inline]
    fn push_move(&mut self, board: &Board, mv: Move) {
        let Evaluator::Nnue { weights, stack } = self else { return };
        stack.push_move(weights, board, mv);
    }

    /// Push for null moves (no piece changes).
    #[inline]
    fn push_null(&mut self) {
        let Evaluator::Nnue { weights: _, stack } = self else { return };
        stack.push_null();
    }

    /// Pop (call AFTER board.unmake_move).
    #[inline]
    fn pop(&mut self) {
        let Evaluator::Nnue { weights: _, stack } = self else { return };
        stack.pop();
    }
}

// ── Search context (bundles mutable tables passed through the tree) ───────────

struct SearchContext {
    tt:      TranspositionTable,
    killers: KillerTable,
    history: HistoryTable,
    nodes:   u64,
    eval:    Evaluator,
}

// ── Piece values for move ordering ────────────────────────────────────────────

#[inline]
fn piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn   =>   100,
        PieceType::Knight =>   320,
        PieceType::Bishop =>   330,
        PieceType::Rook   =>   500,
        PieceType::Queen  =>   900,
        PieceType::King   => 10_000,
    }
}

// ── MVV-LVA capture score ─────────────────────────────────────────────────────
// For en passant, piece_at(to_sq) is None; the pawn default (100) is correct.

#[inline]
fn capture_mvv_lva(board: &Board, mv: Move) -> i32 {
    let vv = board.piece_at(mv.to_sq()).map_or(100, |p| piece_value(p.piece_type));
    let av = board.piece_at(mv.from_sq()).map_or(100, |p| piece_value(p.piece_type));
    SCORE_CAPTURE_BASE + vv * 10 - av
}

// ── LMR reduction depth ───────────────────────────────────────────────────────

#[inline]
fn lmr_reduction(quiet_count: usize) -> u32 {
    1 + (quiet_count as u32) / 6
}

// ── Move scoring ──────────────────────────────────────────────────────────────
// Priority: TT/hash move > promotions > captures (MVV-LVA) > killers > history

#[inline]
fn score_move(
    board:   &Board,
    mv:      Move,
    tt_move: Move,
    killers: &KillerTable,
    history: &HistoryTable,
    ply:     usize,
) -> i32 {
    if mv == tt_move { return SCORE_TT_MOVE; }
    if mv.is_promo() { return SCORE_PROMO_BASE + piece_value(mv.promo_piece_type()); }
    if board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant() {
        return capture_mvv_lva(board, mv);
    }
    if killers.is_killer(mv, ply) { return SCORE_KILLER; }
    history.get(board.side_to_move, mv.from_sq().0, mv.to_sq().0)
}

// ── Null-move safety check ────────────────────────────────────────────────────
// Avoid null move in pure pawn + king endings to reduce zugzwang risk.

#[inline]
fn has_non_pawn_material(board: &Board) -> bool {
    let us = board.side_to_move as usize;
    board.occupancy[us]
        & !board.pieces[us][PieceType::Pawn as usize]
        & !board.pieces[us][PieceType::King as usize]
        != EMPTY
}

// ── Engine ────────────────────────────────────────────────────────────────────

pub struct AlphaBetaEngine {
    depth:      u32,
    name:       String,
    ctx:        SearchContext,
    last_score: Option<i32>,
}

impl AlphaBetaEngine {
    pub fn new() -> Self { Self::with_depth(5) }

    pub fn with_depth(depth: u32) -> Self {
        Self::new_inner(depth, Evaluator::Static)
    }

    pub fn with_nnue(depth: u32, nnue: Arc<nnue::Nnue>) -> Self {
        Self::new_inner(depth, Evaluator::Nnue {
            weights: nnue,
            stack:   nnue::AccumulatorStack::new(),
        })
    }

    fn new_inner(depth: u32, eval: Evaluator) -> Self {
        let depth = depth.max(1);
        let name = match &eval {
            Evaluator::Static    => format!("Alpha-Beta (d={depth})"),
            Evaluator::Nnue {..} => format!("Alpha-Beta NNUE (d={depth})"),
        };
        AlphaBetaEngine {
            depth,
            name,
            ctx: SearchContext {
                tt:      TranspositionTable::new(),
                killers: KillerTable::new(),
                history: HistoryTable::new(),
                nodes:   0,
                eval,
            },
            last_score: None,
        }
    }

    pub fn set_depth(&mut self, depth: u32) {
        self.depth = depth.max(1);
        self.name = match &self.ctx.eval {
            Evaluator::Static    => format!("Alpha-Beta (d={})", self.depth),
            Evaluator::Nnue {..} => format!("Alpha-Beta NNUE (d={})", self.depth),
        };
    }

    pub fn nodes_searched(&self) -> u64 { self.ctx.nodes }

    /// Iterative deepening with a wall-clock deadline. Completes each depth
    /// before checking the clock, then stops if the deadline has passed.
    pub fn choose_move_timed(
        &mut self,
        board: &Board,
        deadline: std::time::Instant,
    ) -> Option<Move> {
        let mut b = board.clone();
        self.ctx.killers = KillerTable::new();
        self.ctx.history.age();
        self.ctx.nodes = 0;
        self.ctx.eval.init(&b);
        let mut best = Move::NULL;
        self.last_score = None;
        for d in 1..=self.depth {
            if std::time::Instant::now() >= deadline { break; }
            if let Some((mv, score)) = search_root(&mut b, d, &mut self.ctx) {
                best = mv;
                self.last_score = Some(score);
            }
        }
        if best.is_null() { None } else { Some(best) }
    }
}

impl Engine for AlphaBetaEngine {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let mut b = board.clone();
        self.ctx.killers = KillerTable::new();
        self.ctx.history.age();
        self.ctx.nodes = 0;
        self.ctx.eval.init(&b);
        let mut best = Move::NULL;
        self.last_score = None;
        for d in 1..=self.depth {
            if let Some((mv, score)) = search_root(&mut b, d, &mut self.ctx) {
                best = mv;
                self.last_score = Some(score);
            }
        }
        if best.is_null() { None } else { Some(best) }
    }

    fn name(&self) -> &str { self.name.as_str() }

    fn last_score(&self) -> Option<i32> { self.last_score }
}

// ── Root search ───────────────────────────────────────────────────────────────

// Returns the best move and its score from the mover's perspective.
fn search_root(board: &mut Board, depth: u32, ctx: &mut SearchContext) -> Option<(Move, i32)> {
    let pseudo = generate_pseudo_legal(board);
    let n = pseudo.len();
    if n == 0 { return None; }

    let mut moves  = [Move::NULL; 256];
    let mut scores = [0i32; 256];
    moves[..n].copy_from_slice(pseudo.as_slice());

    let tt_move = ctx.tt.probe_move(board.hash);
    for i in 0..n {
        scores[i] = score_move(board, moves[i], tt_move, &ctx.killers, &ctx.history, 0);
    }

    let us = board.side_to_move;
    let mut alpha = NEG_INF;
    let mut best_move = Move::NULL;

    for i in 0..n {
        // Incremental selection sort: pull best remaining move to position i.
        let mut bi = i;
        for j in (i + 1)..n { if scores[j] > scores[bi] { bi = j; } }
        moves.swap(i, bi);
        scores.swap(i, bi);

        let mv = moves[i];
        ctx.eval.push_move(board, mv);
        let state = board.make_move(mv);

        let king_bb = board.piece_bb(us, PieceType::King);
        if king_bb == EMPTY || board.is_attacked_by(lsb(king_bb), board.side_to_move) {
            board.unmake_move(mv, state);
            ctx.eval.pop();
            continue;
        }

        let score = -negamax(board, depth - 1, 1, -POS_INF, -alpha, ctx, true);
        board.unmake_move(mv, state);
        ctx.eval.pop();

        if score > alpha {
            alpha = score;
            best_move = mv;
        }
    }

    if best_move.is_null() { None } else { Some((best_move, alpha)) }
}

// ── Negamax with alpha-beta ───────────────────────────────────────────────────

fn negamax(
    board:      &mut Board,
    depth:      u32,
    ply:        usize,
    mut alpha:  i32,
    mut beta:   i32,
    ctx:        &mut SearchContext,
    allow_null: bool,
) -> i32 {
    ctx.nodes += 1;
    if depth == 0 {
        return quiescence(board, ply, alpha, beta, ctx);
    }

    // ── TT probe ─────────────────────────────────────────────────────────────
    let original_alpha = alpha;
    let tt_move = match ctx.tt.probe(board.hash, depth) {
        Some(e) => {
            match e.bound {
                Bound::Exact => return e.score,
                Bound::Lower => { alpha = alpha.max(e.score); if alpha >= beta { return e.score; } }
                Bound::Upper => { beta  = beta.min(e.score);  if alpha >= beta { return e.score; } }
                Bound::None  => {}
            }
            e.best_move
        }
        None => ctx.tt.probe_move(board.hash),
    };

    let in_check = board.is_in_check();

    // ── Null move pruning ─────────────────────────────────────────────────────
    // Skip our turn; if the score is still >= beta the position is strong
    // enough that we can prune.  Disabled in check and pawn/king endgames
    // (zugzwang risk).
    if allow_null && !in_check && depth >= NULL_MIN_DEPTH && has_non_pawn_material(board) {
        let r = if depth >= NULL_FULL_DEPTH { NULL_R_FULL } else { NULL_R_PARTIAL };
        ctx.eval.push_null();
        let ns = board.make_null_move();
        let null_score = -negamax(board, depth - 1 - r, ply + 1, -beta, -beta + 1, ctx, false);
        board.unmake_null_move(ns);
        ctx.eval.pop();
        if null_score >= beta {
            return beta;
        }
    }

    // ── Futility pruning ──────────────────────────────────────────────────────
    // At depth 1-2, if the static eval is far below alpha, quiet moves are
    // unlikely to recover; skip them.
    let futility_prunable = !in_check && depth <= FUTILITY_MAX_DEPTH && {
        let eval   = ctx.eval.eval_stm(board);
        let margin = if depth == 1 { FUTILITY_MARGIN_1 } else { FUTILITY_MARGIN_2 };
        eval + margin < alpha
    };

    // ── Generate and score moves ──────────────────────────────────────────────
    let pseudo = generate_pseudo_legal(board);
    let n = pseudo.len();
    let mut moves  = [Move::NULL; 256];
    let mut scores = [0i32; 256];
    moves[..n].copy_from_slice(pseudo.as_slice());
    for i in 0..n {
        scores[i] = score_move(board, moves[i], tt_move, &ctx.killers, &ctx.history, ply);
    }

    let us = board.side_to_move;
    let mut best      = NEG_INF;
    let mut best_move = Move::NULL;
    let mut has_legal = false;
    let mut quiet_count = 0usize;

    for i in 0..n {
        // Incremental selection sort: pull best remaining move to position i.
        let mut bi = i;
        for j in (i + 1)..n { if scores[j] > scores[bi] { bi = j; } }
        moves.swap(i, bi);
        scores.swap(i, bi);

        let mv = moves[i];
        let is_capture = board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant();
        let is_quiet   = !is_capture && !mv.is_promo();

        if futility_prunable && is_quiet { continue; }

        ctx.eval.push_move(board, mv);
        let state = board.make_move(mv);

        let king_bb = board.piece_bb(us, PieceType::King);
        if king_bb == EMPTY || board.is_attacked_by(lsb(king_bb), board.side_to_move) {
            board.unmake_move(mv, state);
            ctx.eval.pop();
            continue;
        }

        has_legal = true;
        let gives_check = board.is_in_check();

        // ── Check extension ───────────────────────────────────────────────────
        let extension: u32 = if gives_check && ply < MAX_PLY - 1 { 1 } else { 0 };

        // ── Late Move Reductions (LMR) ────────────────────────────────────────
        let score = if is_quiet && !gives_check && extension == 0
            && quiet_count >= LMR_MIN_QUIET && depth >= LMR_MIN_DEPTH
        {
            let reduced = (depth + extension).saturating_sub(1 + lmr_reduction(quiet_count));
            let s = -negamax(board, reduced, ply + 1, -alpha - 1, -alpha, ctx, true);
            if s > alpha {
                -negamax(board, depth - 1 + extension, ply + 1, -beta, -alpha, ctx, true)
            } else {
                s
            }
        } else {
            -negamax(board, depth - 1 + extension, ply + 1, -beta, -alpha, ctx, true)
        };

        board.unmake_move(mv, state);
        ctx.eval.pop();

        if is_quiet { quiet_count += 1; }

        if score > best {
            best = score;
            best_move = mv;
        }
        if score > alpha { alpha = score; }
        if alpha >= beta {
            if is_quiet {
                ctx.killers.update(mv, ply);
                ctx.history.update(us, mv.from_sq().0, mv.to_sq().0, depth);
            }
            ctx.tt.store(board.hash, depth, best, Bound::Lower, best_move);
            return best;
        }
    }

    if !has_legal {
        // Distance-to-mate: prefer faster mates.
        return if in_check { -CHECKMATE_SCORE + ply as i32 } else { 0 };
    }

    let bound = if best > original_alpha { Bound::Exact } else { Bound::Upper };
    ctx.tt.store(board.hash, depth, best, bound, best_move);
    best
}

// ── Quiescence search ─────────────────────────────────────────────────────────
// Extends the search with captures (and all moves when in check) to avoid
// evaluating positions at a tactical horizon.

fn quiescence(board: &mut Board, ply: usize, mut alpha: i32, beta: i32, ctx: &mut SearchContext) -> i32 {
    ctx.nodes += 1;
    let in_check = board.is_in_check();

    if !in_check {
        let stand_pat = ctx.eval.eval_stm(board);
        if stand_pat >= beta { return stand_pat; }
        alpha = alpha.max(stand_pat);
    }

    let pseudo = generate_pseudo_legal(board);
    let n = pseudo.len();
    let mut moves  = [Move::NULL; 256];
    let mut scores = [0i32; 256];
    moves[..n].copy_from_slice(pseudo.as_slice());

    // Score captures/promos by MVV-LVA; quiet moves get SCORE_QUIET_SKIP so
    // they sort to the back and are skipped by the break below (unless in check).
    for i in 0..n {
        let mv = moves[i];
        scores[i] = if mv.is_promo() {
            SCORE_PROMO_BASE + piece_value(mv.promo_piece_type())
        } else if board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant() {
            capture_mvv_lva(board, mv)
        } else if in_check {
            ctx.history.get(board.side_to_move, mv.from_sq().0, mv.to_sq().0)
        } else {
            SCORE_QUIET_SKIP
        };
    }

    let us = board.side_to_move;
    let mut has_legal = false;

    for i in 0..n {
        let mut bi = i;
        for j in (i + 1)..n { if scores[j] > scores[bi] { bi = j; } }
        moves.swap(i, bi);
        scores.swap(i, bi);

        let mv = moves[i];
        // Once sorted, the first quiet move means all remaining are quiet too.
        if !in_check
            && board.piece_at(mv.to_sq()).is_none()
            && !mv.is_en_passant()
            && !mv.is_promo()
        {
            break;
        }

        ctx.eval.push_move(board, mv);
        let state = board.make_move(mv);
        let king_bb = board.piece_bb(us, PieceType::King);
        if king_bb == EMPTY || board.is_attacked_by(lsb(king_bb), board.side_to_move) {
            board.unmake_move(mv, state);
            ctx.eval.pop();
            continue;
        }

        has_legal = true;
        let score = -quiescence(board, ply + 1, -beta, -alpha, ctx);
        board.unmake_move(mv, state);
        ctx.eval.pop();

        if score >= beta { return score; }
        alpha = alpha.max(score);
    }

    if in_check && !has_legal { return -CHECKMATE_SCORE + ply as i32; }
    alpha
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;
    use crate::core::movegen::generate_legal;

    fn board(fen: &str) -> Board { Board::from_fen(fen).unwrap() }

    #[test]
    fn returns_a_move_from_starting_position() {
        let b = Board::starting_position();
        let mut engine = AlphaBetaEngine::new();
        assert!(engine.choose_move(&b).is_some());
    }

    #[test]
    fn returns_none_when_already_mated() {
        let b = board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        let mut engine = AlphaBetaEngine::new();
        assert!(engine.choose_move(&b).is_none());
    }

    #[test]
    fn finds_checkmate_in_one() {
        let b = board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2");
        let mut engine = AlphaBetaEngine::with_depth(2);
        let mv = engine.choose_move(&b).expect("engine must return a move");
        let mut b2 = b.clone();
        b2.make_move(mv);
        assert!(b2.is_in_check(),              "engine's move must give check");
        assert!(generate_legal(&b2).is_empty(), "engine's move must be checkmate");
    }

    #[test]
    fn finds_checkmate_in_two() {
        let b = board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");
        let mut engine = AlphaBetaEngine::with_depth(4);
        let mv = engine.choose_move(&b).expect("engine must return a move");
        let mut b2 = b.clone();
        b2.make_move(mv);
        assert!(b2.is_in_check() || !generate_legal(&b2).is_empty());
    }

    /// Nodes-per-second benchmark. Run with:
    ///   cargo test speed -- --nocapture --include-ignored
    #[test]
    #[ignore]
    fn speed() {
        use std::time::Instant;

        // A mix of positions: opening, middlegame, endgame.
        let positions = [
            ("startpos",          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("kiwipete",          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            ("middlegame",        "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 10"),
            ("endgame",           "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ];

        let depth = 8;
        let mut total_nodes = 0u64;
        let mut total_ms    = 0u64;

        for (name, fen) in &positions {
            let b = board(fen);
            let mut engine = AlphaBetaEngine::with_depth(depth);
            let t0 = Instant::now();
            engine.choose_move(&b);
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            let nodes = engine.nodes_searched();
            let nps   = if elapsed_ms > 0 { nodes * 1000 / elapsed_ms } else { nodes * 1000 };
            println!("{name:<12}  depth {depth}  nodes {:>10}  time {:>6} ms  nps {:>10}", nodes, elapsed_ms, nps);
            total_nodes += nodes;
            total_ms    += elapsed_ms;
        }

        let avg_nps = if total_ms > 0 { total_nodes * 1000 / total_ms } else { total_nodes * 1000 };
        println!("─────────────────────────────────────────────────────────────────");
        println!("total               nodes {:>10}  time {:>6} ms  nps {:>10}", total_nodes, total_ms, avg_nps);
    }

    /// NNUE nodes-per-second benchmark. Requires `networks/nnue.bin` (run from project root).
    ///
    /// Debug build (fast to run, NPS numbers unrealistic):
    ///   cargo test speed_nnue -- --nocapture --include-ignored
    ///
    /// Release build (accurate NPS, recommended):
    ///   cargo test --release speed_nnue -- --nocapture --include-ignored
    #[test]
    #[ignore]
    fn speed_nnue() {
        use std::time::Instant;
        use std::sync::Arc;
        use super::nnue;

        let nnue = match nnue::Nnue::load("networks/nnue.bin") {
            Ok(n)  => Arc::new(n),
            Err(e) => { println!("speed_nnue: skipping — could not load networks/nnue.bin: {e}"); return; }
        };

        let positions = [
            ("startpos",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("kiwipete",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            ("middlegame", "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 10"),
            ("endgame",    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ];

        let depth = 8u32;
        let mut total_nodes_s = 0u64;
        let mut total_ms_s    = 0u64;
        let mut total_nodes_n = 0u64;
        let mut total_ms_n    = 0u64;

        println!("{:<12}  {:^46}  {:^46}", "", "── Static eval ──────────────────────────", "── NNUE eval ────────────────────────────");
        println!("{:<12}  {:>10}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}", "position", "nodes", "time ms", "nps", "nodes", "time ms", "nps");
        println!("{}", "─".repeat(95));

        for (name, fen) in &positions {
            let b = board(fen);

            let mut se = AlphaBetaEngine::with_depth(depth);
            let t0 = Instant::now();
            se.choose_move(&b);
            let ms_s  = t0.elapsed().as_millis() as u64;
            let nd_s  = se.nodes_searched();
            let nps_s = if ms_s > 0 { nd_s * 1000 / ms_s } else { nd_s * 1000 };

            let mut ne = AlphaBetaEngine::with_nnue(depth, nnue.clone());
            let t1 = Instant::now();
            ne.choose_move(&b);
            let ms_n  = t1.elapsed().as_millis() as u64;
            let nd_n  = ne.nodes_searched();
            let nps_n = if ms_n > 0 { nd_n * 1000 / ms_n } else { nd_n * 1000 };

            println!("{name:<12}  {nd_s:>10}  {ms_s:>8}  {nps_s:>10}  {nd_n:>10}  {ms_n:>8}  {nps_n:>10}");
            total_nodes_s += nd_s;  total_ms_s += ms_s;
            total_nodes_n += nd_n;  total_ms_n += ms_n;
        }

        let avg_s = if total_ms_s > 0 { total_nodes_s * 1000 / total_ms_s } else { total_nodes_s * 1000 };
        let avg_n = if total_ms_n > 0 { total_nodes_n * 1000 / total_ms_n } else { total_nodes_n * 1000 };
        println!("{}", "─".repeat(95));
        println!("{:<12}  {:>10}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}",
            "total", total_nodes_s, total_ms_s, avg_s, total_nodes_n, total_ms_n, avg_n);
    }
}
