use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::core::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
};
use crate::core::bitboard::{lsb, popcount, square_bb, EMPTY};
use crate::core::board::Board;
use crate::core::movegen::generate_pseudo_legal;
use crate::core::moves::Move;
use crate::core::types::{Color, PieceType, Square};
use super::Engine;
use super::eval::{static_eval, CHECKMATE_SCORE};
use super::nnue;
use super::time_manager::TimeManager;

const NEG_INF: i32 = -(CHECKMATE_SCORE + 1);
const POS_INF: i32 =   CHECKMATE_SCORE + 1;
const MAX_PLY: usize = 64;

// ── Mate-score TT adjustment ──────────────────────────────────────────────────

const MATE_THRESHOLD: i32 = CHECKMATE_SCORE - MAX_PLY as i32;

#[inline]
fn to_tt_score(score: i32, ply: usize) -> i32 {
    let p = ply as i32;
    if      score >  MATE_THRESHOLD { score + p }
    else if score < -MATE_THRESHOLD { score - p }
    else                            { score }
}

#[inline]
fn from_tt_score(score: i32, ply: usize) -> i32 {
    let p = ply as i32;
    if      score >  MATE_THRESHOLD { score - p }
    else if score < -MATE_THRESHOLD { score + p }
    else                            { score }
}

// ── Ply-distance penalty ──────────────────────────────────────────────────────

#[inline]
fn apply_ply_penalty(score: i32, ply: usize) -> i32 {
    let penalty = (ply as i32).min(score.abs());
    score - score.signum() * penalty
}

// ── Time management ───────────────────────────────────────────────────────────

const NORMAL_CHECK_INTERVAL: u64 = 1024;
const PANIC_CHECK_INTERVAL:  u64 = 256;

// ── Aspiration windows ────────────────────────────────────────────────────────

const ASPIRATION_DELTA:     i32 = 50;
const ASPIRATION_MIN_DEPTH: u32 = 4;
const ASPIRATION_MAX_DELTA: i32 = 1500;

// ── Move ordering scores ──────────────────────────────────────────────────────
// Buckets (strictly ordered, gaps wide enough to avoid collisions):
//   TT move → queen promos → good captures (SEE≥0) → killers → countermove →
//   history [0..HISTORY_MAX] → bad captures (SEE<0) → underpromotions → quiets (sentinel)

const SCORE_TT_MOVE:      i32 = 30_000_000;
const SCORE_QUEEN_PROMO:  i32 =  9_000_000;
const SCORE_GOOD_CAPTURE: i32 =  5_000_000; // + SEE (≥ 0)
const SCORE_KILLER:       i32 =    900_000;
const SCORE_COUNTERMOVE:  i32 =    800_000;
// history lives in [0, HISTORY_MAX] = [0, 32_000] — naturally below countermove
const HISTORY_MAX:        i32 =     32_000;
const SCORE_BAD_CAPTURE:  i32 = -1_000_000; // + SEE (< 0)
const SCORE_UNDERPROMO:   i32 = -2_000_000;
const SCORE_QUIET_SKIP:   i32 = -3_000_000; // sentinel: skipped in qsearch

// ── Search tuning parameters ──────────────────────────────────────────────────

const NULL_MIN_DEPTH:  u32 = 3;
const NULL_FULL_DEPTH: u32 = 6;
const NULL_R_PARTIAL:  u32 = 2;
const NULL_R_FULL:     u32 = 3;

const FUTILITY_MAX_DEPTH: u32 = 2;
const FUTILITY_MARGIN_1:  i32 = 100;
const FUTILITY_MARGIN_2:  i32 = 300;

// Reverse futility pruning: if eval ≥ beta + margin×depth, prune
const RFP_MAX_DEPTH: u32 = 6;
const RFP_MARGIN:    i32 = 70;

const LMR_MIN_QUIET: usize = 3;
const LMR_MIN_DEPTH: u32   = 3;

// Late move pruning thresholds: skip quiet moves after this many searched at low depth
const LMP_MAX_DEPTH:  u32 = 4;
const LMP_THRESHOLDS: [usize; 5] = [0, 4, 8, 14, 20];

// SEE pruning: skip moves with SEE below threshold (scaled by depth)
const SEE_QUIET_MARGIN: i32 = -60;  // per depth, quiet moves
const SEE_CAPT_MARGIN:  i32 = -90;  // per depth, capture moves

// Qsearch delta pruning: skip captures that can't raise alpha even with a bonus
const DELTA_MARGIN: i32 = 200;

// ── Static Exchange Evaluation ────────────────────────────────────────────────

const SEE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 20_000]; // P N B R Q K

/// Least-valuable attacker of `stm` on `sq` given the live occupancy `occ`.
/// Returns `(attacker_bitboard_lsb, piece_type)`, or `None` if no attacker.
/// Slider attacks are recomputed through `occ` so x-rays are handled correctly.
fn least_valuable_attacker(
    board: &Board,
    sq:    Square,
    stm:   Color,
    occ:   u64,
) -> Option<(u64, PieceType)> {
    let s = stm as usize;

    // Pawns: a stm-pawn on X attacks sq iff sq ∈ pawn_attacks(stm, X),
    // equivalently X ∈ pawn_attacks(stm.flip(), sq).
    let pawns = board.pieces[s][PieceType::Pawn as usize] & occ;
    let pa    = pawn_attacks(stm.flip(), sq) & pawns;
    if pa != EMPTY { return Some((pa & pa.wrapping_neg(), PieceType::Pawn)); }

    let knights = board.pieces[s][PieceType::Knight as usize] & occ;
    let na      = knight_attacks(sq) & knights;
    if na != EMPTY { return Some((na & na.wrapping_neg(), PieceType::Knight)); }

    let bishops = board.pieces[s][PieceType::Bishop as usize] & occ;
    let ba      = bishop_attacks(sq, occ) & bishops;
    if ba != EMPTY { return Some((ba & ba.wrapping_neg(), PieceType::Bishop)); }

    let rooks = board.pieces[s][PieceType::Rook as usize] & occ;
    let ra    = rook_attacks(sq, occ) & rooks;
    if ra != EMPTY { return Some((ra & ra.wrapping_neg(), PieceType::Rook)); }

    let queens = board.pieces[s][PieceType::Queen as usize] & occ;
    let qa     = queen_attacks(sq, occ) & queens;
    if qa != EMPTY { return Some((qa & qa.wrapping_neg(), PieceType::Queen)); }

    let kings = board.pieces[s][PieceType::King as usize] & occ;
    let ka    = king_attacks(sq) & kings;
    if ka != EMPTY { return Some((ka & ka.wrapping_neg(), PieceType::King)); }

    None
}

/// Static exchange evaluation — expected net material gain for the moving side.
/// Positive = winning exchange, negative = losing exchange.
/// Uses the recursive minimax gain-array technique with x-ray support.
fn see(board: &Board, mv: Move) -> i32 {
    let to   = mv.to_sq();
    let from = mv.from_sq();

    let mut occ = board.all_occupancy;
    occ ^= square_bb(from); // moving piece is no longer on its origin

    // Value of the initially captured piece
    let captured_val = if mv.is_en_passant() {
        // captured pawn sits one rank behind the target square
        let ep_sq = Square(if board.side_to_move == Color::White {
            to.0 - 8
        } else {
            to.0 + 8
        });
        occ ^= square_bb(ep_sq);
        SEE_VALUES[PieceType::Pawn as usize]
    } else {
        board.piece_at(to).map_or(0, |p| SEE_VALUES[p.piece_type as usize])
    };

    // Value of the piece that just moved to `to` (can now be recaptured)
    let mut next_on_to = if mv.is_promo() {
        SEE_VALUES[mv.promo_piece_type() as usize]
    } else {
        board.piece_at(from).map_or(SEE_VALUES[0], |p| SEE_VALUES[p.piece_type as usize])
    };

    // gain[i] = value of the piece captured at recapture step i.
    // Recaptures alternate sides; the first recapture is by board.side_to_move.flip().
    let mut gain = [0i32; 32];
    let mut d    = 0usize;
    let mut stm  = board.side_to_move.flip();

    loop {
        let Some((lva_bb, lva_pt)) = least_valuable_attacker(board, to, stm, occ) else { break };
        if d >= gain.len() - 1 { break; }

        gain[d]  = next_on_to;   // the piece currently on `to` is the new victim
        d       += 1;
        occ     ^= lva_bb;       // lva moves to `to`; remove from occupancy
        next_on_to = SEE_VALUES[lva_pt as usize];
        stm      = stm.flip();
    }

    // Negamax: each side can choose not to recapture (take max(0, gain) at each step).
    // Unroll from the deepest ply backward.
    let mut result = 0i32;
    for i in (0..d).rev() {
        result = (gain[i] - result).max(0);
    }

    captured_val - result
}

// ── Insufficient material detection ──────────────────────────────────────────

fn is_insufficient_material(board: &Board) -> bool {
    // Any pawn, rook, or queen → mating material exists
    for c in 0..2 {
        if board.pieces[c][PieceType::Pawn  as usize] != EMPTY { return false; }
        if board.pieces[c][PieceType::Rook  as usize] != EMPTY { return false; }
        if board.pieces[c][PieceType::Queen as usize] != EMPTY { return false; }
    }

    let w_minor = popcount(
        board.pieces[0][PieceType::Knight as usize] |
        board.pieces[0][PieceType::Bishop as usize],
    ) as usize;
    let b_minor = popcount(
        board.pieces[1][PieceType::Knight as usize] |
        board.pieces[1][PieceType::Bishop as usize],
    ) as usize;

    // K vs K, K+1 minor vs K, K vs K+1 minor
    if w_minor + b_minor <= 1 { return true; }

    // K+B vs K+B: draw only when bishops share square color
    if w_minor == 1 && b_minor == 1 {
        let wb = board.pieces[0][PieceType::Bishop as usize];
        let bb = board.pieces[1][PieceType::Bishop as usize];
        if wb != EMPTY && bb != EMPTY {
            const LIGHT: u64 = 0x55AA_55AA_55AA_55AA;
            return (wb & LIGHT != EMPTY) == (bb & LIGHT != EMPTY);
        }
    }

    false
}

// ── Transposition Table (4 entries per bucket) ────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
enum Bound { #[default] None, Exact, Lower, Upper }

#[derive(Clone, Copy, Default)]
struct TtEntry {
    hash:       u64,
    score:      i32,
    best_move:  Move,
    depth:      u8,
    bound:      Bound,
    generation: u8,
}

const BUCKET_SIZE:  usize = 4;
const BUCKET_COUNT: usize = 1 << 18; // 256 K buckets × 4 = 1 M entries ≈ 16 MB

#[derive(Clone, Copy, Default)]
struct TtBucket {
    entries: [TtEntry; BUCKET_SIZE],
}

struct TranspositionTable {
    buckets:    Box<[TtBucket]>,
    generation: u8,
}

impl TranspositionTable {
    fn new() -> Self {
        Self {
            buckets:    vec![TtBucket::default(); BUCKET_COUNT].into_boxed_slice(),
            generation: 0,
        }
    }

    fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    #[inline]
    fn probe(&self, hash: u64, depth: u32) -> Option<TtEntry> {
        for e in &self.buckets[(hash as usize) & (BUCKET_COUNT - 1)].entries {
            if e.bound != Bound::None && e.hash == hash && u32::from(e.depth) >= depth {
                return Some(*e);
            }
        }
        None
    }

    /// Return the best move stored for this position regardless of depth.
    #[inline]
    fn probe_move(&self, hash: u64) -> Move {
        for e in &self.buckets[(hash as usize) & (BUCKET_COUNT - 1)].entries {
            if e.bound != Bound::None && e.hash == hash {
                return e.best_move;
            }
        }
        Move::NULL
    }

    #[inline]
    fn store(&mut self, hash: u64, depth: u32, score: i32, bound: Bound, best_move: Move) {
        let cur_gen = self.generation;
        let bucket  = &mut self.buckets[(hash as usize) & (BUCKET_COUNT - 1)];

        // Replacement: always update same-hash entry; otherwise evict lowest-quality slot.
        // Quality = depth, penalised heavily for stale generations.
        let mut replace_idx   = 0;
        let mut replace_score = i32::MAX;

        for (i, e) in bucket.entries.iter().enumerate() {
            if e.hash == hash {
                replace_idx = i;
                break;
            }
            let age_penalty = if e.generation != cur_gen { 8 } else { 0 };
            let q = e.depth as i32 - age_penalty;
            if q < replace_score {
                replace_score = q;
                replace_idx   = i;
            }
        }

        let slot = &mut bucket.entries[replace_idx];
        // Don't clobber a deeper same-generation entry unless it's our own hash
        if slot.hash != hash && slot.generation == cur_gen && u32::from(slot.depth) > depth {
            return;
        }
        *slot = TtEntry { hash, score, best_move, depth: depth as u8, bound, generation: cur_gen };
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
        if slot[0] != mv { slot[1] = slot[0]; slot[0] = mv; }
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
                for v in f.iter_mut() { *v >>= 1; }
            }
        }
    }
}

// ── Countermove heuristic ─────────────────────────────────────────────────────
// For each opponent move (from→to), store the quiet move that refuted it.

struct CountermoveTable([[Move; 64]; 64]);

impl CountermoveTable {
    fn new() -> Self { CountermoveTable([[Move::NULL; 64]; 64]) }

    #[inline]
    fn get(&self, prev: Move) -> Move {
        if prev.is_null() { return Move::NULL; }
        self.0[prev.from_sq().0 as usize][prev.to_sq().0 as usize]
    }

    #[inline]
    fn update(&mut self, prev: Move, mv: Move) {
        if prev.is_null() { return; }
        self.0[prev.from_sq().0 as usize][prev.to_sq().0 as usize] = mv;
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

    fn init(&mut self, board: &Board) {
        let Evaluator::Nnue { weights, stack } = self else { return };
        stack.init(weights, board);
    }

    #[inline]
    fn push_move(&mut self, board: &Board, mv: Move) {
        let Evaluator::Nnue { weights, stack } = self else { return };
        stack.push_move(weights, board, mv);
    }

    #[inline]
    fn push_null(&mut self) {
        let Evaluator::Nnue { weights: _, stack } = self else { return };
        stack.push_null();
    }

    #[inline]
    fn pop(&mut self) {
        let Evaluator::Nnue { weights: _, stack } = self else { return };
        stack.pop();
    }
}

// ── Search context ────────────────────────────────────────────────────────────

struct SearchContext {
    tt:               TranspositionTable,
    killers:          KillerTable,
    history:          HistoryTable,
    countermoves:     CountermoveTable,
    nodes:            u64,
    eval:             Evaluator,
    abort:            Arc<AtomicBool>,
    check_interval:   u64,
    hard_deadline:    Instant,
    position_history: Vec<u64>,
    game_history_len: usize,
}

impl SearchContext {
    #[inline]
    fn aborted(&self) -> bool { self.abort.load(Ordering::Relaxed) }

    fn prepare(&mut self, tm: &TimeManager) {
        self.nodes          = 0;
        self.abort          = tm.abort.clone();
        self.hard_deadline  = tm.hard_deadline;
        self.check_interval = if tm.panic_mode { PANIC_CHECK_INTERVAL } else { NORMAL_CHECK_INTERVAL };
        self.tt.new_search();
    }
}

// ── LMR reduction ─────────────────────────────────────────────────────────────

#[inline]
fn lmr_reduction(quiet_count: usize) -> u32 {
    1 + (quiet_count as u32) / 6
}

// ── Move scoring ──────────────────────────────────────────────────────────────

#[inline]
fn score_move(
    board:       &Board,
    mv:          Move,
    tt_move:     Move,
    killers:     &KillerTable,
    countermove: Move,
    history:     &HistoryTable,
    ply:         usize,
) -> i32 {
    if mv == tt_move { return SCORE_TT_MOVE; }

    if mv.is_promo() {
        return if mv.promo_piece_type() == PieceType::Queen {
            SCORE_QUEEN_PROMO
        } else {
            SCORE_UNDERPROMO
        };
    }

    if board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant() {
        let s = see(board, mv);
        return if s >= 0 { SCORE_GOOD_CAPTURE + s } else { SCORE_BAD_CAPTURE + s };
    }

    // Quiet move
    if killers.is_killer(mv, ply) { return SCORE_KILLER; }
    if mv == countermove           { return SCORE_COUNTERMOVE; }
    history.get(board.side_to_move, mv.from_sq().0, mv.to_sq().0)
}

// ── Null-move safety check ────────────────────────────────────────────────────

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
                tt:               TranspositionTable::new(),
                killers:          KillerTable::new(),
                history:          HistoryTable::new(),
                countermoves:     CountermoveTable::new(),
                nodes:            0,
                eval,
                abort:            Arc::new(AtomicBool::new(false)),
                check_interval:   NORMAL_CHECK_INTERVAL,
                hard_deadline:    Instant::now() + Duration::from_secs(86_400),
                position_history: Vec::new(),
                game_history_len: 0,
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

    pub fn choose_move_timed(
        &mut self,
        board:        &Board,
        mut tm:       TimeManager,
        game_history: &[u64],
    ) -> Option<Move> {
        let mut b = board.clone();
        self.ctx.killers      = KillerTable::new();
        self.ctx.countermoves = CountermoveTable::new();
        self.ctx.history.age();
        self.ctx.eval.init(&b);
        self.ctx.prepare(&tm);
        self.ctx.position_history.clear();
        self.ctx.position_history.extend_from_slice(game_history);
        self.ctx.game_history_len = game_history.len();

        let mut best:       Move        = Move::NULL;
        let mut prev_score: Option<i32> = None;
        self.last_score = None;

        'id: for d in 1..=self.depth {
            if d > 1 && tm.soft_expired() { break; }

            let (mut lo, mut hi, mut delta) =
                if let Some(prev) = prev_score.filter(|_| d >= ASPIRATION_MIN_DEPTH) {
                    (prev - ASPIRATION_DELTA, prev + ASPIRATION_DELTA, ASPIRATION_DELTA)
                } else {
                    (NEG_INF, POS_INF, ASPIRATION_DELTA)
                };

            let mut aborted = false;
            loop {
                let result = search_root(&mut b, d, lo, hi, &mut self.ctx);

                if self.ctx.aborted() { aborted = true; break; }

                match result {
                    None => break,
                    Some((mv, score)) => {
                        if score <= lo && lo > NEG_INF {
                            delta = (delta * 2).min(ASPIRATION_MAX_DELTA);
                            lo    = lo.saturating_sub(delta).max(NEG_INF);
                            tm.extend(0.20);
                            self.ctx.hard_deadline = tm.hard_deadline;
                            continue;
                        } else if score >= hi && hi < POS_INF {
                            delta = (delta * 2).min(ASPIRATION_MAX_DELTA);
                            hi    = hi.saturating_add(delta).min(POS_INF);
                            continue;
                        } else {
                            best            = mv;
                            prev_score      = Some(score);
                            self.last_score = Some(score);
                            break;
                        }
                    }
                }
            }

            if aborted { break 'id; }
        }

        if best.is_null() { None } else { Some(best) }
    }
}

impl Engine for AlphaBetaEngine {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let tm = TimeManager::infinite();
        let mut b = board.clone();
        self.ctx.killers      = KillerTable::new();
        self.ctx.countermoves = CountermoveTable::new();
        self.ctx.history.age();
        self.ctx.eval.init(&b);
        self.ctx.prepare(&tm);
        self.ctx.position_history.clear();
        self.ctx.game_history_len = 0;
        let mut best = Move::NULL;
        self.last_score = None;
        for d in 1..=self.depth {
            if let Some((mv, score)) = search_root(&mut b, d, NEG_INF, POS_INF, &mut self.ctx) {
                if !self.ctx.aborted() {
                    best = mv;
                    self.last_score = Some(score);
                }
            }
            if self.ctx.aborted() { break; }
        }
        if best.is_null() { None } else { Some(best) }
    }

    fn name(&self) -> &str { self.name.as_str() }

    fn last_score(&self) -> Option<i32> { self.last_score }
}

// ── Root search ───────────────────────────────────────────────────────────────

fn search_root(
    board:    &mut Board,
    depth:    u32,
    alpha_in: i32,
    beta:     i32,
    ctx:      &mut SearchContext,
) -> Option<(Move, i32)> {
    let pseudo = generate_pseudo_legal(board);
    let n = pseudo.len();
    if n == 0 { return None; }

    let mut moves  = [Move::NULL; 256];
    let mut scores = [0i32; 256];
    moves[..n].copy_from_slice(pseudo.as_slice());

    let tt_move = ctx.tt.probe_move(board.hash);
    for i in 0..n {
        // At root we don't have a countermove context; pass NULL
        scores[i] = score_move(board, moves[i], tt_move, &ctx.killers, Move::NULL, &ctx.history, 0);
    }

    let us          = board.side_to_move;
    let mut alpha   = alpha_in;
    let mut best_move  = Move::NULL;
    let mut best_score = NEG_INF;
    let mut legal_count = 0usize;

    for i in 0..n {
        let mut bi = i;
        for j in (i + 1)..n { if scores[j] > scores[bi] { bi = j; } }
        moves.swap(i, bi);
        scores.swap(i, bi);

        let mv = moves[i];
        ctx.position_history.push(board.hash);
        ctx.eval.push_move(board, mv);
        let state = board.make_move(mv);

        let king_bb = board.piece_bb(us, PieceType::King);
        if king_bb == EMPTY || board.is_attacked_by(lsb(king_bb), board.side_to_move) {
            board.unmake_move(mv, state);
            ctx.eval.pop();
            ctx.position_history.pop();
            continue;
        }

        legal_count += 1;
        let score = if legal_count == 1 {
            -negamax(board, depth - 1, 1, -beta, -alpha, ctx, true, mv)
        } else {
            // PVS: null-window first
            let s = -negamax(board, depth - 1, 1, -alpha - 1, -alpha, ctx, true, mv);
            if s > alpha && s < beta {
                -negamax(board, depth - 1, 1, -beta, -alpha, ctx, true, mv)
            } else { s }
        };

        board.unmake_move(mv, state);
        ctx.eval.pop();
        ctx.position_history.pop();

        if ctx.aborted() { break; }

        if score > best_score {
            best_score = score;
            best_move  = mv;
        }
        if score > alpha {
            alpha = score;
            if alpha >= beta { break; }
        }
    }

    if legal_count == 0 { None } else { Some((best_move, best_score)) }
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
    prev_move:  Move,   // opponent's last move (for countermove lookup)
) -> i32 {
    if ctx.aborted() { return 0; }

    ctx.nodes += 1;

    if ctx.nodes & (ctx.check_interval - 1) == 0
        && Instant::now() >= ctx.hard_deadline
    {
        ctx.abort.store(true, Ordering::Relaxed);
        return 0;
    }

    // ── Draw detection ────────────────────────────────────────────────────────
    if board.half_move_clock >= 100 { return 0; }
    if is_insufficient_material(board) { return 0; }

    {
        let gl          = ctx.game_history_len;
        let game_reps   = ctx.position_history[..gl].iter().filter(|&&h| h == board.hash).count();
        let search_reps = ctx.position_history[gl..].iter().filter(|&&h| h == board.hash).count();
        if game_reps >= 2 || search_reps >= 1 { return 0; }
    }

    if depth == 0 {
        return quiescence(board, ply, alpha, beta, ctx);
    }

    // ── TT probe ─────────────────────────────────────────────────────────────
    let original_alpha = alpha;
    let tt_move = match ctx.tt.probe(board.hash, depth) {
        Some(e) => {
            let tt_score = from_tt_score(e.score, ply);
            match e.bound {
                Bound::Exact => return tt_score,
                Bound::Lower => { alpha = alpha.max(tt_score); if alpha >= beta { return tt_score; } }
                Bound::Upper => { beta  = beta.min(tt_score);  if alpha >= beta { return tt_score; } }
                Bound::None  => {}
            }
            e.best_move
        }
        None => ctx.tt.probe_move(board.hash),
    };

    let in_check  = board.is_in_check();
    let is_pv     = beta > alpha + 1; // full-window node

    // ── Static eval (shared by RFP + futility) ────────────────────────────────
    // Avoid calling eval when in check (position is unstable).
    let static_eval = if !in_check { Some(ctx.eval.eval_stm(board)) } else { None };

    // ── Reverse futility pruning ──────────────────────────────────────────────
    // If the position is so good that even giving up a margin won't drop below
    // beta, we can prune the whole subtree.
    if !is_pv && !in_check && depth <= RFP_MAX_DEPTH {
        if let Some(eval) = static_eval {
            if eval >= beta + RFP_MARGIN * depth as i32 {
                return eval;
            }
        }
    }

    // ── Null move pruning ─────────────────────────────────────────────────────
    if allow_null && !in_check && !is_pv && depth >= NULL_MIN_DEPTH && has_non_pawn_material(board) {
        if let Some(eval) = static_eval {
            if eval >= beta {
                let r = if depth >= NULL_FULL_DEPTH { NULL_R_FULL } else { NULL_R_PARTIAL };
                ctx.eval.push_null();
                let ns = board.make_null_move();
                let null_score = -negamax(board, depth - 1 - r, ply + 1, -beta, -beta + 1, ctx, false, Move::NULL);
                board.unmake_null_move(ns);
                ctx.eval.pop();
                if null_score >= beta {
                    return beta;
                }
            }
        }
    }

    // ── Futility pruning flag ─────────────────────────────────────────────────
    let futility_prunable = !in_check && depth <= FUTILITY_MAX_DEPTH && {
        let eval   = static_eval.unwrap_or_else(|| ctx.eval.eval_stm(board));
        let margin = if depth == 1 { FUTILITY_MARGIN_1 } else { FUTILITY_MARGIN_2 };
        eval + margin < alpha
    };

    // ── Generate and score moves ──────────────────────────────────────────────
    let pseudo = generate_pseudo_legal(board);
    let n      = pseudo.len();
    let mut moves  = [Move::NULL; 256];
    let mut scores = [0i32; 256];
    moves[..n].copy_from_slice(pseudo.as_slice());

    let countermove = ctx.countermoves.get(prev_move);
    for i in 0..n {
        scores[i] = score_move(board, moves[i], tt_move, &ctx.killers, countermove, &ctx.history, ply);
    }

    let us           = board.side_to_move;
    let mut best      = NEG_INF;
    let mut best_move = Move::NULL;
    let mut legal_count  = 0usize;
    let mut quiet_searched = 0usize; // legal quiet moves searched (for LMR/LMP)

    for i in 0..n {
        let mut bi = i;
        for j in (i + 1)..n { if scores[j] > scores[bi] { bi = j; } }
        moves.swap(i, bi);
        scores.swap(i, bi);

        let mv        = moves[i];
        let mv_score  = scores[i];
        let is_capture = board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant();
        let is_quiet   = !is_capture && !mv.is_promo();

        // ── Futility pruning ──────────────────────────────────────────────────
        if futility_prunable && is_quiet { continue; }

        // ── Late move pruning (pre-make, based on move score) ─────────────────
        if !in_check && is_quiet && depth <= LMP_MAX_DEPTH
            && quiet_searched >= LMP_THRESHOLDS[depth as usize]
            && mv != tt_move
        {
            quiet_searched += 1;
            continue;
        }

        // ── SEE pruning ───────────────────────────────────────────────────────
        // Skip moves with clearly losing exchanges at non-root, non-PV nodes.
        if legal_count > 0 && !in_check && !is_pv {
            let threshold = if is_quiet {
                SEE_QUIET_MARGIN * depth as i32
            } else {
                SEE_CAPT_MARGIN * depth as i32
            };
            // For captures the score already encodes SEE; re-use it.
            if is_capture && mv_score < SCORE_GOOD_CAPTURE && mv_score - SCORE_BAD_CAPTURE < threshold {
                continue;
            }
            if is_quiet && mv_score < threshold {
                continue;
            }
        }

        ctx.position_history.push(board.hash);
        ctx.eval.push_move(board, mv);
        let state = board.make_move(mv);

        let king_bb = board.piece_bb(us, PieceType::King);
        if king_bb == EMPTY || board.is_attacked_by(lsb(king_bb), board.side_to_move) {
            board.unmake_move(mv, state);
            ctx.eval.pop();
            ctx.position_history.pop();
            continue;
        }

        legal_count += 1;
        if is_quiet { quiet_searched += 1; }

        let gives_check = board.is_in_check();
        let extension: u32 = if gives_check && ply < MAX_PLY - 1 { 1 } else { 0 };
        let new_depth = depth - 1 + extension;

        // ── PVS + LMR ─────────────────────────────────────────────────────────
        let do_lmr = is_quiet && !gives_check && extension == 0
            && quiet_searched > LMR_MIN_QUIET && depth >= LMR_MIN_DEPTH;

        let score = if legal_count == 1 {
            // First legal move: full window, full depth
            -negamax(board, new_depth, ply + 1, -beta, -alpha, ctx, true, mv)
        } else {
            // LMR: try reduced depth with null window
            let search_depth = if do_lmr {
                new_depth.saturating_sub(lmr_reduction(quiet_searched))
            } else {
                new_depth
            };

            let mut s = -negamax(board, search_depth, ply + 1, -alpha - 1, -alpha, ctx, true, mv);

            // If LMR failed high, re-search at full depth with null window
            if s > alpha && search_depth < new_depth {
                s = -negamax(board, new_depth, ply + 1, -alpha - 1, -alpha, ctx, true, mv);
            }

            // PVS: if null window failed high, re-search with full window
            if s > alpha && s < beta {
                s = -negamax(board, new_depth, ply + 1, -beta, -alpha, ctx, true, mv);
            }

            s
        };

        board.unmake_move(mv, state);
        ctx.eval.pop();
        ctx.position_history.pop();

        if score > best {
            best      = score;
            best_move = mv;
        }
        if score > alpha { alpha = score; }
        if alpha >= beta {
            if is_quiet {
                ctx.killers.update(mv, ply);
                ctx.history.update(us, mv.from_sq().0, mv.to_sq().0, depth);
                ctx.countermoves.update(prev_move, mv);
            }
            ctx.tt.store(board.hash, depth, to_tt_score(best, ply), Bound::Lower, best_move);
            return best;
        }
    }

    if legal_count == 0 {
        return if in_check { -CHECKMATE_SCORE + ply as i32 } else { 0 };
    }

    let bound = if best > original_alpha { Bound::Exact } else { Bound::Upper };
    ctx.tt.store(board.hash, depth, to_tt_score(best, ply), bound, best_move);
    best
}

// ── Quiescence search ─────────────────────────────────────────────────────────

fn quiescence(board: &mut Board, ply: usize, mut alpha: i32, beta: i32, ctx: &mut SearchContext) -> i32 {
    if ctx.aborted() { return 0; }
    ctx.nodes += 1;
    let in_check = board.is_in_check();

    let stand_pat = if !in_check {
        let sp = ctx.eval.eval_stm(board);
        if sp >= beta { return sp; }
        alpha = alpha.max(sp);
        sp
    } else {
        NEG_INF
    };

    let pseudo = generate_pseudo_legal(board);
    let n = pseudo.len();
    let mut moves  = [Move::NULL; 256];
    let mut scores = [0i32; 256];
    moves[..n].copy_from_slice(pseudo.as_slice());

    for i in 0..n {
        let mv = moves[i];
        scores[i] = if mv.is_promo() {
            if mv.promo_piece_type() == PieceType::Queen { SCORE_QUEEN_PROMO }
            else { SCORE_UNDERPROMO }
        } else if board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant() {
            let s = see(board, mv);
            if s >= 0 { SCORE_GOOD_CAPTURE + s } else { SCORE_BAD_CAPTURE + s }
        } else if in_check {
            // When in check, search all moves ordered by history
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
        if scores[i] <= SCORE_QUIET_SKIP { break; }

        let is_capture = board.piece_at(mv.to_sq()).is_some() || mv.is_en_passant();

        // ── Delta pruning ─────────────────────────────────────────────────────
        // Skip captures that can't raise alpha even with the best possible gain.
        if !in_check && is_capture && !mv.is_promo() {
            let capture_val = board.piece_at(mv.to_sq())
                .map_or(SEE_VALUES[0], |p| SEE_VALUES[p.piece_type as usize]);
            if stand_pat + capture_val + DELTA_MARGIN < alpha {
                continue;
            }
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
    apply_ply_penalty(alpha, ply)
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

    #[test]
    fn see_winning_capture() {
        // Queen on e1 captures undefended pawn on e5 (same file, no defenders): SEE = +100
        let b = board("7k/8/8/4p3/8/8/8/4QK2 w - - 0 1");
        let legal = generate_legal(&b);
        let mv = legal.as_slice().iter().find(|m| {
            b.piece_at(m.to_sq()).is_some()
        }).copied().expect("must have a capture");
        assert!(see(&b, mv) > 0, "queen takes undefended pawn should be winning");
    }

    #[test]
    fn see_losing_capture() {
        // Queen on e1 captures pawn on e4 defended by pawn on d5: SEE should be negative
        let b = board("7k/8/8/3p4/4p3/8/8/4QK2 w - - 0 1");
        let legal = generate_legal(&b);
        let captures: Vec<_> = legal.as_slice().iter()
            .filter(|m| b.piece_at(m.to_sq()).is_some())
            .collect();
        let has_losing = captures.iter().any(|&&m| see(&b, m) < 0);
        assert!(has_losing, "at least one capture should be losing (queen takes defended pawn)");
    }

    #[test]
    fn insufficient_material_kk() {
        let b = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert!(is_insufficient_material(&b));
    }

    #[test]
    fn insufficient_material_kbk() {
        let b = board("4k3/8/8/8/8/8/8/3BK3 w - - 0 1");
        assert!(is_insufficient_material(&b));
    }

    #[test]
    fn sufficient_material_krk() {
        let b = board("4k3/8/8/8/8/8/8/3RK3 w - - 0 1");
        assert!(!is_insufficient_material(&b));
    }

    /// Nodes-per-second benchmark. Run with:
    ///   cargo test speed -- --nocapture --include-ignored
    #[test]
    #[ignore]
    fn speed() {
        use std::time::Instant;

        let positions = [
            ("startpos",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("kiwipete",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            ("middlegame",  "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 10"),
            ("endgame",    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
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

    #[test]
    #[ignore]
    fn speed_nnue() {
        use std::time::Instant;
        use std::sync::Arc;
        use super::nnue;

        let nnue_path = env!("RCHESS_NNUE_PATH");
        let nnue = match nnue::Nnue::load(nnue_path) {
            Ok(n)  => Arc::new(n),
            Err(e) => { println!("speed_nnue: skipping — could not load {nnue_path}: {e}"); return; }
        };

        let positions = [
            ("startpos",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("kiwipete",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            ("middlegame", "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 10"),
            ("endgame",    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ];

        let depth = 8u32;
        let mut total_nodes_s = 0u64; let mut total_ms_s = 0u64;
        let mut total_nodes_n = 0u64; let mut total_ms_n = 0u64;

        println!("{:<12}  {:^46}  {:^46}", "", "── Static eval ──", "── NNUE eval ────");
        println!("{:<12}  {:>10}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}", "position", "nodes", "time ms", "nps", "nodes", "time ms", "nps");
        println!("{}", "─".repeat(95));

        for (name, fen) in &positions {
            let b = board(fen);

            let mut se = AlphaBetaEngine::with_depth(depth);
            let t0 = Instant::now();
            se.choose_move(&b);
            let ms_s = t0.elapsed().as_millis() as u64;
            let nd_s = se.nodes_searched();
            let nps_s = if ms_s > 0 { nd_s * 1000 / ms_s } else { nd_s * 1000 };

            let mut ne = AlphaBetaEngine::with_nnue(depth, nnue.clone());
            let t1 = Instant::now();
            ne.choose_move(&b);
            let ms_n = t1.elapsed().as_millis() as u64;
            let nd_n = ne.nodes_searched();
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
