use std::io::{self, BufRead, Write};
use std::sync::Arc;

use crate::core::board::Board;
use crate::core::movegen::generate_legal;
use crate::core::moves::Move;
use crate::core::types::{Color, PieceType, Square};
use crate::engine::alpha_beta::AlphaBetaEngine;
use crate::engine::nnue::Nnue;
use crate::engine::time_manager::TimeManager;


const MAX_SEARCH_DEPTH: u32 = 30;

// ── Move parsing ──────────────────────────────────────────────────────────────

fn parse_square(b: &[u8]) -> Option<Square> {
    if b.len() < 2 { return None; }
    let file = b[0].wrapping_sub(b'a');
    let rank  = b[1].wrapping_sub(b'1');
    if file < 8 && rank < 8 { Some(Square::new(file, rank)) } else { None }
}

/// Match a UCI move string (e.g. "e2e4", "a7a8q") against the legal moves in
/// the current position and return the matching `Move`.
fn parse_move(board: &Board, s: &str) -> Option<Move> {
    let b = s.as_bytes();
    if b.len() < 4 { return None; }
    let from = parse_square(&b[0..2])?;
    let to   = parse_square(&b[2..4])?;
    let promo_char = b.get(4).copied();

    let legal = generate_legal(board);
    for &mv in legal.as_slice() {
        if mv.from_sq() != from || mv.to_sq() != to { continue; }
        if mv.is_promo() {
            let want = match promo_char {
                Some(b'q') | None => PieceType::Queen,
                Some(b'r') => PieceType::Rook,
                Some(b'b') => PieceType::Bishop,
                Some(b'n') => PieceType::Knight,
                _ => PieceType::Queen,
            };
            if mv.promo_piece_type() == want { return Some(mv); }
        } else {
            return Some(mv);
        }
    }
    None
}

// ── Position command ──────────────────────────────────────────────────────────

/// Returns the board and the hash of every position visited before reaching it,
/// so the engine can detect threefold repetitions during search.
fn parse_position(line: &str) -> Option<(Board, Vec<u64>)> {
    let rest = line.strip_prefix("position ")?;

    let (mut board, moves_str) = if let Some(r) = rest.strip_prefix("startpos") {
        let moves = r.trim_start().strip_prefix("moves").unwrap_or("");
        (Board::starting_position(), moves)
    } else if let Some(r) = rest.strip_prefix("fen ") {
        let (fen_part, moves_part) = match r.find(" moves ") {
            Some(i) => (&r[..i], &r[i + 7..]),
            None    => (r, ""),
        };
        (Board::from_fen(fen_part.trim()).ok()?, moves_part)
    } else {
        return None;
    };

    let mut history: Vec<u64> = Vec::new();
    for mv_str in moves_str.split_whitespace() {
        history.push(board.hash);   // record hash BEFORE the move
        let mv = parse_move(&board, mv_str)?;
        board.make_move(mv);
    }
    Some((board, history))
}

// ── Go command: time budget ───────────────────────────────────────────────────

struct GoParams {
    wtime:     Option<u64>,
    btime:     Option<u64>,
    winc:      u64,
    binc:      u64,
    movetime:  Option<u64>,
    depth:     Option<u32>,
    infinite:  bool,
}

fn parse_go(line: &str) -> GoParams {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut p = GoParams { wtime: None, btime: None, winc: 0, binc: 0,
                           movetime: None, depth: None, infinite: false };
    let mut i = 1usize;
    while i < tokens.len() {
        match tokens[i] {
            "wtime"    => { p.wtime    = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "btime"    => { p.btime    = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "winc"     => { p.winc     = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "binc"     => { p.binc     = tokens.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "movetime" => { p.movetime = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "depth"    => { p.depth    = tokens.get(i+1).and_then(|s| s.parse().ok()); i += 2; }
            "infinite" => { p.infinite = true; i += 1; }
            _          => { i += 1; }
        }
    }
    p
}

/// Build a TimeManager and resolve the max search depth from a `go` command.
fn create_time_manager(go: &GoParams, side: Color, overhead_ms: u64) -> (TimeManager, u32) {
    if go.infinite {
        return (TimeManager::infinite(), MAX_SEARCH_DEPTH);
    }
    if let Some(d) = go.depth {
        return (TimeManager::infinite(), d);
    }
    if let Some(ms) = go.movetime {
        let adjusted = ms.saturating_sub(overhead_ms).max(1);
        return (TimeManager::from_movetime(adjusted), MAX_SEARCH_DEPTH);
    }
    // Incremental time control — pass overhead-adjusted time to the manager.
    let (time_ms, inc_ms) = match side {
        Color::White => (go.wtime.unwrap_or(30_000), go.winc),
        Color::Black => (go.btime.unwrap_or(30_000), go.binc),
    };
    let safe_ms = time_ms.saturating_sub(overhead_ms).max(1);
    (TimeManager::new(safe_ms, inc_ms, None), MAX_SEARCH_DEPTH)
}

// ── Engine factory ────────────────────────────────────────────────────────────

fn make_engine(nnue: &Option<Arc<Nnue>>) -> AlphaBetaEngine {
    match nnue {
        Some(nn) => AlphaBetaEngine::with_nnue(MAX_SEARCH_DEPTH, nn.clone()),
        None     => AlphaBetaEngine::with_depth(MAX_SEARCH_DEPTH),
    }
}

// ── Main UCI loop ─────────────────────────────────────────────────────────────

pub fn run(use_nnue: bool) {
    let stdin  = io::stdin();
    let stdout = io::stdout();

    // Try embedded weights first, then fall back to the runtime path from engine.toml.
    // Pass --no-nnue to force static evaluation regardless of available weights.
    // Prefer compile-time embedded weights (embed-nnue feature); fall back to
    // loading the file at runtime so plain `cargo build` still works during dev.
    let nnue: Option<Arc<Nnue>> = if use_nnue {
        Nnue::load_embedded()
            .or_else(Nnue::load_default)
            .map(Arc::new)
    } else {
        None
    };

    let mut engine         = make_engine(&nnue);
    let mut board          = Board::starting_position();
    let mut hash_history:    Vec<u64> = Vec::new();
    let mut move_overhead_ms: u64 = 30;

    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let line = line.trim();

        match line {
            "uci" => {
                let eval_label = if nnue.is_some() { "NNUE" } else { "static" };
                println!("id name rchess ({eval_label})");
                println!("id author augustin101");
                println!("option name Move Overhead type spin default 30 min 0 max 5000");
                println!("option name Hash type spin default 16 min 1 max 1024");
                println!("option name Threads type spin default 1 min 1 max 512");
                println!("option name MultiPV type spin default 1 min 1 max 500");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name UCI_ShowWDL type check default false");
                println!("option name UCI_Chess960 type check default false");
                println!("uciok");
            }
            "isready"    => println!("readyok"),
            "ucinewgame" => {
                engine       = make_engine(&nnue);
                board        = Board::starting_position();
                hash_history = Vec::new();
            }
            "quit" => break,
            "stop" => {}
            _ if line.starts_with("setoption") => {
                // setoption name <Name> value <Value>
                if let Some(rest) = line.strip_prefix("setoption name ") {
                    if let Some(val_str) = rest.strip_prefix("Move Overhead value ") {
                        if let Ok(v) = val_str.trim().parse::<u64>() {
                            move_overhead_ms = v;
                        }
                    }
                    // Hash option: ignored for now (TT size is fixed)
                }
            }
            _ if line.starts_with("position") => {
                if let Some((b, h)) = parse_position(line) {
                    board        = b;
                    hash_history = h;
                }
            }
            _ if line.starts_with("go") => {
                let legal = generate_legal(&board);
                let mv = if legal.len() == 1 {
                    Some(legal.as_slice()[0])
                } else {
                    let go = parse_go(line);
                    let (tm, max_depth) = create_time_manager(&go, board.side_to_move, move_overhead_ms);
                    engine.set_depth(max_depth);
                    engine.choose_move_timed(&board, tm, &hash_history)
                };
                let mv_str = mv.map_or_else(|| "0000".to_string(), |m| m.to_string());
                println!("bestmove {mv_str}");
                stdout.lock().flush().ok();
            }
            _ => {}
        }

        stdout.lock().flush().ok();
    }
}
