use axum::{
    extract::State,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::core::board::Board;
use crate::core::movegen::generate_legal;
use crate::core::moves::{Move, PromoKind};
use crate::core::san::move_to_san;
use crate::core::types::{Color, PieceType, Square};
use crate::engine::eval::{static_eval, CHECKMATE_SCORE};
use crate::engine::random::RandomEngine;
use crate::engine::alpha_beta::AlphaBetaEngine;
use crate::engine::opening_book::{BookEngine, OpeningBook};
use crate::engine::Engine;

// ── Game state ────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum Status {
    Playing,
    Checkmate { winner: Color },
    Stalemate,
    Resigned  { loser: Color },
}

struct Game {
    board:         Board,
    history:       Vec<MoveEntry>,
    board_history: Vec<Board>,
    last_move:     Option<(u8, u8)>,
    status:        Status,
    human_color:   Color,
    engine:        Box<dyn Engine + Send>,
    engine_type:   String,
    depth:         u32,
    use_book:      bool,
    randomness:    u8,
}

fn make_engine(engine_type: &str, depth: u32, use_book: bool, randomness: u8) -> Box<dyn Engine + Send> {
    let book = if use_book { Some(OpeningBook::load_default()) } else { None };
    match (engine_type, book) {
        ("random", Some(b)) => Box::new(BookEngine::new(b, RandomEngine::new(), randomness)),
        ("random", None)    => Box::new(RandomEngine::new()),
        (_,        Some(b)) => Box::new(BookEngine::new(b, AlphaBetaEngine::with_depth(depth), randomness)),
        (_,        None)    => Box::new(AlphaBetaEngine::with_depth(depth)),
    }
}

impl Game {
    fn new(human_color: Color) -> Self {
        Self::from_board(human_color, "alpha-beta", 5, true, 30, Board::starting_position())
    }

    fn from_board(human_color: Color, engine_type: &str, depth: u32, use_book: bool, randomness: u8, board: Board) -> Self {
        let mut game = Game {
            board,
            history:       Vec::new(),
            board_history: Vec::new(),
            last_move:     None,
            status:        Status::Playing,
            human_color,
            engine:        make_engine(engine_type, depth, use_book, randomness),
            engine_type:   engine_type.to_string(),
            depth,
            use_book,
            randomness,
        };
        game.refresh_status();
        game
    }

    fn apply(&mut self, mv: Move) {
        self.board_history.push(self.board.clone());
        let san = move_to_san(&self.board, mv);
        self.last_move = Some((mv.from_sq().0, mv.to_sq().0));
        self.history.push(MoveEntry { uci: mv.to_string(), san });
        self.board.make_move(mv);
        self.refresh_status();
    }

    fn undo(&mut self) {
        if self.board_history.is_empty() { return; }
        // Undo 2 plies (engine + human); 1 ply if only 1 available.
        let plies = if self.board_history.len() >= 2 { 2 } else { 1 };
        for _ in 0..plies {
            if let Some(board) = self.board_history.pop() {
                self.board = board;
                self.history.pop();
            }
        }
        self.last_move = self.history.last().map(|e| {
            let b = e.uci.as_bytes();
            let from = (b[1] - b'1') * 8 + (b[0] - b'a');
            let to   = (b[3] - b'1') * 8 + (b[2] - b'a');
            (from, to)
        });
        self.status = Status::Playing;
    }

    fn refresh_status(&mut self) {
        if !matches!(self.status, Status::Playing) { return; }
        let legal = generate_legal(&self.board);
        if legal.is_empty() {
            self.status = if self.board.is_in_check() {
                Status::Checkmate { winner: self.board.side_to_move.flip() }
            } else {
                Status::Stalemate
            };
        }
    }

    fn eval(&self) -> i32 {
        match &self.status {
            Status::Checkmate { winner } => if *winner == Color::White { CHECKMATE_SCORE } else { -CHECKMATE_SCORE },
            Status::Stalemate            => 0,
            Status::Resigned  { loser }  => if *loser == Color::White  { -CHECKMATE_SCORE } else { CHECKMATE_SCORE },
            Status::Playing              => static_eval(&self.board),
        }
    }

    fn to_response(&self) -> GameState {
        let board = (0u8..64).map(|sq| {
            self.board.piece_at(Square(sq)).map(|p| PieceInfo {
                color:      color_str(p.color),
                piece_type: pt_str(p.piece_type),
            })
        }).collect();

        let legal_moves = if matches!(self.status, Status::Playing) {
            generate_legal(&self.board).as_slice().iter().map(|m| m.to_string()).collect()
        } else {
            vec![]
        };

        let (status, winner) = match &self.status {
            Status::Playing              => ("playing".into(),   None),
            Status::Checkmate { winner } => ("checkmate".into(), Some(color_str(*winner))),
            Status::Stalemate            => ("stalemate".into(), None),
            Status::Resigned  { loser }  => ("resigned".into(),  Some(color_str(loser.flip()))),
        };

        GameState {
            board,
            legal_moves,
            history:      self.history.clone(),
            status,
            winner,
            in_check:     self.board.is_in_check(),
            side_to_move: color_str(self.board.side_to_move),
            human_color:  color_str(self.human_color),
            last_move:    self.last_move,
            engine_name:  self.engine.name().to_string(),
            engine_type:  self.engine_type.clone(),
            depth:        self.depth,
            use_book:     self.use_book,
            randomness:   self.randomness,
            eval:         self.eval(),
            fen:          self.board.to_fen(),
        }
    }
}

fn color_str(c: Color) -> String {
    match c { Color::White => "white".into(), Color::Black => "black".into() }
}

fn pt_str(pt: PieceType) -> String {
    match pt {
        PieceType::Pawn   => "pawn",
        PieceType::Knight => "knight",
        PieceType::Bishop => "bishop",
        PieceType::Rook   => "rook",
        PieceType::Queen  => "queen",
        PieceType::King   => "king",
    }.into()
}

// ── Serde types ───────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct MoveEntry { uci: String, san: String }

#[derive(Serialize)]
struct PieceInfo { color: String, piece_type: String }

#[derive(Serialize)]
struct GameState {
    board:        Vec<Option<PieceInfo>>,
    legal_moves:  Vec<String>,
    history:      Vec<MoveEntry>,
    status:       String,
    winner:       Option<String>,
    in_check:     bool,
    side_to_move: String,
    human_color:  String,
    last_move:    Option<(u8, u8)>,
    engine_name:  String,
    engine_type:  String,
    depth:        u32,
    use_book:     bool,
    randomness:   u8,
    eval:         i32,
    fen:          String,
}

#[derive(Deserialize)]
struct MoveReq { from: u8, to: u8, promo: Option<String> }

#[derive(Deserialize)]
struct RestartReq {
    human_color: Option<String>,
    engine:      Option<String>,
    depth:       Option<u32>,
    use_book:    Option<bool>,
    randomness:  Option<u32>,
}

#[derive(Deserialize)]
struct LoadFenReq {
    fen:         String,
    human_color: Option<String>,
    engine:      Option<String>,
    depth:       Option<u32>,
    use_book:    Option<bool>,
    randomness:  Option<u32>,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

type Shared = Arc<Mutex<Game>>;

async fn index() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

async fn api_state(State(g): State<Shared>) -> Json<GameState> {
    Json(g.lock().unwrap().to_response())
}

async fn api_move(State(g): State<Shared>, Json(req): Json<MoveReq>) -> Json<GameState> {
    let mut game = g.lock().unwrap();
    if !matches!(game.status, Status::Playing) || game.board.side_to_move != game.human_color {
        return Json(game.to_response());
    }

    let promo = req.promo.as_deref().and_then(|s| match s {
        "q" => Some(PromoKind::Queen),  "r" => Some(PromoKind::Rook),
        "b" => Some(PromoKind::Bishop), "n" => Some(PromoKind::Knight),
        _   => None,
    });
    let from = Square(req.from);
    let to   = Square(req.to);

    let mv = generate_legal(&game.board).as_slice().iter().find(|&&mv| {
        mv.from_sq() == from && mv.to_sq() == to
            && match promo {
                Some(pk) => mv.is_promo() && mv.promo_kind() == pk,
                None     => !mv.is_promo(),
            }
    }).copied();

    if let Some(mv) = mv { game.apply(mv); }
    Json(game.to_response())
}

async fn api_engine_move(State(g): State<Shared>) -> Json<GameState> {
    let mut game = g.lock().unwrap();
    if !matches!(game.status, Status::Playing) || game.board.side_to_move == game.human_color {
        return Json(game.to_response());
    }
    let board = game.board.clone();
    if let Some(mv) = game.engine.choose_move(&board) { game.apply(mv); }
    Json(game.to_response())
}

fn parse_color(s: Option<&str>) -> Color {
    match s { Some("black") => Color::Black, _ => Color::White }
}

fn parse_engine_type(s: Option<&str>) -> &'static str {
    match s { Some("random") => "random", _ => "alpha-beta" }
}

fn parse_depth(d: Option<u32>) -> u32 {
    d.unwrap_or(5).clamp(1, 8)
}

fn parse_randomness(r: Option<u32>) -> u8 {
    r.unwrap_or(30).min(100) as u8
}

async fn api_restart(State(g): State<Shared>, Json(req): Json<RestartReq>) -> Json<GameState> {
    let color       = parse_color(req.human_color.as_deref());
    let engine_type = parse_engine_type(req.engine.as_deref());
    let depth       = parse_depth(req.depth);
    let use_book    = req.use_book.unwrap_or(true);
    let randomness  = parse_randomness(req.randomness);
    let mut game    = g.lock().unwrap();
    *game = Game::from_board(color, engine_type, depth, use_book, randomness, Board::starting_position());
    Json(game.to_response())
}

async fn api_load_fen(State(g): State<Shared>, Json(req): Json<LoadFenReq>) -> Json<GameState> {
    let color       = parse_color(req.human_color.as_deref());
    let engine_type = parse_engine_type(req.engine.as_deref());
    let depth       = parse_depth(req.depth);
    let use_book    = req.use_book.unwrap_or(true);
    let randomness  = parse_randomness(req.randomness);
    let mut game    = g.lock().unwrap();
    if let Ok(board) = Board::from_fen(&req.fen) {
        *game = Game::from_board(color, engine_type, depth, use_book, randomness, board);
    }
    Json(game.to_response())
}

async fn api_undo(State(g): State<Shared>) -> Json<GameState> {
    let mut game = g.lock().unwrap();
    game.undo();
    Json(game.to_response())
}

async fn api_resign(State(g): State<Shared>) -> Json<GameState> {
    let mut game = g.lock().unwrap();
    if matches!(game.status, Status::Playing) {
        game.status = Status::Resigned { loser: game.human_color };
    }
    Json(game.to_response())
}

// ── Public router ─────────────────────────────────────────────────────────────

pub fn router() -> Router {
    let state: Shared = Arc::new(Mutex::new(Game::new(Color::White)));
    Router::new()
        .route("/",                get(index))
        .route("/api/state",       get(api_state))
        .route("/api/move",        post(api_move))
        .route("/api/engine-move", post(api_engine_move))
        .route("/api/restart",     post(api_restart))
        .route("/api/load-fen",    post(api_load_fen))
        .route("/api/undo",        post(api_undo))
        .route("/api/resign",      post(api_resign))
        .with_state(state)
}
