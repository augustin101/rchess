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
use crate::engine::random::RandomEngine;
use crate::engine::alpha_beta::AlphaBetaEngine;
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
    board:       Board,
    history:     Vec<MoveEntry>,
    last_move:   Option<(u8, u8)>,
    status:      Status,
    human_color: Color,
    engine:      Box<dyn Engine + Send>,
}

impl Game {
    fn new(human_color: Color) -> Self {
        Self::with_engine(human_color, Box::new(RandomEngine::new()))
    }

    /// Creates a new game with a custom engine
    fn with_engine(human_color: Color, engine: Box<dyn Engine + Send>) -> Self {
        Game {
            board:       Board::starting_position(),
            history:     Vec::new(),
            last_move:   None,
            status:      Status::Playing,
            human_color,
            engine,
        }
    }

    fn apply(&mut self, mv: Move) {
        let san = move_to_san(&self.board, mv);
        self.last_move = Some((mv.from_sq().0, mv.to_sq().0));
        self.history.push(MoveEntry { uci: mv.to_string(), san });
        self.board.make_move(mv);
        self.refresh_status();
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
}

#[derive(Deserialize)]
struct MoveReq { from: u8, to: u8, promo: Option<String> }

#[derive(Deserialize)]
struct RestartReq { human_color: Option<String>, engine: Option<String> }

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

async fn api_restart(State(g): State<Shared>, Json(req): Json<RestartReq>) -> Json<GameState> {
    let color = match req.human_color.as_deref() {
        Some("black") => Color::Black,
        _             => Color::White,
    };
    let engine = match req.engine.as_deref() {
        Some("random") => Box::new(RandomEngine::new()) as Box<dyn Engine + Send>,
        _                 => Box::new(AlphaBetaEngine::new(4)) as Box<dyn Engine + Send>,   
    };

    *g.lock().unwrap() = Game::with_engine(color, engine);
    Json(g.lock().unwrap().to_response())
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
        .route("/api/resign",      post(api_resign))
        .with_state(state)
}
