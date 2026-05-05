# rchess

A chess engine and web server written in Rust. Play against a computer opponent in your browser.

## Playing

```bash
cargo run --release
```

Then open **http://localhost:3000** in your browser. Click a piece to see legal moves, click a destination square to move. The engine replies automatically. Promotions default to queen; a picker appears for other choices.

You can choose to play as Black or restart the game at any time via the in-page controls.

## Architecture

```
src/
├── core/
│   ├── types.rs      – Square, Color, PieceType, Piece primitives
│   ├── bitboard.rs   – Bitboard newtype + shift helpers
│   ├── attacks.rs    – Pre-computed attack tables (magic bitboards for sliders)
│   ├── board.rs      – Hybrid board: bitboards + mailbox; FEN I/O; make/unmake
│   ├── moves.rs      – Move encoding (u16), MoveList (stack-allocated)
│   ├── movegen.rs    – Pseudo-legal + legal move generation
│   ├── san.rs        – Move → Standard Algebraic Notation
│   └── perft.rs      – Bulk-counting perft
├── engine/
│   ├── mod.rs        – Engine trait
│   └── random.rs     – RandomEngine (xorshift64 pick)
└── web/
    ├── mod.rs        – Axum router + game state (REST API)
    └── index.html    – Single-page frontend (vanilla JS)
```

### Board representation

`Board` keeps two views in sync via `put_piece` / `remove_piece`:
- **Bitboards** — `pieces[color][piece_type]`, `occupancy[color]`, `all_occupancy`
- **Mailbox** — `[Option<Piece>; 64]` for O(1) piece lookup by square

Squares are indexed a1=0 … h8=63 (bit N in a `Bitboard` = `Square(N)`).

### Attack generation

Slider attacks (rook, bishop, queen) use **magic bitboards**: each square has a `MagicEntry { mask, magic, shift, offset }` that maps an occupancy subset to an index into a flat pre-computed table (~102 K rook entries, ~5.3 K bishop entries). Leaper tables (pawn, knight, king) are plain arrays.

### Move encoding

`Move(u16)` packs from (6 bits), to (6 bits), promotion piece (2 bits), flag (2 bits: Normal / Promo / EnPassant / Castling). `MoveList` is a stack-allocated `[Move; 256]` — no heap allocation in the hot path.

### Engine

The current engine (`RandomEngine`) picks a uniformly random legal move. The `Engine` trait makes it straightforward to swap in a stronger implementation.

## Development

```bash
cargo build                       # debug build
cargo test                        # run fast tests
cargo test -- --include-ignored   # include slow perft depths (≥5)
cargo test perft_start_d4         # run a specific perft test
```

## Requirements

- Rust 1.85+ (edition 2024)
