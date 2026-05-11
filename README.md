# rchess

A small personal chess engine written in Rust. It includes a browser-based UI to play against it locally, and a UCI-compatible binary for use with chess GUIs or bot frameworks like [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot).

## Running

**Web UI** — play against the engine in your browser:
```bash
cargo run --release
# then open http://localhost:3000
```

**UCI engine** — for chess GUIs or lichess-bot:
```bash
cargo build --release --bin rchess-uci
# binary at target/release/rchess-uci
```

## Tests

```bash
cargo test                        # run all fast tests
cargo test -- --include-ignored   # include slow perft depths (≥5)
cargo test perft_start_d4         # run a specific perft test by name
```

## Architecture

```
src/
├── core/
│   ├── types.rs        – Square, Color, PieceType, Piece primitives
│   ├── bitboard.rs     – Bitboard newtype + shift/pop helpers
│   ├── attacks.rs      – Pre-computed attack tables (magic bitboards for sliders)
│   ├── board.rs        – Hybrid board: bitboards + mailbox; FEN I/O; make/unmake
│   ├── moves.rs        – Move encoding (u16), MoveList (stack-allocated)
│   ├── movegen.rs      – Pseudo-legal + legal move generation
│   ├── zobrist.rs      – Incremental Zobrist hashing
│   ├── san.rs          – Move → Standard Algebraic Notation
│   └── perft.rs        – Bulk-counting perft for move generation testing
├── engine/
│   ├── mod.rs          – Engine trait
│   ├── eval.rs         – Tapered static evaluation (material, PST, pawn structure,
│   │                     mobility, king safety, rook bonuses, bishop pair)
│   ├── pst.rs          – Piece-square tables (middlegame + endgame)
│   ├── alpha_beta.rs   – Iterative-deepening alpha-beta with TT, killers,
│   │                     history, null-move, futility pruning, LMR, quiescence
│   ├── opening_book.rs – Polyglot opening book reader
│   └── random.rs       – RandomEngine (used for testing)
├── uci.rs              – UCI protocol interface
└── web/
    ├── mod.rs          – Axum REST API + game state
    └── index.html      – Single-page browser UI (vanilla JS)
```

### Board representation

`Board` keeps two views in sync via `put_piece` / `remove_piece`:
- **Bitboards** — `pieces[color][piece_type]`, `occupancy[color]`, `all_occupancy`
- **Mailbox** — `[Option<Piece>; 64]` for O(1) piece lookup by square

Squares are indexed a1=0 … h8=63 (bit N in a `Bitboard` = `Square(N)`).

### Move generation

Slider attacks (rook, bishop, queen) use **magic bitboards**: each square has a `MagicEntry { mask, magic, shift, offset }` indexing into a flat pre-computed table (~102 K rook entries, ~5.3 K bishop entries). Leaper tables (pawn, knight, king) are plain arrays. `generate_pseudo_legal` produces geometrically valid moves; `generate_legal` filters them by verifying the king is not left in check.

### Move encoding

`Move(u16)` packs from-square (6 bits), to-square (6 bits), promotion piece (2 bits), and flag (2 bits: Normal / Promo / EnPassant / Castling). `MoveList` is a stack-allocated `[Move; 256]` — no heap allocation in the hot path.

### Search

Iterative-deepening alpha-beta with:
- Transposition table (1M entries, ~24 MB)
- Move ordering: TT move → promotions → MVV-LVA captures → killer moves → history heuristic
- Null-move pruning
- Futility pruning (depth 1–2)
- Late Move Reductions (LMR)
- Check extensions
- Quiescence search

### Evaluation

Tapered evaluation interpolating between middlegame and endgame scores based on remaining material:
- Material + piece-square tables
- Pawn structure (doubled, isolated, islands, passed pawns)
- Rook bonuses (open/semi-open files, 7th rank)
- Bishop pair
- King safety (pawn shield, open file exposure)
- Mobility (squares attacked per piece type)

## Requirements

Rust 1.85+ (edition 2024)

## License

MIT — see [LICENSE](LICENSE)
