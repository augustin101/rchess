# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build                          # compile
cargo test                           # run all tests (fast perft depths only)
cargo test -- --include-ignored      # run all tests including slow perft depths
cargo test <name>                    # run a single test by name substring
cargo run                            # interactive FEN visualiser + bitboard demo
```

To run a specific perft depth test:
```bash
cargo test perft_start_d4
cargo test perft_kiwipete_d3
```

## Architecture

### Hybrid board representation (`src/board.rs`)
`Board` maintains two views simultaneously, kept in sync by `put_piece` / `remove_piece`:
- **Bit-centric**: `pieces[color][piece_type]: [[Bitboard; 6]; 2]` plus `occupancy[2]` (per-color) and `all_occupancy`.
- **Board-centric (mailbox)**: `mailbox: [Option<Piece>; 64]` — O(1) piece lookup by square.

`make_move(mv) -> IrreversibleState` and `unmake_move(mv, state)` mutate the board and return/accept the snapshot of fields that can't be derived from the move alone (captured piece, castling rights, en passant, halfmove clock).

### Square indexing
`Square(u8)`: a1=0, b1=1, …, h1=7, a2=8, …, h8=63. Bit N in a `Bitboard` corresponds to `Square(N)`.

### Attack tables (`src/attacks.rs`)
`OnceLock<Attacks>` initialised on first use. Contains:
- Lookup arrays for pawns (×2 colors), knights, kings.
- **Magic bitboards** for rooks and bishops: per-square `MagicEntry { mask, magic, shift, offset }` indexing into a flat `Vec<Bitboard>` (~102 K rook entries, ~5.3 K bishop entries). Magic constants are embedded `const` arrays. Queen attacks = rook | bishop.

### Move encoding (`src/moves.rs`)
`Move(u16)`: bits 0–5 from, 6–11 to, 12–13 promo piece (N/B/R/Q=0–3), 14–15 flag (Normal/Promo/EnPassant/Castling=0–3). `MoveList` is a stack-allocated `[Move; 256]` + len — no heap allocation in the hot path.

### Move generation (`src/movegen.rs`)
`generate_pseudo_legal` — geometrically valid moves, may leave king in check.
`generate_legal` — filters via `make_move → is_attacked_by(king_sq, them) → unmake_move`. Castling legality (transit squares unattacked) is pre-checked inside pseudo-legal generation.

### Castling rights update
`CASTLING_RIGHTS_MASK: [u8; 64]` — index by `from` and `to` squares and AND both values into the rights unconditionally on every move. Handles king moves, rook moves, and rook captures with no branching.

### Perft (`src/perft.rs`, `tests/perft_tests.rs`)
`perft(board, depth)` uses bulk-counting at depth 1. Depths ≥5 are `#[ignore]`d; run with `--include-ignored`.

### Correct perft values
`move_gen_tests.md` has values:
- Start d3=8902, d4=197281, d5=4865609, d6=119060324
- Kiwipete d1=48, d2=2039, d3=97862, d4=4085603, d5=193690690
- Position 3 d3=2812, d4=43238, d5=674624, d6=11030083
