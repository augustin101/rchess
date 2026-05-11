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

## NNUE training

The engine supports an optional NNUE evaluation. Trained weights live in `networks/nnue.bin` and are loaded automatically at startup if present.

### Setup

```bash
pip install -r training/requirements.txt
```

### Data

Download the [Lichess position evaluation dataset](https://huggingface.co/datasets/Lichess/chess-position-evaluations) parquet files into `data/parquet/`.

### Pipeline (run everything from the project root)

**1. Filter & encode** — converts parquet files to compact 196-byte binary records:
```bash
# Full run (~55 min, uses all CPU cores)
python training/build_binpack.py

# Smoke test — 1 000 rows per file, completes in ~30 s
python training/build_binpack.py --max_rows 1000
```
Output: `data/binpack/shards/shard_00.bin` … `shard_16.bin`

**2. Shuffle & split** — creates fixed validation/test sets and shuffled train shards:
```bash
python training/shuffle_split.py
```
Output: `data/binpack/val.bin`, `data/binpack/test.bin`, `data/binpack/train/`  
The validation set is sampled once with a fixed seed and never changes between runs.

**3. Train**:
```bash
python training/train.py --splits data/binpack/splits.json

# Custom hyperparameters
python training/train.py --splits data/binpack/splits.json \
    --epochs 10 --batch 8192 --lr 1e-3
```
Checkpoints are saved to `checkpoints/`. Loss and learning-rate plots are written to `checkpoints/plots/latest.png` after each epoch.

**4. Export** — quantises float weights to the binary format read by the Rust engine:
```bash
python training/export.py                                    # best checkpoint
python training/export.py checkpoints/epoch_03.pt networks/nnue.bin
```
Output: `networks/nnue.bin` — picked up automatically by `cargo run`.

### Directory layout

```
data/
├── parquet/        – source parquet files (input, not modified)
└── binpack/
    ├── shards/     – one .bin shard per parquet file (build_binpack output)
    ├── train/      – shuffled train shards (shuffle_split output)
    ├── val.bin     – fixed validation set
    ├── test.bin    – held-out test set
    └── splits.json – manifest used by train.py
checkpoints/
├── best.pt         – best checkpoint by validation loss
├── epoch_NN.pt     – per-epoch checkpoints
└── plots/          – loss and LR curve PNGs
networks/
└── nnue.bin        – exported quantised weights (loaded by the engine)
training/
├── build_binpack.py
├── shuffle_split.py
├── train.py
├── export.py
├── model.py
└── dataset.py
```

## Requirements

Rust 1.85+ (edition 2024), Python 3.10+ with dependencies from `training/requirements.txt`.

## License

MIT — see [LICENSE](LICENSE)
