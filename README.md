# rchess

A personal chess engine written in Rust. Includes a browser-based UI to play against it locally and a UCI-compatible binary for chess GUIs or bot frameworks like [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot).

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

# with NNUE weights embedded (recommended for distribution):
cargo build --release --bin rchess-uci --features embed-nnue
```

## Tests

```bash
cargo test                        # run all fast tests
cargo test -- --include-ignored   # include slow perft depths (≥5)
cargo test perft_start_d4         # run a specific perft test by name
```

## Source layout

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
│   ├── nnue.rs         – Quantised dual-perspective NNUE with incremental
│   │                     accumulator and AVX2/NEON SIMD inference
│   ├── time_manager.rs – Soft/hard time limits with panic mode
│   ├── opening_book.rs – Polyglot opening book reader
│   └── random.rs       – RandomEngine (used for testing)
├── uci.rs              – UCI protocol interface
└── web/
    ├── mod.rs          – Axum REST API + game state
    └── index.html      – Single-page browser UI (vanilla JS)
training/
├── model.py            – NNUE PyTorch model (768→256→512→32→32→1)
├── dataset.py          – Binpack dataset loader
├── build_binpack.py    – Parquet → compact binary records
├── shuffle_split.py    – Train/val/test split
├── train.py            – Training loop with live Rich display
├── export.py           – Quantise float weights → .bin for the engine
├── validate.py         – Validation metrics
└── match.py            – NNUE vs static eval match runner
```

## Board representation

`Board` keeps two views in sync via `put_piece` / `remove_piece`:
- **Bitboards** — `pieces[color][piece_type]`, `occupancy[color]`, `all_occupancy`
- **Mailbox** — `[Option<Piece>; 64]` for O(1) piece lookup by square

Squares are indexed a1=0 … h8=63 (bit N in a `Bitboard` = `Square(N)`).

## Move generation

Slider attacks (rook, bishop, queen) use **magic bitboards**: each square has a `MagicEntry { mask, magic, shift, offset }` that maps a blockers bitboard to a pre-computed attacks entry in O(1). Leaper tables (pawn, knight, king) are plain arrays. `generate_pseudo_legal` produces geometrically valid moves; `generate_legal` filters them by verifying the king is not left in check.

## Search

Iterative-deepening alpha-beta with:
- Transposition table (1 M entries, ~24 MB)
- Aspiration windows (±50 cp, widening on failure)
- Move ordering: TT move → promotions → MVV-LVA captures → killers → history
- Null-move pruning (R=2/3)
- Futility pruning at depth 1–2
- Late Move Reductions (LMR)
- Check extensions
- Quiescence search
- Threefold-repetition and 50-move rule detection inside search
- Soft/hard time limits with panic mode

## Evaluation

Tapered evaluation interpolating between middlegame (MG) and endgame (EG) scores:
- Material + piece-square tables
- Pawn structure (doubled, isolated, islands, passed pawns)
- Rook bonuses (open/semi-open files, 7th rank)
- Bishop pair
- King safety (pawn shield, open-file exposure)
- Mobility

When NNUE weights are available the engine uses a neural network evaluation instead (see below).

## NNUE

A dual-perspective quantised NNUE (768→256→512→32→32→1) trained on Lichess evaluation data. Weights are embedded at compile time with `--features embed-nnue` and updated incrementally during search.

### Setup

```bash
pip install -r training/requirements.txt
```

### Data

Download the [Lichess position evaluation dataset](https://huggingface.co/datasets/Lichess/chess-position-evaluations) parquet files into `data/parquet/`.

### Pipeline (run from project root)

```bash
# 1. Filter & encode positions (~55 min full run)
python training/build_binpack.py
# smoke test: python training/build_binpack.py --max_rows 1000

# 2. Shuffle & split into train / val / test
python training/shuffle_split.py

# 3. Train
python training/train.py --splits data/binpack/splits.json
python training/train.py --splits data/binpack/splits.json --epochs 10 --batch 8192 --lr 1e-3

# 4. Export quantised weights
python training/export.py                          # best.pt → networks/nnue.bin
python training/export.py --ckpt checkpoints/epoch_03.pt --out networks/nnue.bin

# 5. Match NNUE vs static eval
python training/match.py --games 20
```

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
├── resume.pt       – full state for resuming training
└── plots/          – loss and LR curve PNGs
networks/
└── nnue.bin        – exported quantised weights (loaded by the engine)
```

## Documentation

Detailed technical documentation lives in `docs/`:

| File | Contents |
|---|---|
| [`docs/search.rst`](docs/search.rst) | Alpha-beta, move ordering, pruning, time management |
| [`docs/eval.rst`](docs/eval.rst) | Tapered static evaluation, all terms and tuning values |
| [`docs/nnue.rst`](docs/nnue.rst) | NNUE architecture, training loss, quantisation maths, SIMD inference |

## Requirements

Rust 1.85+ (edition 2024), Python 3.10+ with dependencies from `training/requirements.txt`.

## License

MIT — see [LICENSE](LICENSE)
