# NNUE Training

All commands are run from the **project root** (the folder containing `Cargo.toml`), not from inside `training/`.

## Setup

```bash
pip install -r training/requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. A CUDA GPU is recommended for full-scale runs.

## Data

Place the Lichess position evaluation parquet files in `data/parquet/`.

## Pipeline

### 1. Filter & encode

Reads all parquet files, applies filters (`depth ≥ 16`, `|cp| ≤ 3000`, no mate scores), and writes compact 196-byte binary records to `data/binpack/shards/`.

```bash
# Full run — all 17 files, ~55 min on 22 CPUs
python training/build_binpack.py

# Smoke test — 1 000 rows per file, completes in ~30 s
python training/build_binpack.py --max_rows 1000
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/parquet` | Source parquet directory |
| `--out_dir` | `data/binpack` | Output directory |
| `--max_rows` | _(all)_ | Limit rows per file (smoke test) |
| `--workers` | _(all CPUs)_ | Parallel worker processes |

### 2. Shuffle & split

Samples a fixed validation set and test set from the shards, then writes shuffled train shards.

```bash
python training/shuffle_split.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--shard_dir` | `data/binpack/shards` | Input shard directory |
| `--out_dir` | `data/binpack` | Output directory |
| `--val_count` | `2 000 000` | Validation set size |
| `--test_count` | `1 000 000` | Test set size |
| `--seed` | `42` | Fixed seed — determines val set, never change |

Output:
```
data/binpack/
├── val.bin         ← fixed forever (seed 42)
├── test.bin
├── train/
│   ├── train_00.bin
│   └── …
└── splits.json     ← manifest read by train.py
```

### 3. Train

```bash
python training/train.py --splits data/binpack/splits.json

# Custom hyperparameters
python training/train.py --splits data/binpack/splits.json \
    --epochs 10 --batch 8192 --lr 1e-3
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--splits` | `data/binpack/splits.json` | Manifest from shuffle_split.py |
| `--epochs` | `5` | Training epochs |
| `--batch` | `4096` | Batch size |
| `--lr` | `1e-3` | Initial learning rate (cosine decay) |
| `--workers` | `8` | DataLoader worker processes |
| `--log_steps` | `500` | Print loss every N steps |

Output:
```
checkpoints/
├── best.pt          ← best checkpoint by validation loss
├── epoch_01.pt … epoch_NN.pt
└── plots/
    ├── latest.png   ← updated after every epoch
    └── epoch_NN.png
```

### 4. Export

Quantises the trained float weights to the binary format read by the Rust engine.

```bash
# Best checkpoint → networks/nnue.bin  (default)
python training/export.py

# Specific checkpoint
python training/export.py checkpoints/epoch_03.pt networks/nnue.bin
```

Output: `networks/nnue.bin` — picked up automatically when you run `cargo run`.

## Full pipeline summary

```
python training/build_binpack.py          →  data/binpack/shards/
python training/shuffle_split.py          →  data/binpack/{val,test,train/,splits.json}
python training/train.py                  →  checkpoints/{best.pt, plots/}
python training/export.py                 →  networks/nnue.bin
cargo run                                 →  loads networks/nnue.bin, NNUE toggle in UI
```

## Architecture

Dual-perspective NNUE:

```
768 (features, White POV) ──┐
                             ├─ FT (shared weights) → 256 → ClipReLU
768 (features, Black POV) ──┘

[stm_acc | opp_acc] → 512 → L1 → 32 → ClipReLU → L2 → 32 → ClipReLU → output → 1
```

Feature index:
- White POV: `piece_type × 128 + color × 64 + square`
- Black POV: `piece_type × 128 + (1-color) × 64 + (square ^ 56)`

Training target: `tanh(cp_stm / 400)` where `cp_stm` is centipawns from the side-to-move's perspective.

## Record format (196 bytes)

| Field | Type | Bytes | Description |
|-------|------|-------|-------------|
| `cp` | i16 | 2 | Centipawns from White's POV |
| `stm` | u8 | 1 | 0 = White, 1 = Black to move |
| `_pad` | u8 | 1 | Alignment |
| `wbits` | u8 × 96 | 96 | Packed 768-bit feature bitmap, White POV |
| `bbits` | u8 × 96 | 96 | Packed 768-bit feature bitmap, Black POV |

## Files

| File | Purpose |
|------|---------|
| `build_binpack.py` | Filter parquet → binary shards |
| `shuffle_split.py` | Shuffle shards, create val/test/train split |
| `train.py` | Training loop with live plots |
| `export.py` | Quantise and export weights to `networks/nnue.bin` |
| `model.py` | NNUE architecture (PyTorch) |
| `dataset.py` | `BinpackIterableDataset`, GPU bitmap unpacking |
| `requirements.txt` | Python dependencies |
