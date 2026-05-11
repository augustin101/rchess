"""
Shuffle shard files and split into train / validation / test sets.

Validation is sampled once with a fixed seed and written to val.bin.
It never changes between training runs.

Usage:
  python training/shuffle_split.py

  # Custom sizes
  python training/shuffle_split.py --val_count 500000 --test_count 200000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from build_binpack import RECORD_DTYPE, RECORD_SIZE


def load_shard(path: Path) -> np.ndarray:
    """Memory-map a shard file for zero-copy reading."""
    n = path.stat().st_size // RECORD_SIZE
    return np.memmap(path, dtype=RECORD_DTYPE, mode='r', shape=(n,))


def write_array(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        f.write(arr.tobytes())
    tmp.rename(path)


def split(args: argparse.Namespace) -> None:
    shard_dir = Path(args.shard_dir)
    out_dir   = Path(args.out_dir)

    shard_paths = sorted(shard_dir.glob('shard_*.bin'))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.bin files in {shard_dir}")

    # Shard sizes
    shard_sizes = [p.stat().st_size // RECORD_SIZE for p in shard_paths]
    total       = sum(shard_sizes)
    print(f"Shards      : {len(shard_paths)}")
    print(f"Total       : {total:,} records  ({total * RECORD_SIZE / 1e9:.1f} GB)")

    # ── Sample validation set (fixed seed, never changes) ─────────────────────
    # We take val_per_shard records from each shard proportionally so that val
    # spans the full distribution.  The global seed guarantees reproducibility.
    val_total  = min(args.val_count,  total // 10)
    test_total = min(args.test_count, total // 20)

    rng = np.random.default_rng(args.seed)

    val_arrays  = []
    test_arrays = []
    train_sizes = []

    print(f"\nExtracting val ({val_total:,}) and test ({test_total:,}) samples …")

    for i, (path, size) in enumerate(tqdm(
        zip(shard_paths, shard_sizes), total=len(shard_paths), unit='shard'
    )):
        shard = load_shard(path)

        # Proportional counts
        frac       = size / total
        n_val      = max(1, round(val_total  * frac))
        n_test     = max(1, round(test_total * frac))
        n_holdout  = n_val + n_test

        # Randomly select holdout indices (seed is deterministic per shard)
        perm     = rng.permutation(size)
        val_idx  = perm[:n_val]
        test_idx = perm[n_val:n_val + n_test]
        train_n  = size - n_holdout

        val_arrays.append(np.array(shard[val_idx]))
        test_arrays.append(np.array(shard[test_idx]))
        train_sizes.append((path, perm[n_holdout:], train_n))

    # Concatenate and shuffle val/test globally
    val_data  = np.concatenate(val_arrays)
    test_data = np.concatenate(test_arrays)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    val_path  = out_dir / 'val.bin'
    test_path = out_dir / 'test.bin'
    write_array(val_data,  val_path)
    write_array(test_data, test_path)
    print(f"  val.bin  : {len(val_data):,} records → {val_path}")
    print(f"  test.bin : {len(test_data):,} records → {test_path}")

    # ── Write train shards (in-place shuffle of remaining indices) ─────────────
    train_dir = out_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(train_sizes)} shuffled train shards …")

    train_total = 0
    for idx, (path, train_perm, n_train) in enumerate(tqdm(
        train_sizes, unit='shard'
    )):
        shard     = load_shard(path)
        rng.shuffle(train_perm)                 # shuffle within shard
        train_arr = np.array(shard[train_perm]) # copy to RAM
        out_path  = train_dir / f'train_{idx:02d}.bin'
        write_array(train_arr, out_path)
        train_total += n_train

    print(f"  {train_total:,} train records in {train_dir}")

    # ── Write splits manifest ──────────────────────────────────────────────────
    manifest = {
        'val_path':   str(val_path),
        'test_path':  str(test_path),
        'train_dir':  str(train_dir),
        'n_val':      int(len(val_data)),
        'n_test':     int(len(test_data)),
        'n_train':    int(train_total),
        'seed':       args.seed,
        'record_size': RECORD_SIZE,
    }
    mpath = out_dir / 'splits.json'
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest    : {mpath}")
    print(f"Total after split: train={train_total:,}  val={len(val_data):,}  test={len(test_data):,}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard_dir',  default='data/binpack/shards')
    p.add_argument('--out_dir',    default='data/binpack')
    p.add_argument('--val_count',  type=int, default=2_000_000,
                   help='Target total validation records')
    p.add_argument('--test_count', type=int, default=1_000_000,
                   help='Target total test records')
    p.add_argument('--seed',       type=int, default=42,
                   help='Fixed seed — determines val set (never change this)')
    split(p.parse_args())
