"""
Filter Lichess parquet files and convert to a compact binpack format.

Record format — 196 bytes, numpy-friendly:
  cp    : int16   (2 bytes)  centipawns from White's perspective
  stm   : uint8   (1 byte)   0=White, 1=Black to move
  _pad  : uint8   (1 byte)   alignment padding
  wbits : uint8×96 (96 bytes) 768-bit packed feature bitmap, White POV
  bbits : uint8×96 (96 bytes) 768-bit packed feature bitmap, Black POV

Feature index (both perspectives must match src/engine/nnue.rs):
  White POV : feat = piece_type*128 + color*64 + square
  Black POV : feat = piece_type*128 + (1-color)*64 + (square^56)

Usage:
  # Full run (all 17 files, ~55 min with 22 CPUs)
  python training/build_binpack.py

  # Quick smoke-test (1000 rows per file, <30 s)
  python training/build_binpack.py --max_rows 1000
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import chess
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Record dtype ──────────────────────────────────────────────────────────────

RECORD_DTYPE = np.dtype([
    ('cp',    '<i2'),
    ('stm',   'u1'),
    ('_pad',  'u1'),
    ('wbits', 'u1', (96,)),
    ('bbits', 'u1', (96,)),
])
RECORD_SIZE = RECORD_DTYPE.itemsize   # 196 bytes

# ── Filters ───────────────────────────────────────────────────────────────────

MIN_DEPTH = 16       # depth < 16 are low-quality evals
MAX_ABS_CP = 3000    # clamp extreme evaluations

# ── Feature index constants ───────────────────────────────────────────────────

_PIECE_IDX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN:  4, chess.KING:   5,
}


def _extract_features_batch(
    fens: list[str],
    cps:  list[int],
    depths: list[int],
) -> np.ndarray:
    """
    Convert a batch of FENs to records.
    Applies all filters; returns a record array (may be shorter than input).
    """
    N = len(fens)
    wf  = np.zeros((N, 768), dtype=np.uint8)
    bf  = np.zeros((N, 768), dtype=np.uint8)
    tmp = np.zeros(N, dtype=RECORD_DTYPE)
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        cp_val = cps[i]
        if cp_val is None:              # mate score — skip
            continue
        cp_val = int(cp_val)
        if abs(cp_val) > MAX_ABS_CP:    # extreme eval — skip
            continue
        if int(depths[i]) < MIN_DEPTH:  # shallow — skip
            continue

        fen = fens[i]
        if fen.count(' ') < 5:
            fen = fen + ' 0 1'          # append dummy half/full move clocks

        try:
            board = chess.Board(fen)
        except Exception:
            continue

        tmp[i]['cp']  = np.int16(cp_val)
        tmp[i]['stm'] = np.uint8(0 if board.turn == chess.WHITE else 1)

        for sq in range(64):
            p = board.piece_at(sq)
            if p is None:
                continue
            pt = _PIECE_IDX[p.piece_type]
            c  = 0 if p.color == chess.WHITE else 1
            wf[i, pt * 128 + c       * 64 + sq]       = 1
            bf[i, pt * 128 + (1 - c) * 64 + (sq ^ 56)] = 1

        valid[i] = True

    # Pack bits for all valid rows at once (vectorised)
    vidx = np.where(valid)[0]
    if len(vidx) == 0:
        return np.zeros(0, dtype=RECORD_DTYPE)

    out = tmp[vidx].copy()
    out['wbits'] = np.packbits(wf[vidx], axis=1)
    out['bbits'] = np.packbits(bf[vidx], axis=1)
    return out


# ── Per-file worker ───────────────────────────────────────────────────────────

def _process_file(args: tuple) -> tuple[int, int]:
    """Worker: read one parquet file → write one shard .bin file."""
    filepath, out_path, max_rows, seed, worker_id = args

    # Each process needs its own RNG
    rng = np.random.default_rng(seed)

    try:
        pf = pq.ParquetFile(filepath)
    except Exception as exc:
        print(f"[worker {worker_id}] ERROR opening {filepath}: {exc}")
        return 0, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix('.tmp')

    n_written = 0
    n_skipped = 0
    rows_seen  = 0
    BATCH = 50_000

    with open(tmp_path, 'wb') as fout:
        for batch in pf.iter_batches(
            batch_size=BATCH,
            columns=['fen', 'cp', 'depth'],
        ):
            fens   = batch.column('fen').to_pylist()
            cps    = batch.column('cp').to_pylist()
            depths = batch.column('depth').to_pylist()

            if max_rows is not None:
                remaining = max_rows - rows_seen
                if remaining <= 0:
                    break
                if len(fens) > remaining:
                    fens   = fens[:remaining]
                    cps    = cps[:remaining]
                    depths = depths[:remaining]

            rows_seen += len(fens)
            n_skipped += len(fens)   # will subtract valid below

            records = _extract_features_batch(fens, cps, depths)
            if len(records):
                rng.shuffle(records)  # chunk-level shuffle
                fout.write(records.tobytes())
                n_written += len(records)
                n_skipped -= len(records)

    # Atomic rename
    tmp_path.rename(out_path)
    return n_written, n_skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    data_dir   = Path(args.data_dir)
    out_dir    = Path(args.out_dir)
    parquet_files = sorted(data_dir.glob('*.parquet'))

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files in {data_dir}")
    if args.max_rows:
        print(f"  ↳ smoke-test mode: {args.max_rows:,} rows per file")
    print(f"Output directory : {out_dir}")
    print(f"Workers          : {args.workers}")
    print(f"Filters          : depth >= {MIN_DEPTH}, |cp| <= {MAX_ABS_CP}")
    print()

    shard_dir = out_dir / 'shards'
    shard_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (
            filepath,
            shard_dir / f'shard_{i:02d}.bin',
            args.max_rows,
            args.seed + i,
            i,
        )
        for i, filepath in enumerate(parquet_files)
    ]

    t0 = time.time()
    total_written = 0
    total_skipped = 0

    n_workers = min(args.workers, len(parquet_files))
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_file, worker_args),
            total=len(parquet_files),
            desc='Shards',
            unit='file',
        ))

    for written, skipped in results:
        total_written += written
        total_skipped += skipped

    elapsed   = time.time() - t0
    shard_size = total_written * RECORD_SIZE
    pass_rate  = total_written / max(1, total_written + total_skipped) * 100

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Written : {total_written:,} records  ({shard_size / 1e9:.1f} GB)")
    print(f"  Skipped : {total_skipped:,}  (pass rate {pass_rate:.1f}%)")
    print(f"  Shards  : {shard_dir}")

    # Write metadata
    meta = {
        'n_shards':    len(parquet_files),
        'n_records':   total_written,
        'record_size': RECORD_SIZE,
        'min_depth':   MIN_DEPTH,
        'max_abs_cp':  MAX_ABS_CP,
        'source':      str(data_dir),
    }
    import json
    (out_dir / 'binpack_meta.json').write_text(json.dumps(meta, indent=2))
    print(f"  Metadata: {out_dir / 'binpack_meta.json'}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  default='data/parquet',
                   help='Directory with parquet files')
    p.add_argument('--out_dir',   default='data/binpack',
                   help='Output directory for shard files')
    p.add_argument('--max_rows',  type=int, default=None,
                   help='Max rows per parquet file (smoke-test)')
    p.add_argument('--workers',   type=int, default=min(17, os.cpu_count() or 1),
                   help='Parallel worker processes')
    p.add_argument('--seed',      type=int, default=42)
    build(p.parse_args())
