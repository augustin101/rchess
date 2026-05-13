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
  # Full run
  python training/build_binpack.py

  # Quick smoke-test (1000 rows per file, <30 s)
  python training/build_binpack.py --max_rows 1000
"""

import argparse
import json
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

MIN_DEPTH  = 18    # depth < 20 are low-quality evals
MAX_ABS_CP = 750   # skip extreme evaluations


def _passes_filters(cp_val, depth: int) -> bool:
    """Return True if the position should be kept in the dataset."""
    if cp_val is None:
        return False          # mate score — no centipawn value
    if abs(int(cp_val)) > MAX_ABS_CP:
        return False          # extreme eval — noisy / likely blunder
    if depth < MIN_DEPTH:
        return False          # shallow search — unreliable eval
    return True


# ── Feature index constants ───────────────────────────────────────────────────

_PIECE_IDX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN:  4, chess.KING:   5,
}


def _extract_features_batch(
    fens:   list[str],
    cps:    list[int],
    depths: list[int],
    moves:  list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Convert a batch of FENs to records.
    Applies all filters; returns (record_array, n_capture_skips).
    """
    N = len(fens)
    wf  = np.zeros((N, 768), dtype=np.uint8)
    bf  = np.zeros((N, 768), dtype=np.uint8)
    tmp = np.zeros(N, dtype=RECORD_DTYPE)
    valid = np.zeros(N, dtype=bool)
    n_capture_skips = 0

    for i in range(N):
        if not _passes_filters(cps[i], int(depths[i])):
            continue
        cp_val = int(cps[i])

        fen = fens[i]
        if fen.count(' ') < 5:
            fen = fen + ' 0 1'          # append dummy half/full move clocks

        try:
            board = chess.Board(fen)
        except Exception:
            continue

        # Skip positions where the main line starts with a capture.
        # board.is_capture() is a free O(1) mailbox lookup — no extra cost
        # since the board is already constructed. Handles en passant correctly.
        move = moves[i] if moves is not None else ''
        if move:
            try:
                if board.is_capture(chess.Move.from_uci(move)):
                    n_capture_skips += 1
                    continue
            except ValueError:
                continue  # malformed UCI move — skip

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
        return np.zeros(0, dtype=RECORD_DTYPE), n_capture_skips

    out = tmp[vidx].copy()
    out['wbits'] = np.packbits(wf[vidx], axis=1)
    out['bbits'] = np.packbits(bf[vidx], axis=1)
    return out, n_capture_skips


# ── Per-file worker ───────────────────────────────────────────────────────────

def _process_file(args: tuple) -> tuple[int, int, int]:
    """Worker: read one parquet file → write one shard .bin file.
    Returns (n_written, n_skipped, n_capture_skips)."""
    filepath, out_path, max_rows, seed, worker_id = args

    rng = np.random.default_rng(seed)

    try:
        pf = pq.ParquetFile(filepath)
    except Exception as exc:
        print(f"[worker {worker_id}] ERROR opening {filepath}: {exc}")
        return 0, 0, 0

    has_move = 'line' in set(pf.schema_arrow.names)
    columns  = ['fen', 'cp', 'depth'] + (['line'] if has_move else [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix('.tmp')

    n_written       = 0
    n_skipped       = 0
    n_capture_skips = 0
    rows_seen       = 0
    BATCH = 50_000

    with open(tmp_path, 'wb') as fout:
        for batch in pf.iter_batches(batch_size=BATCH, columns=columns):
            fens   = batch.column('fen').to_pylist()
            cps    = batch.column('cp').to_pylist()
            depths = batch.column('depth').to_pylist()
            moves  = [l.split()[0] if l else '' for l in batch.column('line').to_pylist()] \
                     if has_move else [''] * len(fens)

            if max_rows is not None:
                remaining = max_rows - rows_seen
                if remaining <= 0:
                    break
                if len(fens) > remaining:
                    fens   = fens[:remaining]
                    cps    = cps[:remaining]
                    depths = depths[:remaining]
                    moves  = moves[:remaining]

            rows_seen += len(fens)
            n_skipped += len(fens)   # will subtract valid below

            records, batch_capture_skips = _extract_features_batch(fens, cps, depths, moves)
            n_capture_skips += batch_capture_skips
            if len(records):
                rng.shuffle(records)  # chunk-level shuffle
                fout.write(records.tobytes())
                n_written += len(records)
                n_skipped -= len(records)

    tmp_path.rename(out_path)
    return n_written, n_skipped, n_capture_skips


# ── Main ──────────────────────────────────────────────────────────────────────

def build(args: argparse.Namespace) -> None:
    data_dir   = Path(args.data_dir)
    out_dir    = Path(args.out_dir)
    parquet_files = sorted(data_dir.glob('*.parquet'))

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    # Inspect first file to detect optional columns
    try:
        _schema = pq.ParquetFile(parquet_files[0]).schema_arrow.names
        has_move_col = 'line' in set(_schema)
    except Exception:
        has_move_col = False

    print(f"Found {len(parquet_files)} parquet files in {data_dir}")
    if args.max_rows:
        print(f"  ↳ smoke-test mode: {args.max_rows:,} rows per file")
    print(f"Output directory : {out_dir}")
    print(f"Workers          : {args.workers}")
    print(f"Filters")
    print(f"  depth >= {MIN_DEPTH}       (shallow evals skipped)")
    print(f"  |cp|  <= {MAX_ABS_CP}      (extreme evals skipped)")
    print(f"  no capture on main line: {'yes' if has_move_col else 'no  (no move column in parquet)'}")
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
    total_written       = 0
    total_skipped       = 0
    total_capture_skips = 0

    n_workers = min(args.workers, len(parquet_files))
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_file, worker_args),
            total=len(parquet_files),
            desc='Shards',
            unit='file',
        ))

    for written, skipped, capture_skips in results:
        total_written       += written
        total_skipped       += skipped
        total_capture_skips += capture_skips

    elapsed    = time.time() - t0
    shard_size = total_written * RECORD_SIZE
    total_seen = total_written + total_skipped
    pass_rate  = total_written / max(1, total_seen) * 100

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Written         : {total_written:,} records  ({shard_size / 1e9:.1f} GB)")
    print(f"  Skipped (total) : {total_skipped:,}  (pass rate {pass_rate:.1f}%)")
    if has_move_col:
        capture_pct = total_capture_skips / max(1, total_seen) * 100
        print(f"    of which capture on main line: {total_capture_skips:,}  ({capture_pct:.1f}%)")
    print(f"  Shards          : {shard_dir}")

    meta = {
        'filters': {
            'min_depth':              MIN_DEPTH,
            'max_abs_cp':             MAX_ABS_CP,
            'skip_capture_main_line': has_move_col,
        },
        'stats': {
            'n_shards':          len(parquet_files),
            'n_records':         total_written,
            'n_skipped':         total_skipped,
            'n_capture_skips':   total_capture_skips if has_move_col else None,
            'pass_rate_pct':     round(pass_rate, 2),
            'elapsed_s':         round(elapsed, 1),
            'shard_size_gb':     round(shard_size / 1e9, 3),
            'record_size_bytes': RECORD_SIZE,
        },
        'source': str(data_dir),
        'shards': str(shard_dir),
    }
    meta_path = out_dir / 'binpack_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata        : {meta_path}")


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
