"""
Binpack dataset for NNUE training.

Reads the binary shard files produced by build_binpack.py + shuffle_split.py.

Feature index (must match src/engine/nnue.rs and build_binpack.py):
  White POV : feat = piece_type*128 + color*64 + square
  Black POV : feat = piece_type*128 + (1-color)*64 + (square^56)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from build_binpack import RECORD_DTYPE, RECORD_SIZE

# ── GPU bitmap unpacking ──────────────────────────────────────────────────────
# Bits were packed MSB-first with np.packbits, so we extract them MSB-first.
# Result: float32 tensors with values in {0, 1}.

_SHIFTS = None   # cached on first call


def unpack_batch(
    wbits: torch.Tensor,
    bbits: torch.Tensor,
    stm:   torch.Tensor,
    cp:    torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack bitmap tensors to float feature tensors on the target device.

    Args:
        wbits : [B, 96] uint8 — packed white-POV features
        bbits : [B, 96] uint8 — packed black-POV features
        stm   : [B]    int64  — 0=White, 1=Black
        cp    : [B]    float  — centipawns from White's perspective

    Returns:
        white_feat : [B, 768] float32
        black_feat : [B, 768] float32
        stm        : [B]      int64
        cp         : [B]      float32
    """
    global _SHIFTS
    if _SHIFTS is None or _SHIFTS.device != device:
        _SHIFTS = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)

    wbits = wbits.to(device, non_blocking=True)   # [B, 96]
    bbits = bbits.to(device, non_blocking=True)

    # Vectorised bit extraction: [B,96,1] >> [8] & 1 → [B,96,8] → [B,768]
    white_feat = ((wbits.unsqueeze(-1) >> _SHIFTS) & 1).float().view(-1, 768)
    black_feat = ((bbits.unsqueeze(-1) >> _SHIFTS) & 1).float().view(-1, 768)

    stm = stm.to(device, non_blocking=True)
    cp  = cp.to(device,  non_blocking=True)
    return white_feat, black_feat, stm, cp


# ── Map-style dataset (val / test — fits in RAM) ──────────────────────────────

class BinpackDataset(Dataset):
    """Fixed-size dataset for validation/test (memmapped, random access)."""

    def __init__(self, path: str | Path):
        path  = Path(path)
        n     = path.stat().st_size // RECORD_SIZE
        self.data = np.memmap(path, dtype=RECORD_DTYPE, mode='r', shape=(n,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        r = self.data[idx]
        return (
            torch.from_numpy(r['wbits'].copy()),   # [96] uint8
            torch.from_numpy(r['bbits'].copy()),   # [96] uint8
            int(r['stm']),                          # scalar int
            float(r['cp']),                         # scalar float
        )


# ── Iterable dataset (train — streams shards, too large for RAM) ──────────────

class BinpackIterableDataset(IterableDataset):
    """
    Streams training records from shard files.

    Each epoch iterates every shard once in a random order.
    Within each shard the records are served in random order.
    Multiple DataLoader workers each handle a disjoint subset of shards.
    """

    def __init__(
        self,
        train_dir:   str | Path,
        shuffle:     bool = True,
        seed:        int  = 0,
    ):
        self.shard_paths = sorted(Path(train_dir).glob('train_*.bin'))
        if not self.shard_paths:
            raise FileNotFoundError(f"No train_*.bin in {train_dir}")
        self.shuffle = shuffle
        self.seed    = seed

    def __len__(self) -> int:
        """Approximate total records (for progress bars)."""
        return sum(p.stat().st_size // RECORD_SIZE for p in self.shard_paths)

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()

        # Divide shards evenly across DataLoader workers
        shards = self.shard_paths
        if worker_info is not None:
            shards = shards[worker_info.id::worker_info.num_workers]

        if self.shuffle:
            shards = list(shards)
            random.shuffle(shards)

        for path in shards:
            n    = path.stat().st_size // RECORD_SIZE
            data = np.memmap(path, dtype=RECORD_DTYPE, mode='r', shape=(n,))

            if self.shuffle:
                indices = np.random.permutation(n)
            else:
                indices = np.arange(n)

            for idx in indices:
                r = data[idx]
                yield (
                    torch.from_numpy(r['wbits'].copy()),
                    torch.from_numpy(r['bbits'].copy()),
                    int(r['stm']),
                    float(r['cp']),
                )


# ── Collate function for DataLoader ──────────────────────────────────────────

def collate_fn(batch):
    wbits = torch.stack([b[0] for b in batch])   # [B, 96] uint8
    bbits = torch.stack([b[1] for b in batch])   # [B, 96] uint8
    stm   = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    cp    = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return wbits, bbits, stm, cp


# ── Convenience loader factory ────────────────────────────────────────────────

def make_loaders(
    splits_json: str | Path,
    batch_size:  int  = 4096,
    num_workers: int  = 4,
    pin_memory:  bool = True,
) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader) from a splits.json manifest.
    train_loader uses IterableDataset (streams shards).
    val / test loaders use map-style Dataset (memmapped).
    """
    from torch.utils.data import DataLoader

    meta = json.loads(Path(splits_json).read_text())

    train_ds = BinpackIterableDataset(meta['train_dir'])
    val_ds   = BinpackDataset(meta['val_path'])
    test_ds  = BinpackDataset(meta['test_path'])

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
        prefetch_factor = 2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        num_workers = min(4, num_workers),
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
        shuffle     = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        num_workers = min(4, num_workers),
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
        shuffle     = False,
    )
    return train_loader, val_loader, test_loader
