"""
Export trained NNUE weights to the quantised binary format read by src/engine/nnue.rs.

Quantisation scheme
───────────────────
    FT weights / biases :  float × QA=127  → i16
    L1/L2/out weights   :  float × QB=64   → i8
    L1 biases           :  float × QA×QB   → i32   (pre-compensates accumulator scale)
    L2 / out biases     :  float × QB²     → i32

Binary layout (RNNUE2\\0\\0, all little-endian):
    magic      :  8 bytes
    header     :  6 × u32  (INPUT=768, L1=256, L2=32, L3=32, OUT=1, version=2)
    ft_weight  :  INPUT × L1  i16          (768×256 rows, input-major)
    ft_bias    :  L1          i16
    l1_weight  :  L2 × (L1×2) i8           (32×512, output-major for SIMD)
    l1_bias    :  L2          i32
    l2_weight  :  L3 × L2     i8
    l2_bias    :  L3          i32
    out_weight :  L3          i8
    out_bias   :  1           i32

Usage (run from project root):
    python training/export.py                              # checkpoints/best.pt → networks/nnue.bin
    python training/export.py checkpoints/epoch_03.pt networks/nnue.bin
"""

import struct
import sys
import os
import numpy as np
import torch

from model import NNUE, INPUT_SIZE, L1_SIZE, L2_SIZE, L3_SIZE, QA, QB

MAGIC   = b"RNNUE2\x00\x00"
VERSION = 2


def quantise_clamp(arr: np.ndarray, scale: float, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(np.round(arr * scale), info.min, info.max).astype(dtype)


def export(checkpoint_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"Loading {checkpoint_path} …")
    data  = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = NNUE()
    model.load_state_dict(data["model"] if "model" in data else data)
    model.eval()

    # ── Extract float weights ─────────────────────────────────────────────────
    # PyTorch Linear weight shape: [out_features, in_features] → transpose to input-major.

    ft_w_f = model.ft.weight.detach().numpy().T   # [INPUT_SIZE, L1_SIZE]
    ft_b_f = model.ft.bias.detach().numpy()       # [L1_SIZE]

    # L1 weights must be output-major [L2_SIZE, L1_SIZE*2] for SIMD dot-product.
    l1_w_f = model.l1.weight.detach().numpy()     # [L2_SIZE, L1_SIZE*2]  (already output-major)
    l1_b_f = model.l1.bias.detach().numpy()       # [L2_SIZE]

    l2_w_f = model.l2.weight.detach().numpy()     # [L3_SIZE, L2_SIZE]
    l2_b_f = model.l2.bias.detach().numpy()       # [L3_SIZE]

    out_w_f = model.out.weight.detach().numpy().flatten()  # [L3_SIZE]
    out_b_f = model.out.bias.detach().numpy()              # [1]

    # ── Shape assertions ──────────────────────────────────────────────────────
    assert ft_w_f.shape  == (INPUT_SIZE,  L1_SIZE),          ft_w_f.shape
    assert l1_w_f.shape  == (L2_SIZE,     L1_SIZE * 2),      l1_w_f.shape
    assert l2_w_f.shape  == (L3_SIZE,     L2_SIZE),          l2_w_f.shape
    assert out_w_f.shape == (L3_SIZE,),                      out_w_f.shape

    # ── Quantise ──────────────────────────────────────────────────────────────
    ft_w_q  = quantise_clamp(ft_w_f,  QA,       np.int16)   # [INPUT, L1]
    ft_b_q  = quantise_clamp(ft_b_f,  QA,       np.int16)   # [L1]

    l1_w_q  = quantise_clamp(l1_w_f,  QB,       np.int8)    # [L2, L1*2]
    l1_b_q  = quantise_clamp(l1_b_f,  QA * QB,  np.int32)   # [L2]

    l2_w_q  = quantise_clamp(l2_w_f,  QB,       np.int8)    # [L3, L2]
    l2_b_q  = quantise_clamp(l2_b_f,  QB * QB,  np.int32)   # [L3]

    out_w_q = quantise_clamp(out_w_f, QB,       np.int8)    # [L3]
    out_b_q = quantise_clamp(out_b_f, QB * QB,  np.int32)   # [1]

    # ── Write binary ──────────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIIII", INPUT_SIZE, L1_SIZE, L2_SIZE, L3_SIZE, 1, VERSION))

        f.write(ft_w_q.astype("<i2").tobytes())   # input-major [INPUT, L1]
        f.write(ft_b_q.astype("<i2").tobytes())

        f.write(l1_w_q.astype("<i1").tobytes())   # output-major [L2, L1*2]
        f.write(l1_b_q.astype("<i4").tobytes())

        f.write(l2_w_q.astype("<i1").tobytes())
        f.write(l2_b_q.astype("<i4").tobytes())

        f.write(out_w_q.astype("<i1").tobytes())
        f.write(out_b_q.astype("<i4").tobytes())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported → {output_path}  ({size_kb:.0f} KB)")

    # ── Sanity check: float vs quantised output on a random position ──────────
    _sanity_check(model, ft_w_q, ft_b_q, l1_w_q, l1_b_q, l2_w_q, l2_b_q,
                  out_w_q, out_b_q)


def _sanity_check(model, ft_w_q, ft_b_q, l1_w_q, l1_b_q, l2_w_q, l2_b_q,
                  out_w_q, out_b_q):
    """Compare float and integer forward passes on the starting position."""
    import chess
    from build_binpack import _extract_features_batch

    # Use the starting position (White to move, cp ≈ 0)
    fen = chess.Board().fen().rsplit(' ', 2)[0]   # drop move clocks
    records = _extract_features_batch([fen], [0], [99])
    if len(records) == 0:
        print("Sanity check: skipped (FEN extraction failed)")
        return

    rec  = records[0]
    stm  = int(rec['stm'])
    wbits = rec['wbits']   # [96] uint8
    bbits = rec['bbits']

    # Unpack bitmaps → float arrays for the float forward pass
    wf = np.unpackbits(wbits).astype(np.float32)   # [768]
    bf = np.unpackbits(bbits).astype(np.float32)

    # Float forward
    wft  = torch.tensor(wf).unsqueeze(0)
    bft  = torch.tensor(bf).unsqueeze(0)
    stmt = torch.tensor([stm], dtype=torch.int64)
    with torch.no_grad():
        float_logit = model(wft, bft, stmt).item()   # raw logit (pre-sigmoid)
    # Convert logit → centipawns: cp = logit * SCALE_CP
    float_cp = float_logit * 400.0

    # Integer forward (mirroring Rust scalar path)
    QA_i, QB_i = int(QA), int(QB)

    acc_w = ft_b_q.astype(np.int32).copy()
    acc_b = ft_b_q.astype(np.int32).copy()
    for i in np.where(wf > 0)[0]:
        acc_w += ft_w_q[i].astype(np.int32)
    for i in np.where(bf > 0)[0]:
        acc_b += ft_w_q[i].astype(np.int32)

    # stm first
    sa = np.clip(acc_w if stm == 0 else acc_b, 0, QA_i).astype(np.uint8)
    oa = np.clip(acc_b if stm == 0 else acc_w, 0, QA_i).astype(np.uint8)
    inp = np.concatenate([sa, oa])   # 512 u8

    l1 = np.array([
        np.clip((int(l1_b_q[j]) + int(inp.astype(np.int32) @ l1_w_q[j].astype(np.int32))) // QA_i,
                0, QB_i)
        for j in range(len(l1_b_q))
    ], dtype=np.int32)

    l2 = np.array([
        np.clip((int(l2_b_q[j]) + int(l1 @ l2_w_q[j].astype(np.int32))) // QB_i,
                0, QB_i)
        for j in range(len(l2_b_q))
    ], dtype=np.int32)

    raw = int(out_b_q[0]) + int(l2 @ out_w_q.astype(np.int32))
    int_cp = raw * 400 // (QB_i * QB_i)

    print(f"Sanity check (start pos): float={float_cp:.1f} cp  quantised={int_cp} cp"
          f"  Δ={abs(float_cp - int_cp):.1f}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    out  = sys.argv[2] if len(sys.argv) > 2 else "networks/nnue.bin"
    export(ckpt, out)
