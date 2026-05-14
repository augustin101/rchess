NNUE — Architecture, Training and Inference
============================================

The engine supports a quantised dual-perspective NNUE (Efficiently Updatable Neural
Network) evaluation.  The Rust implementation is in ``src/engine/nnue.rs``; the
Python training stack lives in ``training/``.

----

Architecture Overview
---------------------

.. code-block:: text

    ┌──────────────────────────────────────────────────────────┐
    │  Position                                                │
    │   White-POV features (768 bits)                         │
    │   Black-POV features (768 bits)                         │
    └────────────┬─────────────────────────┬───────────────────┘
                 │ Feature Transformer (FT)│  (shared weights)
                 ▼                         ▼
    ┌────────────────────┐   ┌────────────────────┐
    │  White accumulator │   │  Black accumulator │
    │  768 → 256  i16    │   │  768 → 256  i16    │
    └──────────┬─────────┘   └──────────┬─────────┘
               │ CReLU [0, QA]           │ CReLU [0, QA]
               └──────────────┬──────────┘
                               │ concatenate stm-first → 512 u8
                               ▼
                    ┌─────────────────────┐
                    │  L1: 512 → 32  i8   │
                    │  bias i32 / QA      │
                    │  CReLU [0, QB]      │
                    └──────────┬──────────┘
                               │ 32 i32
                               ▼
                    ┌─────────────────────┐
                    │  L2: 32 → 32  i8    │
                    │  bias i32 / QB      │
                    │  CReLU [0, QB]      │
                    └──────────┬──────────┘
                               │ 32 i32
                               ▼
                    ┌─────────────────────┐
                    │  Out: 32 → 1  i8    │
                    │  bias i32           │
                    └──────────┬──────────┘
                               │ raw integer
                               ▼
                    centipawns = raw × 400 / QB²

``QA`` = 127 (feature-transformer scale), ``QB`` = 64 (hidden-layer scale).

----

Feature Encoding
----------------

The input to the network is a pair of binary feature vectors, one per perspective.
Each vector has 768 bits: 6 piece types × 2 colors × 64 squares.

**White-perspective index** for a piece of type *pt*, color *c*, on square *sq*::

    feat_w(pt, c, sq) = pt × 128 + c × 64 + sq

**Black-perspective index** — board is vertically mirrored, colors are swapped::

    feat_b(pt, c, sq) = pt × 128 + c.flip() × 64 + (sq XOR 56)

A typical position activates ~30–35 features per perspective (one per piece).

----

Feature Transformer (FT)
------------------------

The FT is a linear layer with no activation, applied separately to each perspective's
feature set using the **same** weight matrix (weight sharing)::

    acc_white = ft_bias + Σ ft_weight[feat_w(pt, c, sq)]  for each piece
    acc_black = ft_bias + Σ ft_weight[feat_b(pt, c, sq)]  for each piece

where ``ft_weight`` has shape ``[INPUT_SIZE=768, L1_SIZE=256]`` and ``ft_bias``
has shape ``[L1_SIZE=256]``.  Both are stored as ``i16``.

The accumulator is maintained incrementally: when a piece moves, only the affected
feature columns are added or subtracted rather than recomputing from scratch.

----

Incremental Accumulator Updates
--------------------------------

During search the engine maintains a stack of accumulators (one per ply, max 128).
Before each ``board.make_move`` the engine calls ``push_move``, which copies the
current accumulator to ``ply + 1`` and applies the feature delta:

**Normal move** (piece *p* from *from* to *to*, possibly capturing *cap*)::

    if capture: sub_feat(cap.type, cap.color, to)
    sub_feat(p.type, p.color, from)
    add_feat(p.type, p.color, to)

**Promotion** (pawn *from* → promoted piece *promo* at *to*)::

    if capture: sub_feat(cap.type, cap.color, to)
    sub_feat(Pawn, us, from)
    add_feat(promo, us, to)

**En passant** (pawn captures en passant; captured pawn sits at ``(to.file, from.rank)``)::

    sub_feat(Pawn, them, ep_sq)
    sub_feat(Pawn, us, from)
    add_feat(Pawn, us, to)

**Castling** (king from → to; rook kingside h→f or queenside a→d)::

    sub_feat(King, us, from);  add_feat(King, us, to)
    sub_feat(Rook, us, rook_from);  add_feat(Rook, us, rook_to)

On ``unmake_move`` the engine pops the accumulator stack (``ply -= 1``), restoring
the parent's pre-move accumulator at zero cost.

A *full refresh* (``O(pieces)``) recomputes the accumulator from scratch at the
root of each new search.

----

Forward Pass — Quantised Integer Arithmetic
-------------------------------------------

All arithmetic uses signed integers; no floating-point operations are needed at
inference time.

**Step 1 — CReLU on the accumulator**

The CReLU activation clamps and narrows each ``i16`` accumulator lane to a ``u8``::

    crelu(x) = clamp(x, 0, QA)    as u8

The stm (side-to-move) accumulator is placed first in the 512-element input vector::

    input[0..256]   = crelu(acc_stm)
    input[256..512] = crelu(acc_opp)

**Step 2 — L1 layer**

For each output neuron *j*::

    dot_j = l1_bias[j] + Σ_{i=0..511} input[i] × l1_weight[j][i]
    l1[j] = clamp(dot_j / QA, 0, QB)

``l1_bias[j]`` is stored at scale ``QA × QB`` (see Quantisation below).
The division by ``QA`` de-scales the dot product from the accumulator scale.
``l1_weight`` has shape ``[L2_SIZE=32, CONCAT=512]`` in output-major order.

**Step 3 — L2 layer**

For each output neuron *j*::

    dot_j = l2_bias[j] + Σ_{i=0..31} l1[i] × l2_weight[j][i]
    l2[j] = clamp(dot_j / QB, 0, QB)

**Step 4 — Output layer**

::

    raw = out_bias + Σ_{i=0..31} l2[i] × out_weight[i]

**Step 5 — Convert to centipawns**

The network is trained with ``logit ≈ cp / EVAL_SCALE``.  In integer arithmetic,
the logit is represented at scale ``QB²``, so::

    centipawns = raw × EVAL_SCALE / QB²
               = raw × 400 / 4096

----

SIMD Inference
--------------

The forward pass is dispatched at runtime to the fastest available path:

* **AVX2** (x86-64 with AVX2) — ``forward_avx2``
* **NEON** (AArch64) — ``forward_neon``
* **Scalar** — ``forward_scalar`` (fallback)

**AVX2 CReLU + pack**

The 256 i16 lanes of one perspective are clamped with ``_mm256_max_epi16`` /
``_mm256_min_epi16`` and packed to 256 u8 with ``_mm256_packus_epi16`` + a
``_mm256_permute4x64_epi64(imm=0xD8)`` to correct the interleaving of the two
128-bit lanes.  Both perspectives are processed in a single loop.

**AVX2 L1 dot product**

The L1 layer processes 512 u8 inputs × i8 weights.  With ``QA = 127`` the
maximum pairwise product is ``127 × 127 = 16129`` and the maximum pair-sum is
``32258 < 32767``, so ``_mm256_maddubs_epi16`` (u8 × i8 → i16, saturating) does
not saturate.  The loop is unrolled 4× per 128-byte chunk::

    a += _mm256_madd_epi16(
             _mm256_maddubs_epi16(input_chunk, weight_chunk),
             ones)   // reduce i16 pairs → i32

Four accumulators (``a0``–``a3``) are horizontally summed at the end using
``_mm_hadd_epi32`` + ``_mm_cvtsi128_si32``.

**NEON CReLU**

Uses ``vmaxq_s16`` / ``vminq_s16`` for clamping and ``vqmovun_s16`` for
narrowing i16→u8 (8 lanes at a time).

L2 and output layers are small (32×32 and 32×1) and use the scalar path on all
architectures.

**Accumulator alignment**

``Accumulator`` is ``#[repr(align(64))]`` so that AVX2 loads/stores never cross
cache-line boundaries.

----

Quantisation Scheme
-------------------

All float weights from training are quantised before being written to the binary
weight file by ``training/export.py``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Scale
     - Storage type
     - Rationale
   * - FT weights
     - × QA = 127
     - i16
     - acc values fit comfortably in i16 (768 × 127 ≪ 32767 for typical sparse inputs)
   * - FT biases
     - × QA = 127
     - i16
     - same scale as accumulated weights
   * - L1 weights
     - × QB = 64
     - i8
     - product with u8 input ≤ 127×64 = 8128 fits i16 for maddubs
   * - L1 biases
     - × QA × QB = 8128
     - i32
     - matches scale of ``Σ crelu × w1`` before dividing by QA
   * - L2 weights
     - × QB = 64
     - i8
     - l1 output is in [0, QB], product ≤ 64×64 = 4096 fits i32
   * - L2 biases
     - × QB² = 4096
     - i32
     - matches scale of ``Σ l1 × w2`` before dividing by QB
   * - Output weights
     - × QB = 64
     - i8
     - same reasoning as L2 weights
   * - Output bias
     - × QB² = 4096
     - i32
     - matches scale of ``Σ l2 × w_out``

The quantisation formula throughout is::

    quantised = clip(round(float_value × scale), dtype_min, dtype_max)

----

Binary File Format
------------------

File: ``networks/nnue.bin`` (magic ``RNNUE2\0\0``, all integers little-endian).

.. code-block:: text

    Offset  Size      Field
    ──────  ────────  ─────────────────────────────────────────────
    0       8 B       magic = b"RNNUE2\x00\x00"
    8       24 B      header: 6 × u32
                        [INPUT=768, L1=256, L2=32, L3=32, OUT=1, version=2]
    32      384 KB    ft_weight: [768][256] i16  (input-major)
    +512 B            ft_bias:  [256] i16
    +16 KB            l1_weight: [32][512] i8    (output-major, for SIMD)
    +128 B            l1_bias:   [32] i32
    +1 KB             l2_weight: [32][32] i8
    +128 B            l2_bias:   [32] i32
    +32 B             out_weight: [32] i8
    +4 B              out_bias:   i32

Total: ≈ 402 KB.

----

Training
--------

**Data**

The training data comes from the `Lichess position evaluation dataset
<https://huggingface.co/datasets/Lichess/chess-position-evaluations>`_ (parquet
files).  Each record contains a FEN string and a centipawn evaluation.
``training/build_binpack.py`` converts these to compact 196-byte binary records:

.. code-block:: text

    ┌──────────┬──────────┬───────┬────────┐
    │ wbits    │ bbits    │  stm  │   cp   │
    │ 96 bytes │ 96 bytes │ uint8 │ int16  │
    └──────────┴──────────┴───────┴────────┘

``wbits`` / ``bbits`` are packed-bit representations of the 768-bit White/Black
feature vectors (``np.packbits``).  ``stm`` is 0 for White to move, 1 for Black.
``cp`` is the centipawn evaluation from White's perspective, clamped to ±3000.

**Loss Function**

A mixed BCE + MSE loss (``MixedLoss`` in ``training/train.py``)::

    target      = σ(cp / SCALE_CP)          SCALE_CP = 400
    BCE(logit, target) = −[ target · log σ(logit) + (1−target) · log(1−σ(logit)) ]
    MSE(logit, cp/SCALE_CP)

    L = (1 − α) · BCE + α · MSE            α = 0.01 (default)

The BCE component trains the network to estimate win probability (strategy);
the MSE component with a small ``α`` adds a penalty on the raw score magnitude
(precision).

**Optimizer and Schedule**

* Optimizer: AdamW (``lr`` default 1e-3, ``weight_decay`` 1e-5).
* LR schedule: flat for the first ``lr_flat_frac`` (default 25 %) of epochs, then
  cosine decay to zero::

      λ(epoch) = 1.0                                    if epoch < flat_epochs
               = 0.5 × (1 + cos(π × (epoch − flat) / decay))   otherwise

* Gradient clipping: ``max_norm = 1.0`` (applied before each optimizer step).

**Checkpoints**

``train.py`` saves:

* ``epoch_NN.pt`` — model weights after each epoch.
* ``best.pt`` — model with the lowest validation BCE loss seen so far.
* ``resume.pt`` — full training state (model, optimizer, scheduler, metrics) for
  resuming interrupted runs with ``--resume``.
* ``vepoch_NNNN.pt`` — optional mid-epoch checkpoints at every ``N`` positions
  seen (``--virtual_epoch_size``).

**Export**

``training/export.py`` quantises a checkpoint's float weights to the binary format
above and writes ``networks/nnue.bin``.  After export a quick sanity check compares
the float and integer forward passes on the starting position; the difference should
be within a few centipawns.

----

Match Evaluation
----------------

``training/match.py`` runs NNUE vs static eval matches to measure network quality.
Both variants are compiled from the same binary (``rchess-uci``), with the
static-eval engine launched with ``--no-nnue``.  Games use a 60 s + 1 s
increment time control; results are recorded as PGN and JSON in ``matches/``.
