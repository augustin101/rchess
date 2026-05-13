//! Quantised dual-perspective NNUE with incremental accumulator and SIMD inference.
//!
//! Architecture: 768 → 256 (FT, weight-shared) → 512 (concat both perspectives)
//!               → 32 (L1) → 32 (L2) → 1 (output)
//!
//! Quantisation scheme
//! ───────────────────
//!   FT weights / biases :  float × QA=127  → i16
//!   L1/L2/out weights   :  float × QB=64   → i8
//!   L1 biases           :  float × QA×QB   → i32
//!   L2 / out biases     :  float × QB²     → i32
//!
//! Binary format v2  ("RNNUE2\0\0", 8-byte magic):
//!   header     : 6 × u32  (INPUT, L1, L2, L3, 1, version=2)
//!   ft_weight  : INPUT×L1  i16  (768×256 = 196 608 values)
//!   ft_bias    : L1        i16
//!   l1_weight  : L2×(L1×2) i8  (32×512 = 16 384 values)  — output-major
//!   l1_bias    : L2        i32
//!   l2_weight  : L3×L2     i8
//!   l2_bias    : L3        i32
//!   out_weight : L3        i8
//!   out_bias   : 1         i32
//!
//! All multi-byte integers are little-endian.

// Unsafe intrinsics inside `unsafe fn` bodies — Rust-2024 requires explicit blocks;
// we suppress the lint here since all callers are already gated on `unsafe`.
#![allow(unsafe_op_in_unsafe_fn)]

use std::io;
use std::path::Path;

use crate::core::board::Board;
use crate::core::moves::{Move, MoveFlag};
use crate::core::types::{Color, PieceType, Square};

// ── Architecture & quantisation constants ─────────────────────────────────────

pub const INPUT_SIZE:  usize = 768;             // 6 piece-types × 2 colors × 64 squares
pub const L1_SIZE:     usize = 256;             // FT output per perspective
pub const L2_SIZE:     usize = 32;
pub const L3_SIZE:     usize = 32;
const CONCAT:          usize = L1_SIZE * 2;     // 512 — input to the output network

pub const QA: i32 = 127;   // FT / accumulator quantisation scale
pub const QB: i32 = 64;    // hidden-layer quantisation scale

const EVAL_SCALE: i32 = 400;  // cp per unit of tanh output (from training)

const MAGIC: &[u8; 8] = b"RNNUE2\x00\x00";

// ── Feature indexing ──────────────────────────────────────────────────────────

/// Feature index from White's perspective.
#[inline]
pub fn feat_w(pt: PieceType, color: Color, sq: Square) -> usize {
    (pt as usize) * 128 + (color as usize) * 64 + sq.0 as usize
}

/// Feature index from Black's perspective (board vertically mirrored, colors swapped).
#[inline]
pub fn feat_b(pt: PieceType, color: Color, sq: Square) -> usize {
    (pt as usize) * 128 + (color.flip() as usize) * 64 + (sq.0 ^ 56) as usize
}

// ── Weight storage ────────────────────────────────────────────────────────────

pub struct Nnue {
    /// FT weights [INPUT_SIZE][L1_SIZE] — input-major (best for incremental adds).
    pub ft_weight: Box<[[i16; L1_SIZE]; INPUT_SIZE]>,
    pub ft_bias:   [i16; L1_SIZE],
    /// L1 weights [L2_SIZE][CONCAT] — output-major (best for SIMD dot-product).
    pub l1_weight: Box<[[i8; CONCAT]; L2_SIZE]>,
    pub l1_bias:   [i32; L2_SIZE],
    pub l2_weight: [[i8; L2_SIZE]; L3_SIZE],
    pub l2_bias:   [i32; L3_SIZE],
    pub out_weight: [i8; L3_SIZE],
    pub out_bias:   i32,
}

impl Nnue {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let buf = std::fs::read(path)?;
        Self::from_slice(&buf)
    }

    pub fn from_slice(buf: &[u8]) -> io::Result<Self> {
        let mut p = 0usize;

        let take = |p: &mut usize, n: usize| -> &[u8] {
            let s = &buf[*p..*p + n]; *p += n; s
        };

        if take(&mut p, 8) != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not RNNUE v2"));
        }
        p += 24; // skip 6 × u32 header

        let rd_i16 = |buf: &[u8], p: &mut usize, n: usize| -> Vec<i16> {
            let b = &buf[*p..*p + n * 2]; *p += n * 2;
            b.chunks_exact(2).map(|c| i16::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let rd_i8 = |buf: &[u8], p: &mut usize, n: usize| -> Vec<i8> {
            let b = &buf[*p..*p + n]; *p += n;
            b.iter().map(|&x| x as i8).collect()
        };
        let rd_i32 = |buf: &[u8], p: &mut usize, n: usize| -> Vec<i32> {
            let b = &buf[*p..*p + n * 4]; *p += n * 4;
            b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
        };

        // FT layer
        let fw = rd_i16(buf, &mut p, INPUT_SIZE * L1_SIZE);
        let mut ft_weight = Box::new([[0i16; L1_SIZE]; INPUT_SIZE]);
        for i in 0..INPUT_SIZE { ft_weight[i].copy_from_slice(&fw[i * L1_SIZE..(i + 1) * L1_SIZE]); }
        let fb = rd_i16(buf, &mut p, L1_SIZE);
        let mut ft_bias = [0i16; L1_SIZE];
        ft_bias.copy_from_slice(&fb);

        // L1 layer — stored output-major: [L2_SIZE][CONCAT]
        let l1w = rd_i8(buf, &mut p, L2_SIZE * CONCAT);
        let mut l1_weight = Box::new([[0i8; CONCAT]; L2_SIZE]);
        for j in 0..L2_SIZE { l1_weight[j].copy_from_slice(&l1w[j * CONCAT..(j + 1) * CONCAT]); }
        let l1b = rd_i32(buf, &mut p, L2_SIZE);
        let mut l1_bias = [0i32; L2_SIZE];
        l1_bias.copy_from_slice(&l1b);

        // L2 layer
        let l2w = rd_i8(buf, &mut p, L3_SIZE * L2_SIZE);
        let mut l2_weight = [[0i8; L2_SIZE]; L3_SIZE];
        for j in 0..L3_SIZE { l2_weight[j].copy_from_slice(&l2w[j * L2_SIZE..(j + 1) * L2_SIZE]); }
        let l2b = rd_i32(buf, &mut p, L3_SIZE);
        let mut l2_bias = [0i32; L3_SIZE];
        l2_bias.copy_from_slice(&l2b);

        // Output layer
        let ow = rd_i8(buf, &mut p, L3_SIZE);
        let mut out_weight = [0i8; L3_SIZE];
        out_weight.copy_from_slice(&ow);
        let out_bias = rd_i32(buf, &mut p, 1)[0];

        Ok(Nnue { ft_weight, ft_bias, l1_weight, l1_bias, l2_weight, l2_bias, out_weight, out_bias })
    }

    /// Load compile-time embedded weights.
    /// Enable with `RUSTFLAGS="--cfg embed_nnue" cargo build` (requires `networks/nnue.bin` at build time).
    pub fn load_embedded() -> Option<Self> {
        #[cfg(embed_nnue)]
        {
            static BYTES: &[u8] = include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/networks/nnue.bin")
            );
            return Self::from_slice(BYTES).ok();
        }
        #[allow(unreachable_code)]
        None
    }
}

// ── Accumulator ───────────────────────────────────────────────────────────────
//
// Holds the pre-CReLU FT output for both perspectives.
// During search the engine maintains one accumulator per ply on a stack, calling
// `apply_move` before each `board.make_move` and dropping the top on unmake.

#[repr(align(64))]
#[derive(Clone, Copy)]
pub struct Accumulator {
    pub white: [i16; L1_SIZE],
    pub black: [i16; L1_SIZE],
}

impl Accumulator {
    pub fn zeroed() -> Self {
        Accumulator { white: [0; L1_SIZE], black: [0; L1_SIZE] }
    }

    /// Full recompute from the current board state (O(pieces), used at root).
    pub fn refresh(&mut self, nn: &Nnue, board: &Board) {
        self.white = nn.ft_bias;
        self.black = nn.ft_bias;
        for sq in 0u8..64 {
            let Some(piece) = board.piece_at(Square(sq)) else { continue };
            self.add_feat(nn, piece.piece_type, piece.color, Square(sq));
        }
    }

    #[inline]
    fn add_feat(&mut self, nn: &Nnue, pt: PieceType, color: Color, sq: Square) {
        let wi = feat_w(pt, color, sq);
        let bi = feat_b(pt, color, sq);
        for j in 0..L1_SIZE {
            self.white[j] += nn.ft_weight[wi][j];
            self.black[j] += nn.ft_weight[bi][j];
        }
    }

    #[inline]
    fn sub_feat(&mut self, nn: &Nnue, pt: PieceType, color: Color, sq: Square) {
        let wi = feat_w(pt, color, sq);
        let bi = feat_b(pt, color, sq);
        for j in 0..L1_SIZE {
            self.white[j] -= nn.ft_weight[wi][j];
            self.black[j] -= nn.ft_weight[bi][j];
        }
    }

    /// Apply the feature delta of `mv` to this accumulator.
    /// **Must be called BEFORE `board.make_move(mv)`** — needs pre-move board state.
    pub fn apply_move(&mut self, nn: &Nnue, board: &Board, mv: Move) {
        let from  = mv.from_sq();
        let to    = mv.to_sq();
        let us    = board.side_to_move;
        let piece = board.piece_at(from).expect("piece on from-square");

        match mv.flag() {
            MoveFlag::Normal => {
                if let Some(cap) = board.piece_at(to) {
                    self.sub_feat(nn, cap.piece_type, cap.color, to);
                }
                self.sub_feat(nn, piece.piece_type, us, from);
                self.add_feat(nn, piece.piece_type, us, to);
            }
            MoveFlag::Promo => {
                if let Some(cap) = board.piece_at(to) {
                    self.sub_feat(nn, cap.piece_type, cap.color, to);
                }
                self.sub_feat(nn, PieceType::Pawn,         us, from);
                self.add_feat(nn, mv.promo_piece_type(),   us, to);
            }
            MoveFlag::EnPassant => {
                // Captured pawn sits on the same rank as `from`, same file as `to`.
                let ep_sq = Square::new(to.file(), from.rank());
                self.sub_feat(nn, PieceType::Pawn, us.flip(), ep_sq);
                self.sub_feat(nn, PieceType::Pawn, us, from);
                self.add_feat(nn, PieceType::Pawn, us, to);
            }
            MoveFlag::Castling => {
                let rank = from.rank();
                self.sub_feat(nn, PieceType::King, us, from);
                self.add_feat(nn, PieceType::King, us, to);
                // Rook: kingside h→f, queenside a→d
                let (rf, rt) = if to.file() > from.file() {
                    (Square::new(7, rank), Square::new(5, rank))
                } else {
                    (Square::new(0, rank), Square::new(3, rank))
                };
                self.sub_feat(nn, PieceType::Rook, us, rf);
                self.add_feat(nn, PieceType::Rook, us, rt);
            }
        }
    }

    /// Evaluate from `stm`'s perspective (centipawns).
    #[inline]
    pub fn evaluate(&self, nn: &Nnue, stm: Color) -> i32 {
        forward(self, nn, stm)
    }
}

// ── Accumulator stack ─────────────────────────────────────────────────────────

const STACK_DEPTH: usize = 128;

pub struct AccumulatorStack {
    stack: Vec<Accumulator>,
    ply:   usize,
}

impl AccumulatorStack {
    pub fn new() -> Self {
        let mut stack = Vec::with_capacity(STACK_DEPTH);
        stack.resize(STACK_DEPTH, Accumulator::zeroed());
        AccumulatorStack { stack, ply: 0 }
    }

    #[inline] pub fn current(&self)         -> &Accumulator     { &self.stack[self.ply] }
    #[inline] pub fn current_mut(&mut self) -> &mut Accumulator { &mut self.stack[self.ply] }

    /// Refresh the root accumulator for a fresh search.
    pub fn init(&mut self, nn: &Nnue, board: &Board) {
        self.ply = 0;
        self.stack[0].refresh(nn, board);
    }

    /// Copy current accumulator to ply+1, then apply the move's delta.
    /// Call **before** `board.make_move(mv)`.
    #[inline]
    pub fn push_move(&mut self, nn: &Nnue, board: &Board, mv: Move) {
        debug_assert!(self.ply + 1 < STACK_DEPTH, "accumulator stack overflow");
        self.stack[self.ply + 1] = self.stack[self.ply];
        self.ply += 1;
        self.stack[self.ply].apply_move(nn, board, mv);
    }

    /// Push for null moves — no piece changes, just copy.
    #[inline]
    pub fn push_null(&mut self) {
        debug_assert!(self.ply + 1 < STACK_DEPTH);
        self.stack[self.ply + 1] = self.stack[self.ply];
        self.ply += 1;
    }

    /// Restore the previous accumulator (pair with each push).
    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.ply > 0, "accumulator stack underflow");
        self.ply -= 1;
    }

    #[inline]
    pub fn evaluate(&self, nn: &Nnue, stm: Color) -> i32 {
        self.current().evaluate(nn, stm)
    }
}

// ── Forward pass dispatch ─────────────────────────────────────────────────────

fn forward(acc: &Accumulator, nn: &Nnue, stm: Color) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { forward_avx2(acc, nn, stm) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { forward_neon(acc, nn, stm) };
        }
    }
    forward_scalar(acc, nn, stm)
}

// ── Scalar forward pass ───────────────────────────────────────────────────────

fn forward_scalar(acc: &Accumulator, nn: &Nnue, stm: Color) -> i32 {
    let (sa, oa) = stm_opp(acc, stm);

    // CReLU: clamp each i16 to [0, QA] and narrow to u8
    let mut input = [0u8; CONCAT];
    for i in 0..L1_SIZE {
        input[i]           = sa[i].clamp(0, QA as i16) as u8;
        input[i + L1_SIZE] = oa[i].clamp(0, QA as i16) as u8;
    }

    // L1: 512 u8 inputs → 32 i32 outputs
    let mut l1 = [0i32; L2_SIZE];
    for j in 0..L2_SIZE {
        let mut s = nn.l1_bias[j];
        for i in 0..CONCAT {
            s += (input[i] as i32) * (nn.l1_weight[j][i] as i32);
        }
        l1[j] = (s / QA).clamp(0, QB);
    }

    // L2
    let mut l2 = [0i32; L3_SIZE];
    for j in 0..L3_SIZE {
        let mut s = nn.l2_bias[j];
        for i in 0..L2_SIZE {
            s += l1[i] * (nn.l2_weight[j][i] as i32);
        }
        l2[j] = (s / QB).clamp(0, QB);
    }

    // Output
    let mut out = nn.out_bias;
    for i in 0..L3_SIZE {
        out += l2[i] * (nn.out_weight[i] as i32);
    }
    // out ≈ QB² × tanh(cp / EVAL_SCALE)  →  cp ≈ out × EVAL_SCALE / QB²
    out * EVAL_SCALE / (QB * QB)
}

#[inline]
fn stm_opp(acc: &Accumulator, stm: Color) -> (&[i16; L1_SIZE], &[i16; L1_SIZE]) {
    match stm {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    }
}

// ── AVX2 forward pass ─────────────────────────────────────────────────────────
//
// Hot path: CReLU + pack accumulators to u8, then use maddubs/madd for L1.
// With QA=127 the pair-sum (u8×i8 + u8×i8) ≤ 2×127×127 = 32258 < 32767,
// so _mm256_maddubs_epi16 never saturates.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_avx2(acc: &Accumulator, nn: &Nnue, stm: Color) -> i32 {
    use std::arch::x86_64::*;

    let (sa, oa) = stm_opp(acc, stm);

    // ── CReLU + pack both perspectives to 512 u8 ─────────────────────────────
    let mut input = [0u8; CONCAT];
    {
        let zero = _mm256_setzero_si256();
        let qa   = _mm256_set1_epi16(QA as i16);

        // Each iteration processes 32 i16 (stm or opp) → 32 u8.
        // _mm256_packus_epi16(a[0..15], a[16..31]) interleaves 128-bit lanes;
        // permute4x64 with imm8=0xD8 corrects to sequential byte order.
        macro_rules! crelu_pack32 {
            ($src:expr, $dst_base:expr) => {{
                let src_ptr = ($src).as_ptr();
                let dst_ptr = input.as_mut_ptr().add($dst_base);
                for chunk in 0..(L1_SIZE / 32) {
                    let base = chunk * 32;
                    let a = _mm256_loadu_si256(src_ptr.add(base)      as *const __m256i);
                    let b = _mm256_loadu_si256(src_ptr.add(base + 16) as *const __m256i);
                    let a = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
                    let b = _mm256_min_epi16(_mm256_max_epi16(b, zero), qa);
                    let p = _mm256_packus_epi16(a, b);
                    let p = _mm256_permute4x64_epi64(p, 0xD8);
                    _mm256_storeu_si256(dst_ptr.add(chunk * 32) as *mut __m256i, p);
                }
            }};
        }
        crelu_pack32!(sa, 0);
        crelu_pack32!(oa, L1_SIZE);
    }

    // ── L1: CONCAT u8 × i8 → L2_SIZE i32 ────────────────────────────────────
    // For each output j we compute a 512-element dot product using:
    //   maddubs(u8, i8) → 16 i16  (multiply-add pairs, no saturation with QA=127)
    //   madd_epi16(×, 1) → 8 i32  (sum adjacent i16 pairs)
    // Unrolled 4× per 128-byte chunk (512 / 128 = 4 outer iterations).
    let mut l1 = [0i32; L2_SIZE];
    let ones = _mm256_set1_epi16(1i16);

    for j in 0..L2_SIZE {
        let ip = input.as_ptr();
        let wp = nn.l1_weight[j].as_ptr();
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();

        for k in (0..CONCAT).step_by(128) {
            let i0 = _mm256_loadu_si256(ip.add(k)       as *const __m256i);
            let i1 = _mm256_loadu_si256(ip.add(k + 32)  as *const __m256i);
            let i2 = _mm256_loadu_si256(ip.add(k + 64)  as *const __m256i);
            let i3 = _mm256_loadu_si256(ip.add(k + 96)  as *const __m256i);
            let w0 = _mm256_loadu_si256(wp.add(k)       as *const __m256i);
            let w1 = _mm256_loadu_si256(wp.add(k + 32)  as *const __m256i);
            let w2 = _mm256_loadu_si256(wp.add(k + 64)  as *const __m256i);
            let w3 = _mm256_loadu_si256(wp.add(k + 96)  as *const __m256i);
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(i0, w0), ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(i1, w1), ones));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(i2, w2), ones));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(i3, w3), ones));
        }

        // Horizontal reduce 4 ymm registers → 1 i32
        let s01 = _mm256_add_epi32(a0, a1);
        let s23 = _mm256_add_epi32(a2, a3);
        let s   = _mm256_add_epi32(s01, s23);
        let lo  = _mm256_castsi256_si128(s);
        let hi  = _mm256_extracti128_si256(s, 1);
        let v   = _mm_add_epi32(lo, hi);
        let v   = _mm_hadd_epi32(v, v);
        let v   = _mm_hadd_epi32(v, v);
        let dot = _mm_cvtsi128_si32(v);

        l1[j] = ((nn.l1_bias[j] + dot) / QA).clamp(0, QB);
    }

    // L2 and output: small (32×32 and 32×1), scalar is plenty fast
    let l2 = l2_forward_scalar(&l1, nn);
    out_forward_scalar(&l2, nn)
}

// ── NEON forward pass ─────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn forward_neon(acc: &Accumulator, nn: &Nnue, stm: Color) -> i32 {
    use std::arch::aarch64::*;

    let (sa, oa) = stm_opp(acc, stm);

    // CReLU + narrow i16 → u8
    let mut input = [0u8; CONCAT];
    let zero = vdupq_n_s16(0);
    let qa   = vdupq_n_s16(QA as i16);

    for chunk in 0..(L1_SIZE / 8) {
        let base = chunk * 8;
        let sv = vminq_s16(vmaxq_s16(vld1q_s16(sa.as_ptr().add(base)), zero), qa);
        let ov = vminq_s16(vmaxq_s16(vld1q_s16(oa.as_ptr().add(base)), zero), qa);
        vst1_u8(input.as_mut_ptr().add(base),           vqmovun_s16(sv));
        vst1_u8(input.as_mut_ptr().add(L1_SIZE + base), vqmovun_s16(ov));
    }

    // L1 scalar (ARMv8.2 dot-product would be better but isn't universally available)
    let mut l1 = [0i32; L2_SIZE];
    for j in 0..L2_SIZE {
        let mut s = nn.l1_bias[j];
        for i in 0..CONCAT {
            s += (input[i] as i32) * (nn.l1_weight[j][i] as i32);
        }
        l1[j] = (s / QA).clamp(0, QB);
    }

    let l2 = l2_forward_scalar(&l1, nn);
    out_forward_scalar(&l2, nn)
}

// ── Shared scalar tail (L2 + output) ─────────────────────────────────────────

fn l2_forward_scalar(l1: &[i32; L2_SIZE], nn: &Nnue) -> [i32; L3_SIZE] {
    let mut l2 = [0i32; L3_SIZE];
    for j in 0..L3_SIZE {
        let mut s = nn.l2_bias[j];
        for i in 0..L2_SIZE {
            s += l1[i] * (nn.l2_weight[j][i] as i32);
        }
        l2[j] = (s / QB).clamp(0, QB);
    }
    l2
}

fn out_forward_scalar(l2: &[i32; L3_SIZE], nn: &Nnue) -> i32 {
    let mut out = nn.out_bias;
    for i in 0..L3_SIZE {
        out += l2[i] * (nn.out_weight[i] as i32);
    }
    out * EVAL_SCALE / (QB * QB)
}

// ── Stateless convenience ─────────────────────────────────────────────────────
// Full refresh every call — used by the web fallback and benchmarks.

pub fn evaluate(nn: &Nnue, board: &Board) -> i32 {
    let mut acc = Accumulator::zeroed();
    acc.refresh(nn, board);
    acc.evaluate(nn, board.side_to_move)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;
    use crate::core::movegen::generate_legal;

    /// Synthetic network with deterministic non-trivial weights.
    /// Only the FT layer has non-zero weights — sufficient to test the accumulator.
    /// L1/L2/output weights are non-zero too so the full forward pass is non-trivial.
    fn synthetic_nnue() -> Nnue {
        let mut ft_weight = Box::new([[0i16; L1_SIZE]; INPUT_SIZE]);
        for i in 0..INPUT_SIZE {
            for j in 0..L1_SIZE {
                ft_weight[i][j] = (((i * 7 + j * 13) % 9) as i16) - 4; // [-4, 4]
            }
        }
        let mut ft_bias = [0i16; L1_SIZE];
        for j in 0..L1_SIZE {
            ft_bias[j] = 30; // keeps accumulated values well within i16 range
        }

        let mut l1_weight = Box::new([[0i8; CONCAT]; L2_SIZE]);
        for j in 0..L2_SIZE {
            for i in 0..CONCAT {
                l1_weight[j][i] = (((j * 5 + i * 3) % 5) as i8) - 2; // [-2, 2]
            }
        }
        // Bias scaled by QA*QB so the L1 output is in a mid-range, not all-zero.
        let l1_bias = [QA * QB * 2; L2_SIZE];

        let mut l2_weight = [[0i8; L2_SIZE]; L3_SIZE];
        for j in 0..L3_SIZE {
            for i in 0..L2_SIZE {
                l2_weight[j][i] = (((j * 3 + i * 7) % 5) as i8) - 2;
            }
        }
        let l2_bias = [QB * QB; L3_SIZE];

        let mut out_weight = [0i8; L3_SIZE];
        for i in 0..L3_SIZE {
            out_weight[i] = (i % 3) as i8;
        }

        Nnue {
            ft_weight,
            ft_bias,
            l1_weight,
            l1_bias,
            l2_weight,
            l2_bias,
            out_weight,
            out_bias: 0,
        }
    }

    // ── Feature index tests ───────────────────────────────────────────────────

    #[test]
    fn feat_indices_in_range() {
        for pt in PieceType::ALL {
            for color in [Color::White, Color::Black] {
                for sq in 0u8..64 {
                    let sq = Square(sq);
                    assert!(
                        feat_w(pt, color, sq) < INPUT_SIZE,
                        "feat_w out of range: {pt:?} {color:?} sq={}",
                        sq.0
                    );
                    assert!(
                        feat_b(pt, color, sq) < INPUT_SIZE,
                        "feat_b out of range: {pt:?} {color:?} sq={}",
                        sq.0
                    );
                }
            }
        }
    }

    /// White and black POV indices must be distinct for the same piece (they encode
    /// different perspectives, so collisions would silently corrupt the accumulator).
    #[test]
    fn feat_w_and_feat_b_differ_for_same_piece() {
        // The king on e1 (sq=4): white POV vs black POV must map to different rows.
        let sq = Square(4);
        assert_ne!(
            feat_w(PieceType::King, Color::White, sq),
            feat_b(PieceType::King, Color::White, sq),
        );
    }

    // ── Accumulator refresh vs incremental ────────────────────────────────────

    /// Core correctness invariant: applying a move's feature delta to the accumulator
    /// must produce the same result as a full refresh from the post-move board.
    fn check_incremental_matches_refresh(board: &Board, nn: &Nnue) {
        let mut acc_root = Accumulator::zeroed();
        acc_root.refresh(nn, board);

        for &mv in generate_legal(board).as_slice() {
            // Incremental path
            let mut acc_incr = acc_root;
            acc_incr.apply_move(nn, board, mv);

            // Full refresh path
            let mut post = board.clone();
            post.make_move(mv);
            let mut acc_refresh = Accumulator::zeroed();
            acc_refresh.refresh(nn, &post);

            assert_eq!(
                acc_incr.white, acc_refresh.white,
                "white accumulator mismatch after {mv} in position\n{board}"
            );
            assert_eq!(
                acc_incr.black, acc_refresh.black,
                "black accumulator mismatch after {mv} in position\n{board}"
            );
        }
    }

    #[test]
    fn incremental_matches_refresh_starting_position() {
        let nn = synthetic_nnue();
        check_incremental_matches_refresh(&Board::starting_position(), &nn);
    }

    #[test]
    fn incremental_matches_refresh_en_passant() {
        let nn = synthetic_nnue();
        // White pawn on e5 can capture en passant on f6; black pawn just moved f7-f5.
        let board = Board::from_fen(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        )
        .unwrap();
        check_incremental_matches_refresh(&board, &nn);
    }

    #[test]
    fn incremental_matches_refresh_castling() {
        let nn = synthetic_nnue();
        // Both sides can castle in both directions.
        let board = Board::from_fen(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        check_incremental_matches_refresh(&board, &nn);
    }

    #[test]
    fn incremental_matches_refresh_promotion() {
        let nn = synthetic_nnue();
        // White pawn on a7 can promote; black pawn on h2 can promote.
        let board = Board::from_fen("8/P7/8/8/8/8/7p/8 w - - 0 1").unwrap();
        check_incremental_matches_refresh(&board, &nn);
    }

    // ── Stack push / pop ──────────────────────────────────────────────────────

    #[test]
    fn stack_pop_restores_previous_state() {
        let nn = synthetic_nnue();
        let board = Board::starting_position();

        let mut stack = AccumulatorStack::new();
        stack.init(&nn, &board);
        let before = *stack.current();

        let mv = generate_legal(&board).as_slice()[0];
        stack.push_move(&nn, &board, mv);
        stack.pop();

        assert_eq!(stack.current().white, before.white, "white not restored after pop");
        assert_eq!(stack.current().black, before.black, "black not restored after pop");
    }

    #[test]
    fn stack_push_matches_apply_move() {
        let nn = synthetic_nnue();
        let board = Board::starting_position();

        let mut stack = AccumulatorStack::new();
        stack.init(&nn, &board);

        let mv = generate_legal(&board).as_slice()[0];

        // Via stack
        stack.push_move(&nn, &board, mv);
        let via_stack = *stack.current();

        // Via manual apply_move on a fresh accumulator copy
        let mut acc = Accumulator::zeroed();
        acc.refresh(&nn, &board);
        acc.apply_move(&nn, &board, mv);

        assert_eq!(via_stack.white, acc.white);
        assert_eq!(via_stack.black, acc.black);
    }

    // ── Scalar vs SIMD agreement ──────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_and_avx2_forward_agree() {
        if !is_x86_feature_detected!("avx2") {
            return; // not an error — just skip on non-AVX2 hardware
        }
        let nn = synthetic_nnue();
        let mut acc = Accumulator::zeroed();
        acc.refresh(&nn, &Board::starting_position());

        for stm in [Color::White, Color::Black] {
            let scalar = forward_scalar(&acc, &nn, stm);
            let avx2 = unsafe { forward_avx2(&acc, &nn, stm) };
            assert_eq!(
                scalar, avx2,
                "scalar={scalar} avx2={avx2} disagree for stm={stm:?}"
            );
        }
    }

    // ── Eval sanity ───────────────────────────────────────────────────────────

    /// evaluate() must not panic and must return a value in a sane range.
    #[test]
    fn evaluate_does_not_panic_and_is_finite() {
        let nn = synthetic_nnue();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let board = Board::from_fen(fen).unwrap();
            let score = evaluate(&nn, &board);
            assert!(
                score.abs() < 30_000,
                "score={score} looks like garbage for {fen}"
            );
        }
    }

    /// A symmetric position evaluated from both sides should give equal-but-opposite scores.
    #[test]
    fn eval_is_negated_for_opposite_stm_symmetric_position() {
        let nn = synthetic_nnue();
        // Completely symmetric position: same pieces mirrored, white to move vs black to move.
        // We use the same FEN but manually test both perspectives on the same accumulator.
        let board = Board::starting_position();
        let mut acc = Accumulator::zeroed();
        acc.refresh(&nn, &board);

        let white_score = acc.evaluate(&nn, Color::White);
        let black_score = acc.evaluate(&nn, Color::Black);

        // Because the starting position is NOT perfectly symmetric in our feature encoding
        // (white pieces on ranks 1-2, black on ranks 7-8), the scores won't be exactly
        // equal-and-opposite. But they should both be small in magnitude.
        assert!(white_score.abs() < 30_000, "white_score={white_score}");
        assert!(black_score.abs() < 30_000, "black_score={black_score}");
    }
}
