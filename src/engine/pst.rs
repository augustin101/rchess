// Piece-square tables (PSTs) for middlegame (MG) and endgame (EG).
// All tables are from White's perspective, indexed by Square (a1=0 … h8=63).
// For Black pieces, mirror vertically: pst[sq ^ 56].
// Values in centipawns.

// ── Game phase weights ────────────────────────────────────────────────────────
// Summing these over all pieces on the board gives current_phase ∈ [0, PHASE_MAX].
// PHASE_MAX = fully-loaded game; 0 = bare kings (pure endgame).

pub const PHASE_QUEEN:  i32 = 4;
pub const PHASE_ROOK:   i32 = 2;
pub const PHASE_BISHOP: i32 = 1;
pub const PHASE_KNIGHT: i32 = 1;
/// 2Q + 4R + 4B + 4N per side.
pub const PHASE_MAX:    i32 = 24;

// ── Material values (phase-interpolated) ─────────────────────────────────────

pub const PAWN_MG:   i32 = 100;
pub const KNIGHT_MG: i32 = 320;
pub const BISHOP_MG: i32 = 330;
pub const ROOK_MG:   i32 = 500;
pub const QUEEN_MG:  i32 = 900;

pub const PAWN_EG:   i32 = 110; // extra value near promotion
pub const KNIGHT_EG: i32 = 300; // worse in open endgames
pub const BISHOP_EG: i32 = 340; // better in open endgames
pub const ROOK_EG:   i32 = 530;
pub const QUEEN_EG:  i32 = 940;

// ── Piece-square tables ───────────────────────────────────────────────────────
// Row order: rank 1 (indices 0-7) → rank 8 (indices 56-63).

#[rustfmt::skip]
pub const PAWN_PST_MG: [i32; 64] = [
//   a    b    c    d    e    f    g    h
     0,   0,   0,   0,   0,   0,   0,   0,  // rank 1 – unreachable for pawns
     5,  10,  10, -20, -20,  10,  10,   5,  // rank 2 – starting rank
     5,  -5, -10,   0,   0, -10,  -5,   5,  // rank 3
     0,   0,   0,  20,  20,   0,   0,   0,  // rank 4
     5,   5,  10,  25,  25,  10,   5,   5,  // rank 5
    10,  10,  20,  30,  30,  20,  10,  10,  // rank 6
    50,  50,  50,  50,  50,  50,  50,  50,  // rank 7 – about to promote
     0,   0,   0,   0,   0,   0,   0,   0,  // rank 8
];

/// In the endgame, pure advancement matters most; central vs. edge matters less.
#[rustfmt::skip]
pub const PAWN_PST_EG: [i32; 64] = [
     0,   0,   0,   0,   0,   0,   0,   0,  // rank 1
     0,   0,   0,   0,   0,   0,   0,   0,  // rank 2
     5,   5,   5,   5,   5,   5,   5,   5,  // rank 3
    10,  10,  10,  10,  10,  10,  10,  10,  // rank 4
    20,  20,  20,  20,  20,  20,  20,  20,  // rank 5
    35,  35,  35,  35,  35,  35,  35,  35,  // rank 6
    55,  55,  55,  55,  55,  55,  55,  55,  // rank 7
     0,   0,   0,   0,   0,   0,   0,   0,  // rank 8
];

#[rustfmt::skip]
pub const KNIGHT_PST_MG: [i32; 64] = [
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -30,   5,  10,  15,  15,  10,   5, -30,
   -30,   0,  15,  20,  20,  15,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
pub const KNIGHT_PST_EG: [i32; 64] = [
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
pub const BISHOP_PST_MG: [i32; 64] = [
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   5,   0,   0,   0,   0,   5, -10,
   -10,  10,  10,  10,  10,  10,  10, -10,
   -10,   0,  10,  10,  10,  10,   0, -10,
   -10,   5,   5,  10,  10,   5,   5, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10, -10, -10, -10, -10, -20,
];

/// Open diagonals are even more valuable when the position opens in the endgame.
#[rustfmt::skip]
pub const BISHOP_PST_EG: [i32; 64] = [
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,  10,  15,  15,  10,   0, -10,
   -10,   0,  10,  15,  15,  10,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
pub const ROOK_PST_MG: [i32; 64] = [
     0,   0,   0,   5,   5,   0,   0,   0,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     5,  10,  10,  10,  10,  10,  10,   5,
     0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
pub const ROOK_PST_EG: [i32; 64] = [
     5,   5,   5,   5,   5,   5,   5,   5,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     5,   5,   5,   5,   5,   5,   5,   5,
];

#[rustfmt::skip]
pub const QUEEN_PST_MG: [i32; 64] = [
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -10,   5,   5,   5,   5,   5,   0, -10,
     0,   0,   5,   5,   5,   5,   0,  -5,
    -5,   0,   5,   5,   5,   5,   0,  -5,
   -10,   0,   5,   5,   5,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10,  -5,  -5, -10, -10, -20,
];

#[rustfmt::skip]
pub const QUEEN_PST_EG: [i32; 64] = [
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,  10,  10,   5,   0,  -5,
    -5,   0,   5,  10,  10,   5,   0,  -5,
   -10,   0,   5,   5,   5,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10,  -5,  -5, -10, -10, -20,
];

/// MG: king hides behind pawns. g1/c1 (castled) rewarded; e1 neutral; center punished.
#[rustfmt::skip]
pub const KING_PST_MG: [i32; 64] = [
    20,  30,  10,   0,   0,  10,  30,  20,  // rank 1 – back rank
    20,  20,   0,   0,   0,   0,  20,  20,  // rank 2
   -10, -20, -20, -20, -20, -20, -20, -10,  // rank 3
   -20, -30, -30, -40, -40, -30, -30, -20,  // rank 4
   -30, -40, -40, -50, -50, -40, -40, -30,  // rank 5
   -30, -40, -40, -50, -50, -40, -40, -30,  // rank 6
   -30, -40, -40, -50, -50, -40, -40, -30,  // rank 7
   -30, -40, -40, -50, -50, -40, -40, -30,  // rank 8
];

/// EG: king becomes an active attacking piece — centralization strongly rewarded.
#[rustfmt::skip]
pub const KING_PST_EG: [i32; 64] = [
   -50, -30, -30, -30, -30, -30, -30, -50,  // rank 1 – back rank poor in EG
   -30, -10,   0,   0,   0,   0, -10, -30,  // rank 2
   -30,   0,  20,  25,  25,  20,   0, -30,  // rank 3
   -30,   5,  25,  30,  30,  25,   5, -30,  // rank 4
   -30,   5,  25,  30,  30,  25,   5, -30,  // rank 5
   -30,   0,  20,  25,  25,  20,   0, -30,  // rank 6
   -30, -10,   0,   0,   0,   0, -10, -30,  // rank 7
   -50, -30, -30, -30, -30, -30, -30, -50,  // rank 8
];
