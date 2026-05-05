use std::sync::OnceLock;
use super::bitboard::*;
use super::types::{Color, Square};

// ── Magic constants ───────────────────────────────────────────────────────────
// Source: Chess Programming Wiki / Pradyumna Kannan's tables.

const ROOK_MAGICS: [u64; 64] = [
    0x8a80104000800020, 0x140002000100040,  0x2801880a0017001,  0x100081001000420,
    0x200020010080420,  0x3001c0002010008,  0x8480008002000100, 0x2080088004402900,
    0x800098204000,     0x2024401000200040, 0x100802000801000,  0x120800800801000,
    0x208808088000400,  0x2802200800400,    0x2200800100020080, 0x801000060821100,
    0x80044006422000,   0x100808020004000,  0x12108a0010204200, 0x140848010000802,
    0x481828014002800,  0x8094004002004100, 0x4010040010010802, 0x20008806104,
    0x100400080208000,  0x2040002120081000, 0x21200680100081,   0x20100080080080,
    0x2000a00200410,    0x20080800400,      0x80088400100102,   0x80004600042881,
    0x4040008040800020, 0x440003000200801,  0x4200011004500,    0x188020010100100,
    0x14800401802800,   0x2080040080800200, 0x124080204001001,  0x200046502000484,
    0x480400080088020,  0x1000422010034000, 0x30200100110040,   0x100021010009,
    0x2002080100110004, 0x202008004008002,  0x20020004010100,   0x2048440040820001,
    0x101002200408200,  0x40802000401080,   0x4008142004410100, 0x2060820c0120200,
    0x1001004080100,    0x20c020080040080,  0x2935610830022400, 0x44440041009200,
    0x280001040802101,  0x2100190040002085, 0x80c0084100102001, 0x4024081001000421,
    0x20030a0244872,    0x12001008414402,   0x2006104900a0804,  0x1004081002402,
];

const BISHOP_MAGICS: [u64; 64] = [
    0x40040844404084,   0x2004208a004208,   0x10190041080202,   0x108060845042010,
    0x581104180800210,  0x2112080446200010, 0x1080820820060210, 0x3c0808410220200,
    0x4050404440404,    0x21001420088,      0x24d0080801082102, 0x1020a0a020400,
    0x40308200402,      0x4011002100800,    0x401484104104005,  0x801010402020200,
    0x400210c3880100,   0x404022024108200,  0x810018200204102,  0x4002801a02003,
    0x85040820080400,   0x810102c808880400, 0xe900410884800,    0x8002020480840102,
    0x220200865090201,  0x2010100a02021202, 0x152048408022401,  0x20080002081110,
    0x4001001021004000, 0x800040400a011002, 0xe4004081011002,   0x1c004001012080,
    0x8004200962a00220, 0x8422100208500202, 0x2000402200300c08, 0x8646020080080080,
    0x80020a0200100808, 0x2010004880111000, 0x623000a080011400, 0x42008c0340209202,
    0x209188240001000,  0x400408a884001800, 0x110400a6080400,   0x1840060a44020800,
    0x90080104000041,   0x201011000808101,  0x1a2208080504f080, 0x8012020600211212,
    0x500861011240000,  0x180806108200800,  0x4000020e01040044, 0x300000261044000a,
    0x802241102020002,  0x20906061210001,   0x5a84841004010310, 0x4010801011c04,
    0xa010109502200,    0x4a02012000,       0x500201010098b028, 0x8040002811040900,
    0x28000010020204,   0x6000020202d0240,  0x8918844842082200, 0x4010011029020020,
];

// ── Relevant bit counts (popcount of mask) ────────────────────────────────────

const ROOK_BITS: [u32; 64] = [
    12,11,11,11,11,11,11,12,
    11,10,10,10,10,10,10,11,
    11,10,10,10,10,10,10,11,
    11,10,10,10,10,10,10,11,
    11,10,10,10,10,10,10,11,
    11,10,10,10,10,10,10,11,
    11,10,10,10,10,10,10,11,
    12,11,11,11,11,11,11,12,
];

const BISHOP_BITS: [u32; 64] = [
    6,5,5,5,5,5,5,6,
    5,5,5,5,5,5,5,5,
    5,5,7,7,7,7,5,5,
    5,5,7,9,9,7,5,5,
    5,5,7,9,9,7,5,5,
    5,5,7,7,7,7,5,5,
    5,5,5,5,5,5,5,5,
    6,5,5,5,5,5,5,6,
];

// ── Magic entry ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct MagicEntry {
    mask:   Bitboard,
    magic:  u64,
    shift:  u32,
    offset: u32,
}

// ── Attack tables ─────────────────────────────────────────────────────────────

pub struct Attacks {
    pub pawn:   [[Bitboard; 64]; 2],
    pub knight: [Bitboard; 64],
    pub king:   [Bitboard; 64],
    rook_magic:   [MagicEntry; 64],
    bishop_magic: [MagicEntry; 64],
    rook_table:   Vec<Bitboard>,
    bishop_table: Vec<Bitboard>,
}

static ATTACKS: OnceLock<Attacks> = OnceLock::new();

pub fn get() -> &'static Attacks {
    ATTACKS.get_or_init(Attacks::init)
}

// ── Public query functions ────────────────────────────────────────────────────

#[inline]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    get().pawn[color as usize][sq.0 as usize]
}

#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    get().knight[sq.0 as usize]
}

#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    get().king[sq.0 as usize]
}

#[inline]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = get();
    let e = &t.bishop_magic[sq.0 as usize];
    let idx = ((occ & e.mask).wrapping_mul(e.magic) >> e.shift) as usize;
    t.bishop_table[e.offset as usize + idx]
}

#[inline]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let t = get();
    let e = &t.rook_magic[sq.0 as usize];
    let idx = ((occ & e.mask).wrapping_mul(e.magic) >> e.shift) as usize;
    t.rook_table[e.offset as usize + idx]
}

#[inline]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

// ── Initialisation ────────────────────────────────────────────────────────────

impl Attacks {
    fn init() -> Self {
        let dummy = MagicEntry { mask: 0, magic: 0, shift: 0, offset: 0 };
        let mut a = Attacks {
            pawn:         [[0; 64]; 2],
            knight:       [0; 64],
            king:         [0; 64],
            rook_magic:   [dummy; 64],
            bishop_magic: [dummy; 64],
            rook_table:   Vec::new(),
            bishop_table: Vec::new(),
        };
        a.init_pawn_attacks();
        a.init_knight_attacks();
        a.init_king_attacks();
        a.init_slider_tables();
        a
    }

    fn init_pawn_attacks(&mut self) {
        for sq in 0u8..64 {
            let bb = 1u64 << sq;
            // White pawn attacks: north-east and north-west
            self.pawn[Color::White as usize][sq as usize] =
                ((bb & !FILE_H) << 9) | ((bb & !FILE_A) << 7);
            // Black pawn attacks: south-east and south-west
            self.pawn[Color::Black as usize][sq as usize] =
                ((bb & !FILE_H) >> 7) | ((bb & !FILE_A) >> 9);
        }
    }

    fn init_knight_attacks(&mut self) {
        for sq in 0u8..64 {
            let bb = 1u64 << sq;
            self.knight[sq as usize] =
                  ((bb & !FILE_A & !FILE_B) >> 10)  // WSW
                | ((bb & !FILE_A)           >> 17)  // NNW (south dir)
                | ((bb & !FILE_H)           >> 15)  // NNE (south dir)
                | ((bb & !FILE_G & !FILE_H) >> 6)   // ENE (south dir)
                | ((bb & !FILE_G & !FILE_H) << 10)  // ESE
                | ((bb & !FILE_H)           << 17)  // SSE (north dir)
                | ((bb & !FILE_A)           << 15)  // SSW (north dir)
                | ((bb & !FILE_A & !FILE_B) << 6);  // WNW
        }
    }

    fn init_king_attacks(&mut self) {
        for sq in 0u8..64 {
            let bb = 1u64 << sq;
            self.king[sq as usize] =
                  ((bb & !FILE_A) >> 1)   // W
                | ((bb & !FILE_H) << 1)   // E
                | (bb >> 8)               // S
                | (bb << 8)               // N
                | ((bb & !FILE_A) >> 9)   // SW
                | ((bb & !FILE_H) >> 7)   // SE
                | ((bb & !FILE_A) << 7)   // NW
                | ((bb & !FILE_H) << 9);  // NE
        }
    }

    fn init_slider_tables(&mut self) {
        // ── Rooks ──
        let mut offset = 0u32;
        for sq in 0u8..64 {
            let mask  = rook_mask(Square(sq));
            let bits  = ROOK_BITS[sq as usize];
            let shift = 64 - bits;
            let size  = 1usize << bits;
            self.rook_magic[sq as usize] = MagicEntry {
                mask, magic: ROOK_MAGICS[sq as usize], shift, offset,
            };
            self.rook_table.resize(offset as usize + size, EMPTY);
            fill_table(
                Square(sq), mask, ROOK_MAGICS[sq as usize], shift,
                offset, &mut self.rook_table, false,
            );
            offset += size as u32;
        }

        // ── Bishops ──
        let mut offset = 0u32;
        for sq in 0u8..64 {
            let mask  = bishop_mask(Square(sq));
            let bits  = BISHOP_BITS[sq as usize];
            let shift = 64 - bits;
            let size  = 1usize << bits;
            self.bishop_magic[sq as usize] = MagicEntry {
                mask, magic: BISHOP_MAGICS[sq as usize], shift, offset,
            };
            self.bishop_table.resize(offset as usize + size, EMPTY);
            fill_table(
                Square(sq), mask, BISHOP_MAGICS[sq as usize], shift,
                offset, &mut self.bishop_table, true,
            );
            offset += size as u32;
        }
    }
}

// ── Helper: fill one square's magic table entries ────────────────────────────

fn fill_table(
    sq:     Square,
    mask:   Bitboard,
    magic:  u64,
    shift:  u32,
    offset: u32,
    table:  &mut Vec<Bitboard>,
    is_bishop: bool,
) {
    // Carry-Rippler enumeration of all subsets of mask
    let mut subset: Bitboard = 0;
    loop {
        let attacks = if is_bishop {
            slow_bishop_attacks(sq, subset)
        } else {
            slow_rook_attacks(sq, subset)
        };
        let idx = (subset.wrapping_mul(magic) >> shift) as usize;
        table[offset as usize + idx] = attacks;
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 { break; }
    }
}

// ── Helper: slow ray-casting (init only) ─────────────────────────────────────

fn slow_rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let (file, rank) = (sq.file() as i32, sq.rank() as i32);
    // Four orthogonal rays
    for (df, dr) in [(1,0),(-1,0),(0,1),(0,-1)] {
        let (mut f, mut r) = (file + df, rank + dr);
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if occ & bit != 0 { break; }
            f += df; r += dr;
        }
    }
    attacks
}

fn slow_bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let (file, rank) = (sq.file() as i32, sq.rank() as i32);
    for (df, dr) in [(1,1),(1,-1),(-1,1),(-1,-1)] {
        let (mut f, mut r) = (file + df, rank + dr);
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if occ & bit != 0 { break; }
            f += df; r += dr;
        }
    }
    attacks
}

// ── Relevant occupancy masks (exclude board edges) ────────────────────────────

fn rook_mask(sq: Square) -> Bitboard {
    let (file, rank) = (sq.file() as i32, sq.rank() as i32);
    let mut mask = EMPTY;
    // Along file (exclude rank 1 and rank 8 edges)
    for r in 1..7i32 {
        if r != rank { mask |= 1u64 << (r * 8 + file); }
    }
    // Along rank (exclude file a and file h edges)
    for f in 1..7i32 {
        if f != file { mask |= 1u64 << (rank * 8 + f); }
    }
    mask
}

fn bishop_mask(sq: Square) -> Bitboard {
    let (file, rank) = (sq.file() as i32, sq.rank() as i32);
    let mut mask = EMPTY;
    for (df, dr) in [(1,1),(1,-1),(-1,1),(-1,-1)] {
        let (mut f, mut r) = (file + df, rank + dr);
        // Exclude the outer edge squares
        while f > 0 && f < 7 && r > 0 && r < 7 {
            mask |= 1u64 << (r * 8 + f);
            f += df; r += dr;
        }
    }
    mask
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pawn_attacks_e4() {
        // White pawn on e4 (sq=28) attacks d5(35) and f5(37)
        let sq = Square::new(4, 3); // e4
        let att = pawn_attacks(Color::White, sq);
        assert_eq!(popcount(att), 2);
        assert!(att & (1u64 << Square::new(3, 4).0) != 0); // d5
        assert!(att & (1u64 << Square::new(5, 4).0) != 0); // f5
    }

    #[test]
    fn knight_attacks_center() {
        let sq = Square::new(3, 3); // d4
        let att = knight_attacks(sq);
        assert_eq!(popcount(att), 8);
    }

    #[test]
    fn knight_attacks_corner() {
        let sq = Square::new(0, 0); // a1
        let att = knight_attacks(sq);
        assert_eq!(popcount(att), 2);
    }

    #[test]
    fn rook_attacks_empty_board() {
        let sq = Square::new(4, 4); // e5
        let att = rook_attacks(sq, EMPTY);
        // On empty board a rook controls the whole rank + file minus its own square
        assert_eq!(popcount(att), 14);
    }

    #[test]
    fn rook_attacks_blocked() {
        let sq = Square::new(0, 0); // a1
        let blocker_sq = Square::new(0, 2); // a3
        let occ = 1u64 << blocker_sq.0;
        let att = rook_attacks(sq, occ);
        // Should see a2, a3 (blocker included) on file; and b1..h1 on rank
        assert!(att & (1u64 << Square::new(0, 1).0) != 0); // a2
        assert!(att & (1u64 << Square::new(0, 2).0) != 0); // a3 (blocker visible)
        assert!(att & (1u64 << Square::new(0, 3).0) == 0); // a4 behind blocker
    }

    #[test]
    fn bishop_attacks_empty_board() {
        let sq = Square::new(3, 3); // d4 centre
        let att = bishop_attacks(sq, EMPTY);
        assert_eq!(popcount(att), 13);
    }

    #[test]
    fn queen_attacks_is_union() {
        let sq = Square::new(3, 3);
        let occ = EMPTY;
        assert_eq!(queen_attacks(sq, occ), rook_attacks(sq, occ) | bishop_attacks(sq, occ));
    }

    #[test]
    fn king_attacks_center() {
        let sq = Square::new(3, 3);
        assert_eq!(popcount(king_attacks(sq)), 8);
    }

    #[test]
    fn king_attacks_corner() {
        let sq = Square::new(0, 0); // a1
        assert_eq!(popcount(king_attacks(sq)), 3);
    }

    // Validate that all magic entries produce consistent attack tables
    // (no two different occupancy subsets collide to a wrong result).
    #[test]
    fn magic_tables_consistent() {
        for sq in 0u8..64 {
            let sq = Square(sq);
            // Check a few occupancy patterns
            for &occ in &[EMPTY, RANK_1, FILE_A, 0x8040201008040201u64] {
                let slow_r = slow_rook_attacks(sq, occ);
                let fast_r = rook_attacks(sq, occ);
                assert_eq!(slow_r, fast_r, "rook mismatch at sq={}", sq.0);

                let slow_b = slow_bishop_attacks(sq, occ);
                let fast_b = bishop_attacks(sq, occ);
                assert_eq!(slow_b, fast_b, "bishop mismatch at sq={}", sq.0);
            }
        }
    }
}
