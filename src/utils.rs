/// xorshift64 pseudo-random number generator.
pub struct Xorshift64(u64);

impl Xorshift64 {
    /// Seed must be non-zero; a 1-bit is ORed in to guarantee this.
    pub fn new(seed: u64) -> Self {
        Xorshift64(seed | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    #[inline]
    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}
