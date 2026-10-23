//! Minimal bit-exact port of what `np.random.default_rng(seed).integers(0, high, dtype=int64)`
//! does for a non-negative integer seed and a range wider than 32 bits:
//! SeedSequence (pool size 4) -> PCG64 (XSL-RR 128/64) -> Lemire bounded draws.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// NumPy's SeedSequence pool for an integer entropy and no spawn key.
fn seed_sequence_pool(seed: u64) -> [u32; POOL_SIZE] {
    let mut entropy = Vec::new();
    let mut s = seed;
    if s == 0 { entropy.push(0); }
    while s > 0 { entropy.push(s as u32); s >>= 32; }

    let mut pool = [0u32; POOL_SIZE];
    let mut hc = INIT_A;
    for i in 0..POOL_SIZE {
        pool[i] = hashmix(*entropy.get(i).unwrap_or(&0), &mut hc);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], &mut hc));
            }
        }
    }
    for &e in entropy.iter().skip(POOL_SIZE) {
        for i_dst in 0..POOL_SIZE {
            pool[i_dst] = mix(pool[i_dst], hashmix(e, &mut hc));
        }
    }
    pool
}

/// SeedSequence.generate_state(n_words, uint64).
fn generate_state_u64(pool: &[u32; POOL_SIZE], n_words: usize) -> Vec<u64> {
    let mut hc = INIT_B;
    let mut w32 = Vec::with_capacity(n_words * 2);
    for i in 0..n_words * 2 {
        let mut v = pool[i % POOL_SIZE] ^ hc;
        hc = hc.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hc);
        v ^= v >> XSHIFT;
        w32.push(v);
    }
    w32.chunks(2).map(|c| (c[0] as u64) | ((c[1] as u64) << 32)).collect()
}

const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

pub struct Generator {
    state: u128,
    inc: u128,
}

impl Generator {
    pub fn from_seed(seed: u64) -> Self {
        let v = generate_state_u64(&seed_sequence_pool(seed), 4);
        let initstate = ((v[0] as u128) << 64) | v[1] as u128;
        let initseq = ((v[2] as u128) << 64) | v[3] as u128;
        let mut g = Generator { state: 0, inc: (initseq << 1) | 1 };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng.wrapping_add(1);
        let mut m = self.next_u64() as u128 * rng_excl as u128;
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = self.next_u64() as u128 * rng_excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `integers(low, high)` (high exclusive) for int64. Only the >32-bit range
    /// path is implemented; Engram's range is ~4.6e13, far above 2^32.
    pub fn integers_i64(&mut self, low: i64, high: i64) -> i64 {
        let rng = (high - 1 - low) as u64;
        assert!(rng > u32::MAX as u64 && rng != u64::MAX, "only the 64-bit Lemire path is ported");
        low + self.bounded_lemire_u64(rng) as i64
    }
}
