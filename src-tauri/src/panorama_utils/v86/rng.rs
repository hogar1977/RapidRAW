const MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341;
const STATE0: u128 = 0x1aa1b5345996452d09585eb7a69561e3;
const INC0: u128 = 0x418ddadb3af71a82588133bc447873a9;

pub struct NumpyRng {
    state: u128,
    inc: u128,
    spare32: Option<u32>,
}

impl NumpyRng {
    pub fn seed0() -> Self {
        Self {
            state: STATE0,
            inc: INC0,
            spare32: None,
        }
    }

    fn next64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
        let xs = ((self.state >> 64) as u64) ^ (self.state as u64);
        let rot = ((self.state >> 122) as u32) & 63;
        xs.rotate_right(rot)
    }

    fn next32(&mut self) -> u32 {
        if let Some(v) = self.spare32.take() {
            return v;
        }
        let r = self.next64();
        self.spare32 = Some((r >> 32) as u32);
        r as u32
    }

    fn bounded_inclusive(&mut self, hi: u64) -> u64 {
        if hi == 0 {
            return 0;
        }
        if hi <= 0xFFFF_FFFF {
            let rng = hi as u32;
            if rng == 0xFFFF_FFFF {
                return self.next32() as u64;
            }
            let rng_excl = u64::from(rng) + 1;
            loop {
                let r = u64::from(self.next32());
                let m = r * rng_excl;
                let leftover = m & 0xFFFF_FFFF;
                if leftover < rng_excl {
                    let threshold = (0xFFFF_FFFFu64.wrapping_sub(u64::from(rng))) % rng_excl;
                    if leftover < threshold {
                        continue;
                    }
                }
                return m >> 32;
            }
        }
        let rng_excl = hi + 1;
        loop {
            let x = self.next64();
            let m = (x as u128).wrapping_mul(rng_excl as u128);
            let leftover = m as u64;
            if leftover < rng_excl {
                let threshold = u64::MAX.wrapping_sub(hi) % rng_excl;
                if leftover < threshold {
                    continue;
                }
            }
            return (m >> 64) as u64;
        }
    }

    pub fn choice(&mut self, pop: usize, size: usize) -> Vec<usize> {
        if size == 0 || pop == 0 || size > pop {
            return Vec::new();
        }
        let pop_i = pop as u64;
        let size_i = size as u64;
        let cutoff = 50u64;
        if pop_i > 10_000 && size_i > pop_i / cutoff {
            let mut idx: Vec<i64> = (0..pop as i64).collect();
            let first = (pop_i - size_i).max(1);
            for i in (first..pop_i).rev() {
                let j = self.bounded_inclusive(i) as usize;
                idx.swap(j, i as usize);
            }
            return idx[(pop - size)..].iter().map(|v| *v as usize).collect();
        }
        let mut idx = vec![0i64; size];
        let set_size = (1.2 * size as f64) as u64;
        let mask = gen_mask(set_size);
        let mut hash = vec![u64::MAX; (mask as usize) + 1];
        for j in (pop_i - size_i)..pop_i {
            let val = self.bounded_inclusive(j);
            let mut loc = (val & mask) as usize;
            while hash[loc] != u64::MAX && hash[loc] != val {
                loc = ((loc as u64 + 1) & mask) as usize;
            }
            let slot = (j - (pop_i - size_i)) as usize;
            if hash[loc] == u64::MAX {
                hash[loc] = val;
                idx[slot] = val as i64;
            } else {
                let mut loc = (j & mask) as usize;
                while hash[loc] != u64::MAX {
                    loc = ((loc as u64 + 1) & mask) as usize;
                }
                hash[loc] = j;
                idx[slot] = j as i64;
            }
        }
        if size > 1 {
            for i in (1..size).rev() {
                let j = self.bounded_inclusive(i as u64) as usize;
                idx.swap(j, i);
            }
        }
        idx.into_iter().map(|v| v as usize).collect()
    }
}

fn gen_mask(max_val: u64) -> u64 {
    let mut mask = max_val;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    mask
}
