#![no_std]

const C1: u64 = 0xbf58476d1ce4e5b9;
const C2: u64 = 0x94d049bb133111eb;

/// A bijection on `[0..2^k)`.
pub trait Bijection {
    fn apply(&self, x: u64) -> u64;
}

/// SplitMix64-based bijection on `[0..2^k)`.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix {
    seed0: u64,
    seed1: u64,
    mask: u64,
    s1: u32,
    s2: u32,
    s3: u32,
}

/// Cycle-walking functional permutation on `[0..n)`.
#[derive(Clone, Copy, Debug)]
pub struct FuncPerm<B> {
    n: u64,
    bijection: B,
}

impl SplitMix {
    pub fn new(k: u32, seed0: u64, seed1: u64) -> Self {
        assert!(k <= 64, "k must be in the range [0..64]");
        let mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

        let half = (k / 2).max(1);
        let max_shift = k.saturating_sub(1).max(1);
        let s1 = half;
        let s2 = (half - 1).max(1);
        let s3 = (half + 1).min(max_shift);

        Self {
            seed0,
            seed1,
            mask,
            s1,
            s2,
            s3,
        }
    }
}

impl Bijection for SplitMix {
    fn apply(&self, x: u64) -> u64 {
        let z = x.wrapping_add(self.seed0) & self.mask;
        let z = (z ^ (z >> self.s1)).wrapping_mul(C1) & self.mask;
        let z = z.wrapping_add(self.seed1) & self.mask;
        let z = (z ^ (z >> self.s2)).wrapping_mul(C2) & self.mask;
        (z ^ (z >> self.s3)) & self.mask
    }
}

impl<B: Bijection> FuncPerm<B> {
    pub fn new(n: u64, bijection: B) -> Self {
        assert!(n > 0, "n must be positive");
        Self { n, bijection }
    }

    pub fn get(&self, x: u64) -> u64 {
        assert!(x < self.n, "x must be less than n");
        let mut y = x;
        loop {
            y = self.bijection.apply(y);
            if y < self.n {
                return y;
            }
        }
    }

    pub fn len(&self) -> u64 {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        // Always false since n > 0 is enforced by the constructor.
        false
    }
}

impl FuncPerm<SplitMix> {
    pub fn new_splitmix(n: u64, seed0: u64, seed1: u64) -> Self {
        let k = if n <= 1 {
            0
        } else {
            64 - (n - 1).leading_zeros()
        };
        Self::new(n, SplitMix::new(k, seed0, seed1))
    }
}

/// Bijection on `[0..2^k)` with 3 multiplications (vs 4 for `SplitMix`).
///
/// Uses one full SplitMix64-style round followed by a 1-multiply finisher.
#[derive(Clone, Copy, Debug)]
pub struct SimpleHash {
    seed0: u64,
    seed1: u64,
    shift: u32,
    mask: u64,
    mul: u64,
}

impl SimpleHash {
    pub fn new(k: u32, seed0: u64, seed1: u64) -> Self {
        let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
        let mul = if k == 0 {
            0
        } else {
            (0x9e3779b97f4a7c15 >> (64 - k)) | 1
        };
        Self {
            seed0,
            seed1,
            shift: (k / 2).max(1),
            mask,
            mul,
        }
    }
}

impl Bijection for SimpleHash {
    #[inline(always)]
    fn apply(&self, x: u64) -> u64 {
        let mut z = (x.wrapping_add(self.seed0)).wrapping_mul(self.mul);
        z = z.wrapping_add(self.seed1) & self.mask;
        z ^ (z >> self.shift)
    }
}

impl FuncPerm<SimpleHash> {
    pub fn new_simple_hash(n: u64, seed0: u64, seed1: u64) -> Self {
        let k = if n <= 1 {
            0
        } else {
            64 - (n - 1).leading_zeros()
        };
        Self::new(n, SimpleHash::new(k, seed0, seed1))
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use super::*;
    use alloc::collections::BTreeSet;

    #[test]
    fn test_bijectivity() -> anyhow::Result<()> {
        for &n in &[1u64, 2, 3, 7, 8, 100, 1000, 1 << 16] {
            for seed in 0..3u64 {
                let perm = FuncPerm::new_splitmix(n, seed, seed * 17);
                let mut seen = BTreeSet::new();
                for x in 0..n {
                    let y = perm.get(x);
                    assert!(y < n, "output {y} out of range [0..{n})");
                    assert!(
                        seen.insert(y),
                        "duplicate output {y} for n={n}, seed={seed}"
                    );
                }
                assert_eq!(seen.len() as u64, n);
            }
        }
        Ok(())
    }

    #[test]
    fn test_different_seeds() -> anyhow::Result<()> {
        let n = 100;
        let perm0 = FuncPerm::new_splitmix(n, 0, 1);
        let perm1 = FuncPerm::new_splitmix(n, 2, 3);
        let mut different = false;
        for x in 0..n {
            if perm0.get(x) != perm1.get(x) {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "different seeds should produce different permutations"
        );
        Ok(())
    }

    #[test]
    fn test_bijectivity_simple_hash() -> anyhow::Result<()> {
        for &n in &[1u64, 2, 3, 7, 8, 100, 1000, 1 << 16] {
            for seed in 0..3u64 {
                let perm = FuncPerm::new_simple_hash(n, seed, seed * 17);
                let mut seen = BTreeSet::new();
                for x in 0..n {
                    let y = perm.get(x);
                    assert!(y < n, "output {y} out of range [0..{n})");
                    assert!(
                        seen.insert(y),
                        "duplicate output {y} for n={n}, seed={seed}"
                    );
                }
                assert_eq!(seen.len() as u64, n);
            }
        }
        Ok(())
    }

    #[test]
    fn test_different_seeds_simple_hash() -> anyhow::Result<()> {
        let n = 100;
        let perm0 = FuncPerm::new_simple_hash(n, 0, 1);
        let perm1 = FuncPerm::new_simple_hash(n, 2, 3);
        let mut different = false;
        for x in 0..n {
            if perm0.get(x) != perm1.get(x) {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "different seeds should produce different permutations"
        );
        Ok(())
    }

    #[test]
    fn test_edge_cases() -> anyhow::Result<()> {
        // n = 1: only possible permutation
        let perm = FuncPerm::new_splitmix(1, 42, 43);
        assert_eq!(perm.get(0), 0);
        assert_eq!(perm.len(), 1);

        // n = 2: must be a bijection for all seeds
        for seed in 0..10u64 {
            let perm = FuncPerm::new_splitmix(2, seed, seed);
            let y0 = perm.get(0);
            let y1 = perm.get(1);
            assert!(y0 < 2);
            assert!(y1 < 2);
            assert_ne!(y0, y1);
        }

        // Powers of 2: no cycle-walking needed
        for &k in &[4u64, 8, 16, 64, 256] {
            let perm = FuncPerm::new_splitmix(k, 12345, 12345);
            let mut seen = BTreeSet::new();
            for x in 0..k {
                let y = perm.get(x);
                assert!(y < k);
                assert!(seen.insert(y));
            }
        }

        Ok(())
    }
}
