/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![no_std]
#![doc = include_str!("../README.md")]

const C1: u64 = 0xbf58476d1ce4e5b9;
const C2: u64 = 0x94d049bb133111eb;

/// Cycle-walking functional permutation on `[0..n)`.
#[derive(Clone, Copy)]
pub struct FuncPerm<B> {
    n: u64,
    bijection: B,
}

impl<B> core::fmt::Debug for FuncPerm<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FuncPerm")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl<B> FuncPerm<B> {
    pub const fn new(n: u64, bijection: B) -> Self {
        assert!(n > 0, "n must be positive");
        Self { n, bijection }
    }

    pub const fn len(&self) -> u64 {
        self.n
    }

    pub const fn is_empty(&self) -> bool {
        // Always false since n > 0 is enforced by the constructor.
        false
    }
}

impl<B: Fn(u64) -> u64> FuncPerm<B> {
    pub fn get(&self, x: u64) -> u64 {
        assert!(x < self.n, "x must be less than n");
        let mut y = x;
        loop {
            y = (self.bijection)(y);
            if y < self.n {
                return y;
            }
        }
    }
}

/// SplitMix64-based cycle-walking permutation on `[0..n)`.
///
/// Two-round mixing with four multiplications per application.
pub fn splitmix(n: u64, seed0: u64, seed1: u64) -> FuncPerm<impl Fn(u64) -> u64 + Copy> {
    let k = if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros()
    };
    let mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

    let half = (k / 2).max(1);
    let max_shift = k.saturating_sub(1).max(1);
    let s1 = half;
    let s2 = (half - 1).max(1);
    let s3 = (half + 1).min(max_shift);

    FuncPerm::new(n, move |x: u64| {
        let z = x.wrapping_add(seed0) & mask;
        let z = (z ^ (z >> s1)).wrapping_mul(C1) & mask;
        let z = z.wrapping_add(seed1) & mask;
        let z = (z ^ (z >> s2)).wrapping_mul(C2) & mask;
        (z ^ (z >> s3)) & mask
    })
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
                let perm = splitmix(n, seed, seed * 17);
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
        let perm0 = splitmix(n, 0, 1);
        let perm1 = splitmix(n, 2, 3);
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
        let perm = splitmix(1, 42, 43);
        assert_eq!(perm.get(0), 0);
        assert_eq!(perm.len(), 1);

        // n = 2: must be a bijection for all seeds
        for seed in 0..10u64 {
            let perm = splitmix(2, seed, seed);
            let y0 = perm.get(0);
            let y1 = perm.get(1);
            assert!(y0 < 2);
            assert!(y1 < 2);
            assert_ne!(y0, y1);
        }

        // Powers of 2: no cycle-walking needed
        for &k in &[4u64, 8, 16, 64, 256] {
            let perm = splitmix(k, 12345, 12345);
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
