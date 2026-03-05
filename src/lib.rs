/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![no_std]
#![doc = include_str!("../README.md")]

pub mod murmur;
pub use murmur::murmur;

/// Cycle-walking functional permutation on [0 . . *n*), given a bijection on [0 . . *m*), *m* ≥ *n*.
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
        Self { n, bijection }
    }

    pub const fn len(&self) -> u64 {
        self.n
    }

    pub const fn is_empty(&self) -> bool {
        self.n == 0
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
