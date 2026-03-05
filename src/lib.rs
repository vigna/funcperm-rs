/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![no_std]
#![doc = include_str!("../README.md")]

pub mod murmur;
pub use murmur::murmur;

/// Cycle-walking functional permutation on [0 . . *n*), given a bijection
/// on [0 . . *m*), where *m* ≥ *n*.
///
/// The bijection must map [0 . . *m*) to itself. The [`get`](Self::get)
/// method applies the bijection repeatedly until the result falls in
/// [0 . . *n*), producing a permutation of [0 . . *n*).
///
/// `PartialEq`, `Eq`, and `Hash` cannot be derived because the bijection
/// is typically a closure.
///
/// # Examples
///
/// Using a custom bijection (XOR with a constant on 4-bit values):
///
/// ```
/// # use funcperm::FuncPerm;
///
/// let perm = FuncPerm::new(10, |x: u64| x ^ 0b1010);
/// assert!(perm.get(0) < 10);
/// ```
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
    /// Creates a new cycle-walking functional permutation on [0 . . *n*) using
    /// the given bijection, which must have domain [0 . . *m*),
    /// where *m* ≥ *n*.
    ///
    /// Note that providing a function that does not satisfy the bijection
    /// requirement may lead to non-termination of the [`get`](Self::get)
    /// method.
    #[must_use]
    pub const fn new(n: u64, bijection: B) -> Self {
        Self { n, bijection }
    }

    /// Returns the size of the permutation domain.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.n
    }

    /// Returns `true` if the permutation domain is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl<B: Fn(u64) -> u64> FuncPerm<B> {
    /// Returns the image of `x` under the permutation.
    ///
    /// # Panics
    ///
    /// Panics if `x` ≥ `n`.
    #[inline]
    #[must_use]
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
