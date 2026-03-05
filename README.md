# FuncPerm

A `no_std` Rust library for cycle-walking functional permutations on
[0 . . *n*).

The [`FuncPerm`] structure provides, given _n_ ≤ _m_, a pseudorandom permutation
of [0 . . *n*) by taking a bijection on [0 . . *m*) and applying [cycle walking]
to restrict the result to [0 . . *n*).

## Bijections

The library provides built-in bijections based on MurmurHash3-like mixing
functions.

- **`splitmix(n, seed0, seed1)`** — Two-round SplitMix64-derived mixing
  function with shift parameters adapted to _k_. Four multiplications
  per application. Good statistical quality.[0 . . 2*ᵏ*) (where _k_ = ⌈lg _n_⌉)
  Different seed pairs yield different
  permutations with good statistical uniformity (see the tests), but they cannot
  guarantee uniform random sampling form the permutation space like [Feistel
  networks] do. In exchange, computing the mapping of an element takes just a few
  nanoseconds.

You can plug in any custom bijection on [0 . . 2*ᵏ*) by passing a
closure to [`FuncPerm::new`].

## Properties

- **`no_std`**: no heap allocation, no standard library dependency.
- **Zero runtime dependencies**.
- **Constant space**: `FuncPerm` is `Clone + Copy` and stores only a
  few `u64`s regardless of `n`.
- **Statistically uniform**: for random seeds, each position maps to
  each output value with near-uniform probability (verified by G-tests
  in the test suite).
- **Deterministic**: same `(n, seed0, seed1)` always produces the same
  permutation.

[cycle walking]: https://en.wikipedia.org/wiki/Permutation#Cycle_walking
[Feistel networks]: https://crates.io/crates/feistel-permutation-rs
[`FuncPerm`]: https://docs.rs/funcperm/latest/funcperm/struct.FuncPerm.html
