# FuncPerm

A `no_std` Rust library for cycle-walking functional permutations on
[0 . . *n*).

The [`FuncPerm`] structure provides, given _n_ ≤ _m_, a pseudorandom permutation
of [0 . . *n*) by taking a bijection on [0 . . *m*) and applying cycle walking
to restrict the result to [0 . . *n*), that is, by applying the bijection
repeatedly until the result is in [0 . . *n*).

The library provides built-in bijections based on [MurmurHash3-like mixing
functions].

[`FuncPerm`]: https://docs.rs/funcperm/latest/funcperm/struct.FuncPerm.html
[MurmurHash3-like mixing functions]: https://docs.rs/funcperm/latest/funcperm/murmur/
