# FuncPerm

A `no_std` Rust library for cycle-walking functional permutations on
\[0 . . *n*).

The [`FuncPerm`] structure provides, given *n* ≤ *m*, a pseudorandom
permutation of \[0 . . *n*) by taking a bijection on \[0 . . *m*) and
applying cycle walking to restrict the result to \[0 . . *n*), that is,
by applying the bijection repeatedly until the result is in \[0 . . *n*).

The library provides built-in bijections based on [MurmurHash3-like
mixing functions].

## Examples

```rust
use funcperm::murmur;

// Create a permutation of [0..100) with seeds (0, 0)
let perm = murmur(100, 0, 0);

// Every element maps to a unique element in [0..100)
let y = perm.get(42);
assert!(y < 100);
```

You can also supply a custom bijection:

```rust
use funcperm::FuncPerm;

let perm = FuncPerm::new(10, |x: u64| x ^ 0b1010);
assert!(perm.get(0) < 10);
```

[`FuncPerm`]: https://docs.rs/funcperm/latest/funcperm/struct.FuncPerm.html
[MurmurHash3-like mixing functions]: https://docs.rs/funcperm/latest/funcperm/murmur/
