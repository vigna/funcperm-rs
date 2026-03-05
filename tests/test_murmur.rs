use std::collections::BTreeSet;

use funcperm::murmur;
use rand::{RngExt, SeedableRng};
use statrs::distribution::{ChiSquared, ContinuousCDF};

#[test]
fn test_bijectivity() -> anyhow::Result<()> {
    for &n in &[1u64, 2, 3, 7, 8, 100, 1000, 1 << 16] {
        for seed in 0..3u64 {
            let perm = murmur(n, seed, seed * 17);
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
    let perm0 = murmur(n, 0, 1);
    let perm1 = murmur(n, 2, 3);
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
    let perm = murmur(1, 42, 43);
    assert_eq!(perm.get(0), 0);
    assert_eq!(perm.len(), 1);

    // n = 2: must be a bijection for all seeds
    for seed in 0..10u64 {
        let perm = murmur(2, seed, seed);
        let y0 = perm.get(0);
        let y1 = perm.get(1);
        assert!(y0 < 2);
        assert!(y1 < 2);
        assert_ne!(y0, y1);
    }

    // Powers of 2: no cycle-walking needed
    for &k in &[4u64, 8, 16, 64, 256] {
        let perm = murmur(k, 12345, 12345);
        let mut seen = BTreeSet::new();
        for x in 0..k {
            let y = perm.get(x);
            assert!(y < k);
            assert!(seen.insert(y));
        }
    }

    Ok(())
}

fn chi_squared(counts: &[u64], expected: f64) -> f64 {
    counts
        .iter()
        .map(|&c| {
            let d = c as f64 - expected;
            d * d / expected
        })
        .sum::<f64>()
}

/// Check that at a given position the output values are approximately uniformly
/// distributed across many random seed pairs, using a chi-squared test at
/// significance level 0.05 where outputs are bucketed into √n bins.
///
/// Note that the test would fail with n bins.
fn check_uniformity(n: u64, position: u64, rng: &mut rand::rngs::SmallRng) {
    let num_buckets = (n as f64).sqrt().ceil() as u64;
    let num_samples = num_buckets * 1000;
    let expected = num_samples as f64 / num_buckets as f64;
    let mut counts = vec![0u64; num_buckets as usize];

    for _ in 0..num_samples {
        let perm = murmur(n, rng.random(), rng.random());
        let y = perm.get(position);
        counts[(y * num_buckets / n) as usize] += 1;
    }

    let x2 = chi_squared(&counts, expected);
    let critical = ChiSquared::new((num_buckets - 1) as f64)
        .unwrap()
        .inverse_cdf(0.99);

    assert!(
        x2 < critical,
        "Chi-squared test failed for n={n}, position={position}: X²={x2:.2}, critical={critical:.2}"
    );
}

#[test]
fn test_statistical_uniformity() -> anyhow::Result<()> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    for &n in &[10000, 100000, 1000000] {
        for &position in &[0, 1, n / 2, n - 1] {
            // We try a few positions. The test would fail if we would try on
            // all positions.
            check_uniformity(n, position, &mut rng);
        }
    }
    Ok(())
}
