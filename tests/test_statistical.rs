use funcperm::FuncPerm;
use rand::{RngExt, SeedableRng};
use statrs::distribution::{ChiSquared, ContinuousCDF};

fn g_test(counts: &[u64], expected: f64) -> f64 {
    2.0 * counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let o = c as f64;
            o * (o / expected).ln()
        })
        .sum::<f64>()
}

/// Check that at a given position the output values are uniformly distributed
/// across many random seed pairs, using a G-test at significance level 0.05.
fn check_uniformity(n: u64, position: u64, rng: &mut rand::rngs::SmallRng) {
    let num_seeds = n * 1000;
    let mut counts = vec![0u64; n as usize];
    let expected = num_seeds as f64 / n as f64;

    for _ in 0..num_seeds {
        let perm = FuncPerm::new_splitmix(n, rng.random(), rng.random());
        let y = perm.get(position);
        counts[y as usize] += 1;
    }

    let g = g_test(&counts, expected);
    let critical = ChiSquared::new((n - 1) as f64).unwrap().inverse_cdf(0.99);

    assert!(
        g < critical,
        "G-test failed for n={n}, position={position}: G={g:.2}, critical={critical:.2}"
    );
}

fn check_uniformity_simple_hash(n: u64, position: u64, rng: &mut rand::rngs::SmallRng) {
    let num_seeds = n * 1000;
    let mut counts = vec![0u64; n as usize];
    let expected = num_seeds as f64 / n as f64;

    for _ in 0..num_seeds {
        let perm = FuncPerm::new_simple_hash(n, rng.random(), rng.random());
        let y = perm.get(position);
        counts[y as usize] += 1;
    }

    let g = g_test(&counts, expected);
    let critical = ChiSquared::new((n - 1) as f64).unwrap().inverse_cdf(0.99);

    assert!(
        g < critical,
        "G-test failed for SimpleHash n={n}, position={position}: G={g:.2}, critical={critical:.2}"
    );
}

#[test]
fn test_statistical_uniformity() -> anyhow::Result<()> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    for &n in &[100, 1000, 10000] {
        for &position in &[0, 1, n / 2, n - 1] {
            check_uniformity(n, position, &mut rng);
        }
    }
    Ok(())
}

#[test]
fn test_statistical_uniformity_simple_hash() -> anyhow::Result<()> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    for &n in &[100, 1000, 10000] {
        for &position in &[0, 1, n / 2, n - 1] {
            check_uniformity_simple_hash(n, position, &mut rng);
        }
    }
    Ok(())
}
