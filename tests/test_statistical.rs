use funcperm::FuncPerm;
use rand::{RngExt, SeedableRng};

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

/// Inverse standard normal (quantile) via rational approximation from
/// Peter Acklam. Accurate to ~1.15e-9 over the full range.
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Wilson-Hilferty approximation for the chi-squared critical value at the
/// given significance level (right-tail probability).
fn chi2_critical(df: f64, alpha: f64) -> f64 {
    let z = normal_quantile(1.0 - alpha);
    let term = 2.0 / (9.0 * df);
    let x = 1.0 - term + z * term.sqrt();
    df * x * x * x
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
    let critical = chi2_critical((n - 1) as f64, 0.01);

    assert!(
        g < critical,
        "G-test failed for n={n}, position={position}: G={g:.2}, critical={critical:.2}"
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
