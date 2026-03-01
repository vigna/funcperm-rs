//! Optimizer for SplitMix mixer constants.
//!
//! Uses simulated annealing to find optimal per-k parameters `(s1, s2, s3,
//! C1, C2)` that minimize avalanche bias, following the methodology of
//! Stafford and Kagstrom.
//!
//! The mixer being optimized has this structure (on k-bit values):
//! ```text
//! z = (x + seed0) & mask
//! z = (z ^ (z >> s1)) * C1 & mask
//! z = (z + seed1) & mask
//! z = (z ^ (z >> s2)) * C2 & mask
//! z = (z ^ (z >> s3)) & mask
//! ```
//!
//! Build with: `cargo run --bin optimize --features optimize --release -- [OPTIONS]`

use clap::Parser;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// CLI arguments for the optimizer.
#[derive(Parser, Debug)]
#[command(name = "optimize")]
struct Args {
    /// Optimize a single bit size (overrides --min-k / --max-k)
    #[arg(short = 'k', long)]
    bit_size: Option<u32>,

    /// Minimum bit size [default: 1]
    #[arg(long, default_value_t = 1)]
    min_k: u32,

    /// Maximum bit size [default: 64]
    #[arg(long, default_value_t = 64)]
    max_k: u32,

    /// Number of random seed pairs per evaluation (in addition to 3 fixed
    /// degenerate pairs)
    #[arg(long, default_value_t = 50)]
    num_seeds: usize,

    /// Number of sequential test inputs per seed pair. Capped internally at
    /// 2^k for small k, since inputs beyond the domain repeat.
    #[arg(long, default_value_t = 10000)]
    num_inputs: u64,

    /// Simulated annealing iterations per k
    #[arg(long, default_value_t = 100000)]
    iterations: u64,

    /// Initial SA temperature (higher = more exploration early on)
    #[arg(long, default_value_t = 1.0)]
    initial_temp: f64,

    /// Geometric cooling rate: T_{i+1} = T_i * cooling_rate.
    /// Should be close to 1; with 100K iterations, 0.9999 cools from
    /// T₀ to ~T₀/22000.
    #[arg(long, default_value_t = 0.9999)]
    cooling_rate: f64,

    /// Start from random constants instead of current library values
    #[arg(long)]
    random_start: bool,

    /// RNG seed for reproducibility (random if not specified)
    #[arg(long)]
    sa_seed: Option<u64>,
}

/// The two multiplicative constants from the library's SplitMix mixer.
const LIB_C1: u64 = 0xbf58476d1ce4e5b9;
const LIB_C2: u64 = 0x94d049bb133111eb;

/// The five tunable parameters of the mixer for a given bit size k.
#[derive(Clone, Copy, Debug)]
struct Params {
    s1: u32, // first xor-shift amount
    s2: u32, // second xor-shift amount
    s3: u32, // final xor-shift amount
    c1: u64, // first multiplicative constant (must be odd)
    c2: u64, // second multiplicative constant (must be odd)
}

/// Returns the k-bit mask: 2^k - 1 (all-ones for k = 64).
fn mask_for_k(k: u32) -> u64 {
    if k == 64 {
        u64::MAX
    } else {
        (1u64 << k) - 1
    }
}

/// Apply the mixer with the given parameters.
///
/// This must match the structure of `SplitMix::apply` in `src/lib.rs`,
/// with parameterized constants and shifts. Since
/// `(x * C) & mask == (x * (C & mask)) & mask` for power-of-2 moduli,
/// we can use pre-truncated constants.
#[inline(always)]
fn apply(x: u64, seed0: u64, seed1: u64, mask: u64, p: &Params) -> u64 {
    let z = x.wrapping_add(seed0) & mask;
    let z = (z ^ (z >> p.s1)).wrapping_mul(p.c1) & mask;
    let z = z.wrapping_add(seed1) & mask;
    let z = (z ^ (z >> p.s2)).wrapping_mul(p.c2) & mask;
    (z ^ (z >> p.s3)) & mask
}

/// Returns the range of shifts to try: [k/2 - 3, k/2 + 3] clamped to [1, k-1].
fn shift_range(k: u32) -> (u32, u32) {
    let max_shift = k.saturating_sub(1).max(1);
    let half = k as i32 / 2;
    let lo = (half - 3).max(1) as u32;
    let hi = (half + 3).min(max_shift as i32) as u32;
    (lo, hi)
}

/// Compute the SAC score for a single (params, seed-pair) combination.
///
/// Builds a k×k bias matrix where entry `[b][j]` = `|flip_rate - 0.5|`.
/// Returns mean(bias) + max(bias). Lower is better (0 = perfect avalanche).
fn evaluate_one(k_usize: usize, mask: u64, p: &Params, seed0: u64, seed1: u64, num_inputs: u64) -> f64 {
    let mut flip_count = vec![vec![0u64; k_usize]; k_usize];

    for x in 0..num_inputs {
        // Compute f(x) once, then test each single-bit perturbation.
        let fx = apply(x, seed0, seed1, mask, p);
        for b in 0..k_usize {
            let x_flipped = x ^ (1u64 << b);
            let fx_flipped = apply(x_flipped, seed0, seed1, mask, p);
            let diff = fx ^ fx_flipped;
            for j in 0..k_usize {
                flip_count[b][j] += (diff >> j) & 1;
            }
        }
    }

    let mut sum_bias = 0.0;
    let mut max_bias: f64 = 0.0;
    let num_entries = (k_usize * k_usize) as f64;

    for b in 0..k_usize {
        for j in 0..k_usize {
            let bias = (flip_count[b][j] as f64 / num_inputs as f64 - 0.5).abs();
            sum_bias += bias;
            max_bias = max_bias.max(bias);
        }
    }

    sum_bias / num_entries + max_bias
}

/// Evaluate constants (c1, c2) by exhaustively trying all shift triples in
/// [k/2 - 3, k/2 + 3] and returning the best (lowest) score along with the
/// optimal shifts.
///
/// All (shift-triple, seed-pair) combinations are evaluated in parallel
/// (up to 343 × num_seeds work units), maximizing core utilization.
/// For each shift triple, the score is the worst case (max) across seed
/// pairs. The best (minimum) score across shift triples is returned.
fn evaluate_best_shifts(
    k: u32,
    c1: u64,
    c2: u64,
    seed_pairs: &[(u64, u64)],
    num_inputs: u64,
) -> (f64, Params) {
    let (lo, hi) = shift_range(k);
    let mask = mask_for_k(k);
    let k_usize = k as usize;

    // Build all shift triples.
    let mut shift_triples: Vec<(u32, u32, u32)> = Vec::new();
    for s1 in lo..=hi {
        for s2 in lo..=hi {
            for s3 in lo..=hi {
                shift_triples.push((s1, s2, s3));
            }
        }
    }

    // Build all (shift_index, seed_index) work units and evaluate in parallel.
    let num_shifts = shift_triples.len();
    let num_seeds = seed_pairs.len();
    let total_units = num_shifts * num_seeds;

    // Each work unit computes the SAC score for one (shift-triple, seed-pair).
    let scores: Vec<f64> = (0..total_units)
        .into_par_iter()
        .map(|idx| {
            let shift_idx = idx / num_seeds;
            let seed_idx = idx % num_seeds;
            let (s1, s2, s3) = shift_triples[shift_idx];
            let p = Params { s1, s2, s3, c1, c2 };
            let (seed0, seed1) = seed_pairs[seed_idx];
            evaluate_one(k_usize, mask, &p, seed0, seed1, num_inputs)
        })
        .collect();

    // For each shift triple, take the max score across seed pairs (worst case).
    // Then pick the shift triple with the minimum worst-case score.
    let mut best_score = f64::INFINITY;
    let mut best_params = Params { s1: lo, s2: lo, s3: lo, c1, c2 };

    for (shift_idx, &(s1, s2, s3)) in shift_triples.iter().enumerate() {
        let base = shift_idx * num_seeds;
        let worst = scores[base..base + num_seeds]
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        if worst < best_score {
            best_score = worst;
            best_params = Params { s1, s2, s3, c1, c2 };
        }
    }

    (best_score, best_params)
}

/// Generate a random (c1, c2) pair for a given k.
fn random_constants(k: u32, rng: &mut SmallRng) -> (u64, u64) {
    let mask = mask_for_k(k);
    // | 1 ensures constants are odd (required for bijective multiplication
    // modulo a power of 2).
    let c1 = (rng.random::<u64>() & mask) | 1;
    let c2 = (rng.random::<u64>() & mask) | 1;
    (c1, c2)
}

/// Generate a neighbor (c1, c2) by flipping 1–4 random bits (never bit 0,
/// to preserve oddness) across c1 and c2. Shifts are not mutated — they
/// are determined exhaustively at evaluation time.
fn neighbor_constants(c1: u64, c2: u64, k: u32, rng: &mut SmallRng) -> (u64, u64) {
    if k <= 2 {
        return (c1, c2);
    }

    let mask = mask_for_k(k);
    let mut new_c1 = c1;
    let mut new_c2 = c2;

    let num_flips = rng.random_range(1..=4u32);
    for _ in 0..num_flips {
        let bit_pos = rng.random_range(1..k);
        if rng.random::<bool>() {
            new_c1 ^= 1u64 << bit_pos;
        } else {
            new_c2 ^= 1u64 << bit_pos;
        }
    }

    (new_c1 & mask, new_c2 & mask)
}

/// Run simulated annealing for a single bit size k.
///
/// SA searches over (C1, C2) only. At each evaluation, all shift triples
/// in [k/2 - 3, k/2 + 3] are tried exhaustively (at most 7^3 = 343
/// combinations), and the best shifts for the given constants are selected.
///
/// Returns `(best_params, best_score, initial_score)`.
fn optimize_k(k: u32, args: &Args, rng: &mut SmallRng) -> (Params, f64, f64) {
    // Seed pairs for evaluation: 3 fixed degenerate pairs that stress-test the
    // mixer (trivial seed additions), plus user-specified random pairs.
    let mut seed_pairs: Vec<(u64, u64)> = vec![(0, 0), (0, 1), (1, 0)];
    for _ in 0..args.num_seeds {
        seed_pairs.push((rng.random(), rng.random()));
    }

    // Cap inputs at domain size.
    let domain_size = if k == 64 { u64::MAX } else { 1u64 << k };
    let num_inputs = args.num_inputs.min(domain_size);

    // Choose starting constants: library defaults or random.
    let (mut cur_c1, mut cur_c2) = if args.random_start {
        random_constants(k, rng)
    } else {
        let mask = mask_for_k(k);
        ((LIB_C1 & mask) | 1, (LIB_C2 & mask) | 1)
    };

    // Evaluate starting constants with exhaustive shift search.
    let (initial_score, initial_params) =
        evaluate_best_shifts(k, cur_c1, cur_c2, &seed_pairs, num_inputs);

    eprintln!(
        "k={:2}: start s=({},{},{}) C1=0x{:x}, C2=0x{:x}  score={:.6}{}",
        k, initial_params.s1, initial_params.s2, initial_params.s3,
        cur_c1, cur_c2, initial_score,
        if args.random_start { " (random)" } else { " (library)" }
    );

    let mut current_score = initial_score;
    let mut best = initial_params;
    let mut best_score = initial_score;
    let mut accepted = 0u64;
    let mut improved = 0u64;

    let mut temp = args.initial_temp;

    for i in 0..args.iterations {
        // Mutate only the constants; shifts will be found exhaustively.
        let (cand_c1, cand_c2) = neighbor_constants(cur_c1, cur_c2, k, rng);
        let (cand_score, cand_params) =
            evaluate_best_shifts(k, cand_c1, cand_c2, &seed_pairs, num_inputs);

        // Standard Metropolis acceptance: always accept improvements,
        // accept worsening moves with probability exp(-delta/T).
        let delta = cand_score - current_score;
        if delta < 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
            cur_c1 = cand_c1;
            cur_c2 = cand_c2;
            current_score = cand_score;
            accepted += 1;
            if current_score < best_score {
                best = cand_params;
                best_score = current_score;
                improved += 1;
            }
        }

        // Geometric cooling.
        temp *= args.cooling_rate;

        // Progress logging every 1000 iterations and at the end.
        if (i + 1) % 1000 == 0 || i + 1 == args.iterations {
            eprintln!(
                "  iter {:6}/{}: best={:.6}  current={:.6}  T={:.2e}  accepted={}  improved={}",
                i + 1, args.iterations, best_score, current_score, temp, accepted, improved
            );
        }
    }

    eprintln!(
        "k={:2}: done. accepted={}/{}({:.1}%)  improved={}  score {:.6} -> {:.6}",
        k, accepted, args.iterations,
        100.0 * accepted as f64 / args.iterations.max(1) as f64,
        improved, initial_score, best_score
    );

    (best, best_score, initial_score)
}

fn main() {
    let args = Args::parse();

    // Initialize RNG: deterministic from --sa-seed, or from OS entropy.
    let mut rng = match args.sa_seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => {
            let mut buf = [0u8; 8];
            getrandom::fill(&mut buf).expect("failed to get random seed");
            SmallRng::seed_from_u64(u64::from_le_bytes(buf))
        }
    };

    // -k overrides --min-k / --max-k.
    let (min_k, max_k) = match args.bit_size {
        Some(k) => (k, k),
        None => (args.min_k, args.max_k),
    };

    assert!(min_k >= 1, "min-k must be at least 1");
    assert!(max_k <= 64, "max-k must be at most 64");
    assert!(min_k <= max_k, "min-k must be <= max-k");

    let mut results: Vec<(u32, Params, f64, f64)> = Vec::new();

    for k in min_k..=max_k {
        let (best, best_score, initial_score) = optimize_k(k, &args, &mut rng);
        println!(
            "k={:2}: s=({},{},{}) C1=0x{:x}, C2=0x{:x}  score={:.6}  (was {:.6})",
            k, best.s1, best.s2, best.s3, best.c1, best.c2, best_score, initial_score
        );
        results.push((k, best, best_score, initial_score));
    }

    // Emit a Rust array containing only the optimized range.
    println!();
    println!("const PARAMS: [(u32, u32, u32, u64, u64); {}] = [", max_k - min_k + 1);
    for &(k, p, _, _) in &results {
        println!("    // k={}", k);
        println!(
            "    ({}, {}, {}, 0x{:x}, 0x{:x}),",
            p.s1, p.s2, p.s3, p.c1, p.c2
        );
    }
    println!("];");
}
