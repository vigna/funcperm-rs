/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/// The constants in this file were computed by running the following code:
///
/// ```text
/// cargo run --bin optimize --features optimize --release -- --iterations 10000 --num-seeds 100 --num-inputs 10000 --sa-seed 0 --random-start
/// ```
///
/// Results:
///
/// ```text
/// k= 1: s=(1,1,1) C1=0x1, C2=0x1  score=1.000000  (was 1.000000)
/// k= 2: s=(1,1,1) C1=0x1, C2=0x1  score=1.000000  (was 1.000000)
/// k= 3: s=(1,1,1) C1=0x1, C2=0x3  score=0.777778  (was 0.777778)
/// k= 4: s=(3,3,1) C1=0xb, C2=0x5  score=0.593750  (was 0.671875)
/// k= 5: s=(4,3,3) C1=0x5, C2=0xb  score=0.370000  (was 0.650000)
/// k= 6: s=(3,4,2) C1=0x23, C2=0x27  score=0.324653  (was 0.368056)
/// k= 7: s=(3,3,4) C1=0x63, C2=0x4b  score=0.255740  (was 0.279337)
/// k= 8: s=(3,3,4) C1=0xbb, C2=0xcb  score=0.197266  (was 0.281982)
/// k= 9: s=(4,4,5) C1=0x149, C2=0x1db  score=0.142554  (was 0.303819)
/// k=10: s=(6,4,4) C1=0xe9, C2=0x2db  score=0.111719  (was 0.190078)
/// k=11: s=(6,5,5) C1=0x7b3, C2=0x735  score=0.080256  (was 0.097931)
/// k=12: s=(7,6,6) C1=0x865, C2=0x59b  score=0.061822  (was 0.216824)
/// k=13: s=(7,6,6) C1=0x1f65, C2=0x1a2d  score=0.046161  (was 0.099208)
/// k=14: s=(8,6,7) C1=0x975, C2=0xca5  score=0.039208  (was 0.224963)
/// k=15: s=(6,7,7) C1=0x7127, C2=0x4cb5  score=0.035835  (was 0.081579)
/// k=16: s=(9,7,7) C1=0xae4b, C2=0x68ad  score=0.033803  (was 0.055343)
/// k=17: s=(8,8,8) C1=0x5ed3, C2=0xe9b3  score=0.032693  (was 0.047592)
/// k=18: s=(8,8,8) C1=0x12393, C2=0x31753  score=0.031744  (was 0.061282)
/// k=19: s=(9,9,10) C1=0x3397, C2=0x656cb  score=0.032003  (was 0.060876)
/// k=20: s=(9,8,10) C1=0x63067, C2=0xfe593  score=0.030870  (was 0.046864)
/// k=21: s=(8,10,11) C1=0x3b265, C2=0xd3a95  score=0.030926  (was 0.047231)
/// k=22: s=(11,10,10) C1=0x3e39eb, C2=0x185acb  score=0.030806  (was 0.051713)
/// k=23: s=(9,10,11) C1=0x609e73, C2=0x1d12d3  score=0.030381  (was 0.064035)
/// k=24: s=(10,10,13) C1=0xc66343, C2=0xfa8527  score=0.030260  (was 0.042744)
/// k=25: s=(14,11,12) C1=0x10334f3, C2=0x15cb59b  score=0.030158  (was 0.066483)
/// k=26: s=(11,11,12) C1=0x1da25a7, C2=0x2e46d35  score=0.030169  (was 0.042414)
/// k=27: s=(12,12,12) C1=0x4fbf34f, C2=0x94a35d  score=0.029657  (was 0.032399)
/// k=28: s=(14,13,13) C1=0x6a71b4f, C2=0xeba5159  score=0.030239  (was 0.067522)
/// k=29: s=(16,16,13) C1=0x1806c64d, C2=0x6f94dad  score=0.030219  (was 0.050545)
/// k=30: s=(14,14,14) C1=0x216b0ae5, C2=0xad688ad  score=0.030303  (was 0.044555)
/// k=31: s=(17,14,15) C1=0x29f428c7, C2=0x45cd465b  score=0.030105  (was 0.036247)
/// k=32: s=(16,15,18) C1=0x3891bd71, C2=0xaf0cad7  score=0.029862  (was 0.036664)
/// k=33: s=(18,15,14) C1=0x19733a8d5, C2=0x1f4b3c753  score=0.029964  (was 0.196848)
/// k=34: s=(19,15,18) C1=0x3382c32c3, C2=0xdbcd0d6b  score=0.029942  (was 0.056396)
/// k=35: s=(19,15,17) C1=0x469974da7, C2=0x706e85cad  score=0.029996  (was 0.036367)
/// k=36: s=(19,17,16) C1=0x94a04e97, C2=0x1185476a5  score=0.030122  (was 0.032163)
/// k=37: s=(19,17,19) C1=0x136db2bc7, C2=0x9e2d93529  score=0.030225  (was 0.033651)
/// k=38: s=(20,17,17) C1=0x1006636c67, C2=0x2a4e1b6863  score=0.030372  (was 0.032577)
/// k=39: s=(21,20,21) C1=0x621cf6ac7b, C2=0x1b7614e055  score=0.030248  (was 0.030985)
/// k=40: s=(19,18,18) C1=0x1947c429fb, C2=0x6f1c515183  score=0.029827  (was 0.031815)
/// k=41: s=(18,20,19) C1=0xbcc5842315, C2=0x126c8eba035  score=0.029781  (was 0.048463)
/// k=42: s=(19,21,21) C1=0x346649d8ec1, C2=0x1052c149395  score=0.030387  (was 0.231805)
/// k=43: s=(21,22,22) C1=0x4950fa3b5f7, C2=0x2eeb98465af  score=0.030500  (was 0.047957)
/// k=44: s=(21,21,23) C1=0xaa9613b09b9, C2=0x758aedd5ab3  score=0.030328  (was 0.031862)
/// k=45: s=(22,24,21) C1=0x1483a15c6e11, C2=0x6ecf1ecda35  score=0.030391  (was 0.033000)
/// k=46: s=(22,22,24) C1=0xc48be7edf69, C2=0x7b6b4ecbf97  score=0.030102  (was 0.034636)
/// k=47: s=(21,21,23) C1=0x4e052e1ac9af, C2=0x4c50d66cf21b  score=0.030517  (was 0.032496)
/// k=48: s=(22,24,26) C1=0x858ac45c2475, C2=0x43742b3911cf  score=0.030397  (was 0.034217)
/// k=49: s=(24,24,26) C1=0xaa73502dc6bb, C2=0x1e4838dae7395  score=0.030151  (was 0.033417)
/// k=50: s=(25,26,26) C1=0x1e5286469fe4f, C2=0x16a2e5258f96d  score=0.030307  (was 0.035172)
/// k=51: s=(24,26,25) C1=0x2bc734cd7ab9b, C2=0x1f18e4527d3a5  score=0.030386  (was 0.031998)
/// k=52: s=(27,24,28) C1=0xf70149834faab, C2=0x96badd4df43af  score=0.030502  (was 0.032237)
/// k=53: s=(26,26,25) C1=0x1199a5257379c7, C2=0x1b3afc0bb44ca9  score=0.030461  (was 0.033041)
/// k=54: s=(27,25,27) C1=0x2975e7fff3bccf, C2=0x64e643b734a67  score=0.030009  (was 0.032197)
/// k=55: s=(29,26,25) C1=0x751e8155a8c8bd, C2=0x5b75ff5b3ea9f1  score=0.030551  (was 0.031912)
/// k=56: s=(28,26,28) C1=0xfd951983105da7, C2=0xf50b49dabc44e7  score=0.030243  (was 0.032426)
/// k=57: s=(28,27,29) C1=0x11cd51acff1e14d, C2=0x18fc3dc8f749cd5  score=0.030731  (was 0.032900)
/// k=58: s=(29,29,30) C1=0x386facde8c583a3, C2=0x20cb64884504041  score=0.030582  (was 0.032982)
/// k=59: s=(27,31,31) C1=0x5eed2a0920955d, C2=0x2a595cfc3765bcd  score=0.030782  (was 0.032632)
/// k=60: s=(28,30,30) C1=0xf0e21dde8d85b95, C2=0x7b41b00ca64787b  score=0.030545  (was 0.033232)
/// k=61: s=(31,30,30) C1=0x1534ba91990e6c8b, C2=0x3a905de9c29a9c9  score=0.030741  (was 0.032450)
/// k=62: s=(31,33,29) C1=0x141bd542362e317b, C2=0x184318cf0dc65219  score=0.030543  (was 0.033056)
/// k=63: s=(31,29,30) C1=0x69efc60d13898c7f, C2=0x4d551a644d29ea7d  score=0.030052  (was 0.032173)
/// k=64: s=(32,32,31) C1=0x6cde5df80197351, C2=0x8ba1aec6676989af  score=0.030725  (was 0.032228)
/// ```
const PARAMS: [(u32, u32, u32, u64, u64); 64] = [
    // k=1
    (1, 1, 1, 0x1, 0x1),
    // k=2
    (1, 1, 1, 0x1, 0x1),
    // k=3
    (1, 1, 1, 0x1, 0x3),
    // k=4
    (3, 3, 1, 0xb, 0x5),
    // k=5
    (4, 3, 3, 0x5, 0xb),
    // k=6
    (3, 4, 2, 0x23, 0x27),
    // k=7
    (3, 3, 4, 0x63, 0x4b),
    // k=8
    (3, 3, 4, 0xbb, 0xcb),
    // k=9
    (4, 4, 5, 0x149, 0x1db),
    // k=10
    (6, 4, 4, 0xe9, 0x2db),
    // k=11
    (6, 5, 5, 0x7b3, 0x735),
    // k=12
    (7, 6, 6, 0x865, 0x59b),
    // k=13
    (7, 6, 6, 0x1f65, 0x1a2d),
    // k=14
    (8, 6, 7, 0x975, 0xca5),
    // k=15
    (6, 7, 7, 0x7127, 0x4cb5),
    // k=16
    (9, 7, 7, 0xae4b, 0x68ad),
    // k=17
    (8, 8, 8, 0x5ed3, 0xe9b3),
    // k=18
    (8, 8, 8, 0x12393, 0x31753),
    // k=19
    (9, 9, 10, 0x3397, 0x656cb),
    // k=20
    (9, 8, 10, 0x63067, 0xfe593),
    // k=21
    (8, 10, 11, 0x3b265, 0xd3a95),
    // k=22
    (11, 10, 10, 0x3e39eb, 0x185acb),
    // k=23
    (9, 10, 11, 0x609e73, 0x1d12d3),
    // k=24
    (10, 10, 13, 0xc66343, 0xfa8527),
    // k=25
    (14, 11, 12, 0x10334f3, 0x15cb59b),
    // k=26
    (11, 11, 12, 0x1da25a7, 0x2e46d35),
    // k=27
    (12, 12, 12, 0x4fbf34f, 0x94a35d),
    // k=28
    (14, 13, 13, 0x6a71b4f, 0xeba5159),
    // k=29
    (16, 16, 13, 0x1806c64d, 0x6f94dad),
    // k=30
    (14, 14, 14, 0x216b0ae5, 0xad688ad),
    // k=31
    (17, 14, 15, 0x29f428c7, 0x45cd465b),
    // k=32
    (16, 15, 18, 0x3891bd71, 0xaf0cad7),
    // k=33
    (18, 15, 14, 0x19733a8d5, 0x1f4b3c753),
    // k=34
    (19, 15, 18, 0x3382c32c3, 0xdbcd0d6b),
    // k=35
    (19, 15, 17, 0x469974da7, 0x706e85cad),
    // k=36
    (19, 17, 16, 0x94a04e97, 0x1185476a5),
    // k=37
    (19, 17, 19, 0x136db2bc7, 0x9e2d93529),
    // k=38
    (20, 17, 17, 0x1006636c67, 0x2a4e1b6863),
    // k=39
    (21, 20, 21, 0x621cf6ac7b, 0x1b7614e055),
    // k=40
    (19, 18, 18, 0x1947c429fb, 0x6f1c515183),
    // k=41
    (18, 20, 19, 0xbcc5842315, 0x126c8eba035),
    // k=42
    (19, 21, 21, 0x346649d8ec1, 0x1052c149395),
    // k=43
    (21, 22, 22, 0x4950fa3b5f7, 0x2eeb98465af),
    // k=44
    (21, 21, 23, 0xaa9613b09b9, 0x758aedd5ab3),
    // k=45
    (22, 24, 21, 0x1483a15c6e11, 0x6ecf1ecda35),
    // k=46
    (22, 22, 24, 0xc48be7edf69, 0x7b6b4ecbf97),
    // k=47
    (21, 21, 23, 0x4e052e1ac9af, 0x4c50d66cf21b),
    // k=48
    (22, 24, 26, 0x858ac45c2475, 0x43742b3911cf),
    // k=49
    (24, 24, 26, 0xaa73502dc6bb, 0x1e4838dae7395),
    // k=50
    (25, 26, 26, 0x1e5286469fe4f, 0x16a2e5258f96d),
    // k=51
    (24, 26, 25, 0x2bc734cd7ab9b, 0x1f18e4527d3a5),
    // k=52
    (27, 24, 28, 0xf70149834faab, 0x96badd4df43af),
    // k=53
    (26, 26, 25, 0x1199a5257379c7, 0x1b3afc0bb44ca9),
    // k=54
    (27, 25, 27, 0x2975e7fff3bccf, 0x64e643b734a67),
    // k=55
    (29, 26, 25, 0x751e8155a8c8bd, 0x5b75ff5b3ea9f1),
    // k=56
    (28, 26, 28, 0xfd951983105da7, 0xf50b49dabc44e7),
    // k=57
    (28, 27, 29, 0x11cd51acff1e14d, 0x18fc3dc8f749cd5),
    // k=58
    (29, 29, 30, 0x386facde8c583a3, 0x20cb64884504041),
    // k=59
    (27, 31, 31, 0x5eed2a0920955d, 0x2a595cfc3765bcd),
    // k=60
    (28, 30, 30, 0xf0e21dde8d85b95, 0x7b41b00ca64787b),
    // k=61
    (31, 30, 30, 0x1534ba91990e6c8b, 0x3a905de9c29a9c9),
    // k=62
    (31, 33, 29, 0x141bd542362e317b, 0x184318cf0dc65219),
    // k=63
    (31, 29, 30, 0x69efc60d13898c7f, 0x4d551a644d29ea7d),
    // k=64
    (32, 32, 31, 0x6cde5df80197351, 0x8ba1aec6676989af),
];

/// [MurmurHash3](https://en.wikipedia.org/wiki/MurmurHash)-based functional
/// permutations.
///
/// This function returns a functional permutation on [0 . . *n*) based on a
/// two-round mixing function derived from the
/// [MurmurHash3](https://en.wikipedia.org/wiki/MurmurHash) finalizer. The
/// modified finalizer provides a bijection on [0 . . 2*ᵏ*), where _k_ = ⌈lg
/// _n_⌉. The function takes two seeds that are injected just before the first
/// and second xorshift. Each pair of seeds is expected to provide a different
/// permutation. Note that only the lower _k_ bits of the seeds are used: his
/// means, in particular, that in general you can obtain at most *n*² different
/// permutations.
///
/// Different seed pairs are expected to yield different permutations with
/// reasonable statistical uniformity, but they cannot guarantee uniform random
/// sampling from the permutation space like [Feistel
/// networks](https://crates.io/crates/feistel-permutation-rs) using
/// cryptographic functions do. In the tests, we check that the *n* functions
/// with both equal seeds in [0 . . *n*) map √*n* equispaced elements to each of
/// the *n* outputs uniformly using a χ² test at significance level 0.01 (the
/// same test on individual positions would fail).
///
/// In exchange, computing the mapping of an element takes just a dozen
/// nanoseconds (see the benchmarks).
///
/// # Examples
///
/// ```
/// # use funcperm::murmur;
/// let perm = murmur(10, 0, 0);
/// let mut seen = [false; 10];
/// for i in 0..10 {
///     let x = perm.get(i);
///     assert!(x < 10);
///     assert!(!seen[x as usize]);
///     seen[x as usize] = true;
/// }
/// ```
#[must_use]
pub fn murmur(
    n: u64,
    mut seed0: u64,
    mut seed1: u64,
) -> crate::FuncPerm<impl Fn(u64) -> u64 + Copy> {
    let k = if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros()
    };

    let mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

    seed0 &= mask;
    seed1 &= mask;

    let (s1, s2, s3, c1, c2) = if k == 0 {
        (1, 1, 1, 1u64, 1u64)
    } else {
        let p = PARAMS[k as usize - 1];
        (p.0, p.1, p.2, p.3, p.4)
    };

    crate::FuncPerm::new(n, move |x: u64| {
        let z = x.wrapping_add(seed0) & mask;
        let z = (z ^ (z >> s1)).wrapping_mul(c1) & mask;
        let z = z.wrapping_add(seed1) & mask;
        let z = (z ^ (z >> s2)).wrapping_mul(c2) & mask;
        z ^ (z >> s3)
    })
}
