use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use funcperm::murmur;
use std::hint::black_box;

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_murmur");
    for &n in &[1 << 20, (1 << 20) + 1, 1 << 24, (1 << 24) + 1] {
        let perm = murmur(n, 42, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &perm, |b, perm| {
            let mut x = 0u64;
            b.iter(|| {
                x = perm.get(x);
                black_box(x);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_get);
criterion_main!(benches);
