use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use funcperm::murmur;
use std::hint::black_box;

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_murmur(throughput)");
    for &n in &[1 << 20, (1 << 20) + 1, 1 << 24, (1 << 24) + 1] {
        let perm = murmur(n, 42, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &perm, |b, perm| {
            let mut x = 0u64;
            b.iter(|| {
                let y = perm.get(x);
                x += 1;
                if x == n {
                    x = 0;
                }
                black_box(y);
            })
        });
    }
    group.finish();
}

fn bench_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_murmur(latency)");
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

criterion_group!(benches, bench_throughput, bench_latency);
criterion_main!(benches);
