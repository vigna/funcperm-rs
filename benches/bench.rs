use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use funcperm::FuncPerm;

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_splitmix");
    for &n in &[1 << 20, (1 << 20) + 1, 1 << 24, (1 << 24) + 1] {
        let perm = FuncPerm::new_splitmix(n, 42, 42);
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

fn bench_get_simple_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_simple_hash");
    for &n in &[1 << 20, (1 << 20) + 1, 1 << 24, (1 << 24) + 1] {
        let perm = FuncPerm::new_simple_hash(n, 42, 42);
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

criterion_group!(benches, bench_get, bench_get_simple_hash);
criterion_main!(benches);
