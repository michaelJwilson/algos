use criterion::{criterion_group, criterion_main, Criterion};

fn example_function() -> i32 {
    (1..1000).sum()
}

fn benchmark_example(c: &mut Criterion) {
    c.bench_function("example_function", |b| b.iter(|| example_function()));
}

criterion_group!(benches, benchmark_example);
criterion_main!(benches);