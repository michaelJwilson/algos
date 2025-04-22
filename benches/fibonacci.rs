use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;

    match n {
        0 => b,

        // NB all else.
        _ => {
            for _ in 0..n {
                let c = a + b;

                a = b;
                b = c;
            }

            b
        }
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // NB black_box prevents optimization that scrub code which
    //    doesn't produce a used result.
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
