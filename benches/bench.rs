use algos::ford_fulkerson::get_large_graph_fixture;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;

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
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));

    c.bench_function("petgraph_ford_fulkerson", |b| {
        b.iter(|| {
            // NB visium is 5K spots.
            let (source, sink, _, g) = get_large_graph_fixture(5_000);
            let (max_flow, _) = petgraph_ford_fulkerson(&g, source, sink);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
