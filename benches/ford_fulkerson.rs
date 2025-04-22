use algos::ford_fulkerson::get_large_graph_fixture;
use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("petgraph ford fulkerson large", |b| {
        b.iter(|| {
            // NB prevent compiler optimizations?
            let (source, sink, _, g) = get_large_graph_fixture(black_box(200));
            let (max_flow, _) = petgraph_ford_fulkerson(black_box(&g), black_box(source), black_box(sink));

            println!("{:?}", max_flow);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
