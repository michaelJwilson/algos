use criterion::{black_box, criterion_group, criterion_main, Criterion};
use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;
use algos::ford_fulkerson::get_large_graph_fixture;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("petgraph_ford_fulkerson", |b| {
        b.iter(|| {
            let (source, sink, _, g) = get_large_graph_fixture(10_000);
            let (max_flow, _) =
                petgraph_ford_fulkerson(&g, source, sink);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
