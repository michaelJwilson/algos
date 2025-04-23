use algos::collatz::collatz;
use algos::counter::get_counter_fixture;
use algos::dijkstra::{dijkstra, get_adjacencies_fixture_large};
use algos::ford_fulkerson::{edmonds_karp, get_adjacencies_fixture, get_large_graph_fixture};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::DMatrix;
use ndarray::Array2;
use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;
use std::cell::RefCell;

fn fibonacci_slow(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci_slow(n - 1) + fibonacci_slow(n - 2),
    }
}

fn fibonacci_fast(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;

    match n {
        0 => b,
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
    c.bench_function("fib(20)", |b| b.iter(|| fibonacci_fast(black_box(20))));

    c.bench_function("collatz(100)", |b| b.iter(|| collatz(black_box(100))));

    c.bench_function("dijkstra", |b| {
        let num_fail = RefCell::new(0);

        b.iter(|| {
            let adjs = get_adjacencies_fixture_large(100);
            let result = dijkstra(adjs, 0, 100);

            match result {
                Some(cost) => assert!(cost > 0),
                None => *num_fail.borrow_mut() += 1,
            }
        });

        assert_eq!(num_fail, 0.into());
    });

    c.bench_function("ford_fulkerson", |b| {
        b.iter(|| {
            let (source, sink, _, graph) = get_adjacencies_fixture();
            edmonds_karp(graph, source, sink);
        })
    });

    c.bench_function("petgraph_ford_fulkerson", |b| {
        b.iter(|| {
            // NB visium is 5K spots.
            let (source, sink, _, g) = get_large_graph_fixture(5_000);
            let _ = petgraph_ford_fulkerson(&g, source, sink);
        })
    });

    c.bench_function("counter", |b| {
        b.iter(|| {
            let _ = get_counter_fixture(10_000);
        })
    });

    c.bench_function("linear_algebra_nalgebra", |b| {
        b.iter(|| {
            let a = DMatrix::from_element(10_000, 10_000, 1.0);
            let b = DMatrix::<f64>::zeros(10_000, 10_000);

            let _ = a.dot(&b);
        })
    });

    c.bench_function("linear_algebra_ndarray_fast", |b| {
        b.iter(|| {
            let a = Array2::<f64>::zeros((10_000, 10_000));
            let b = Array2::<f64>::ones((10_000, 10_000));

            let _ = a.dot(&b);
        })
    });

    c.bench_function("linear_algebra_ndarray_slow", |b| {
        b.iter(|| {
            let a = Array2::<f64>::zeros((10_000, 10_000));
            let b = Array2::<f64>::ones((10_000, 10_000));

            let mut result = Array2::<f64>::zeros((10_000, 10_000));

            for i in 0..10_000 {
                for j in 0..10_000 {
                    let mut sum = 0.0;

                    for k in 0..10_000 {
                        sum += a[[i, k]] * b[[k, j]];
                    }

                    result[[i, j]] = sum;
                }
            }
        })
    });
}

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fibonacci");

    for i in [20u64, 21u64].iter() {
        group.bench_with_input(BenchmarkId::new("Recursive", i), i, |b, i| {
            b.iter(|| fibonacci_slow(*i))
        });

        group.bench_with_input(BenchmarkId::new("Iterative", i), i, |b, i| {
            b.iter(|| fibonacci_fast(*i))
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
// criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
