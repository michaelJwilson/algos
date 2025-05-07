use algos::collatz::collatz;
use algos::counter::get_counter_fixture;
use algos::dijkstra::{dijkstra, get_adjacencies_fixture_large};
use algos::felsenstein::{compute_likelihood, get_felsenstein_fixture};
use algos::max_flow::{edmonds_karp, get_adj_matrix_fixture, get_large_graph_fixture};
use algos::leapfrog::{leapfrog, acceleration, get_leapfrog_fixture};
use algos::streaming::{get_nbinom_fixture, get_betabinom_fixture, basic_nbinom_logpmf, stream_nbinom_logpmf, basic_betabinom_logpmf, stream_betabinom_logpmf};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::DMatrix;
use ndarray::Array2;
use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;

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

    c.bench_function("stream_nb", |b| {
        let (k, r, p, q) = get_nbinom_fixture(1_000);

        b.iter(|| {
            let _ = stream_nbinom_logpmf(&k, &r, &p, &q);
        });
    });
    
    c.bench_function("stream_bb", |b| {
        let (num_trials, num_states) = (1_000, 5);
        let (k, n, alpha, beta, q) = get_betabinom_fixture(num_trials, num_states);

        b.iter(|| {
            let _ = stream_betabinom_logpmf(&k, &n, &alpha, &beta, &q);
        });
    });

    c.bench_function("leapfrog", |b| b.iter(|| {
        let (initial_position, initial_velocity, time_step, num_steps) = get_leapfrog_fixture();

        let _ = leapfrog(
            acceleration,
            initial_position,
            initial_velocity,
            time_step,
            num_steps,
        );                                 
    }));

    c.bench_function("dijkstra", |b| {
        let adjs = get_adjacencies_fixture_large(100);

        b.iter(|| {
            let _ = dijkstra(&adjs, 0, 100);
        });
    });

    c.bench_function("felsenstein", |b| {
        let (root, transition_matrix, branch_lengths) = get_felsenstein_fixture();

        b.iter(|| {
            let _ = compute_likelihood(&root, &transition_matrix, &branch_lengths);
        })
    });

    c.bench_function("edmonds_karp", |b| {
        let (source, sink, _, graph) = get_adj_matrix_fixture();

        b.iter(|| {
            edmonds_karp(&graph, source, sink);
        })
    });

    c.bench_function("petgraph_ford_fulkerson", |b| {
        // NB visium is 5K spots.
        let (nodes, g) = get_large_graph_fixture::<u32, u32>(5_000, 1.);

        let source = nodes[0];
        let sink = nodes[nodes.len() - 1];

        b.iter(|| {
            // NB visium is 5K spots.
            // let (nodes, g) = get_large_graph_fixture::<u32, u32>(5_000, 1.);

            // let source = nodes[0];
            // let sink = nodes[nodes.len() - 1];

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

criterion_group!(benches, criterion_benchmark, bench_fibonacci);
// criterion_group!(benches, bench_fibonacci);

criterion_main!(benches);
