#![feature(test)] // Enable the unstable test feature
extern crate test;

use test::Bencher;

#[bench]
fn example_benchmark(bench: &mut Bencher) {
    bench.iter(|| {
        // Code to benchmark
        let sum: i32 = (1..1000).sum();
        sum
    });
}