A repository of algorithm examples in Rust (and python).


Tests:
    RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_merge_sorted_two -- --nocapture
    
LeetCode Top 150:
    - merge_sorted.rs

uv usage:
    - source .venv/bin/activate

log usage:
    - RUST_LOG=info ./target/release/algos 

See also:
    - Comprehensive Rust by android @ Google
      https://google.github.io/comprehensive-rust/

    - Hidden Markov Models in Rust
      https://github.com/paulkernfeld/hmmm/blob/master/Cargo.toml

    - The Rust performance book
      https://nnethercote.github.io/perf-book/build-configuration.html

    - Criterion
      https://bheisler.github.io/criterion.rs/book/user_guide/comparing_functions.html

    - pprof-rs
      ...

    - nalgebra (NB column-major order!)
      ...