// use std::collections::HashMap;
use rustc_hash::FxHashMap as HashMap;

use std::hash::Hash;
use rand::Rng;

/// NB counter counts (u64) the number of times each value
///    of (generic) type T has been seen.  Data-class like.
pub struct Counter<T> {
    values: HashMap<T, u64>,
}

/// NB (generic) T must have traits for equal and hashable.
impl<T: Eq + Hash> Counter<T> {
    /// NB Create a new Counter.
    fn new() -> Self {
        Counter {
            values: HashMap::default(),
        }
    }

    /// NB Count an occurrence of the given value.
    fn count(&mut self, value: T) {
        *self.values.entry(value).or_default() += 1;
    }

    /// NB Return the number of times the given value has been seen.
    fn times_seen(&self, value: T) -> u64 {
        self.values.get(&value).copied().unwrap_or_default()
    }
}

pub fn get_counter_fixture(num_samples: usize) -> Counter<usize> {
    let mut rng = rand::rng();
    let mut ctr = Counter::new();

    for jj in 0..num_samples {
        let draw: usize = rng.random_range(0..num_samples);

        ctr.count(draw);
    }

    ctr
}

#[cfg(test)]
mod tests {
    // cargo test hash_map -- --nocapture
    use super::*;

    #[test]
    fn test_counter() {
        let ctr = get_counter_fixture(10_000);

        for i in 10..20 {
            println!("saw {} values equal to {}", ctr.times_seen(i), i);
        }

        let mut strctr = Counter::new();

        strctr.count("apple");
        strctr.count("orange");
        strctr.count("apple");

        println!("got {} apples", strctr.times_seen("apple"));
    }
}
