// use std::collections::HashMap;
use rustc_hash::FxHashMap as HashMap;


use rand::Rng;
use std::hash::Hash;

/// NB counter counts (u32) the number of times each value
///    of (generic) type T has been seen.  Data-class like.
pub struct Counter<T> {
    values: HashMap<T, u32>,
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
    #[inline]
    fn count(&mut self, value: T) {
        *self.values.entry(value).or_default() += 1;
    }

    /// NB Return the number of times the given value has been seen.
    #[inline]
    fn times_seen(&self, value: T) -> u32 {
        // self.values.get(&value).copied().unwrap_or_default()

        if let Some(&count) = self.values.get(&value) {
            count
        } else {
            0
        }
    }

    fn times_seen_all(&self) -> u32 {
        self.values.values().sum()
    }
}

pub fn get_counter_fixture(num_samples: usize) -> Counter<u32> {
    let mut rng = rand::rng();
    let mut ctr = Counter::new();

    
    for _ in 0..num_samples {
        let draw: u32 = rng.random_range(0..num_samples) as u32;

        ctr.count(draw);
    }
    
    assert_eq!(num_samples as u32, ctr.times_seen_all());

    ctr
}

#[cfg(test)]
mod tests {
    // cargo test counter -- --nocapture
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
