use std::collections::HashMap;
use std::hash::Hash;

/// NB counter counts (u64) the number of times each value
///    of (generic) type T has been seen.  Data-class like.
struct Counter<T> {
    values: HashMap<T, u64>,
}

/// NB (generic) T must have traits for equal and hashable.
impl<T: Eq + Hash> Counter<T> {
    /// NB Create a new Counter.
    fn new() -> Self {
        Counter {
            values: HashMap::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let mut ctr = Counter::new();

        ctr.count(13);
        ctr.count(14);
        ctr.count(16);
        ctr.count(14);
        ctr.count(14);
        ctr.count(11);

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
