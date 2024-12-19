use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

const rand_seed: u64 = 42;

fn create_lattice(size: usize, num_clones: i32) -> Array2<i32> {
   let mut rng = ChaCha8Rng::seed_from_u64(rand_seed);
   let lattice: Array2<i32> = Array2::from_shape_fn((size, size), |_| rng.gen_range(0..num_clones));

   lattice
}

fn get_clone_sizes (lattice: Array2<i32>) -> HashMap<i32, usize> {
   let mut counts: HashMap<i32, usize> = HashMap::new();

   // Count the occurrences of each entry in the lattice
   for &value in lattice.iter() {
       *counts.entry(value).or_insert(0) += 1;
   }

   counts
}

pub fn test_potts() {
   let size = 5_000;
   let num_clones = 5;

   let start = Instant::now();

   let lattice = create_lattice(size, num_clones);
   let clone_sizes = get_clone_sizes(lattice);

   let duration = start.elapsed();

   println!("Found {clone_sizes:?} in {duration:?}.")
}
