use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

const RAND_SEED: u64 = 42;

fn create_lattice(size: usize, num_clones: i32) -> Array2<i32> {
   let mut rng = ChaCha8Rng::seed_from_u64(RAND_SEED);
   let lattice: Array2<i32> = Array2::from_shape_fn((size, size), |_| rng.gen_range(0..num_clones));

   lattice
}

fn get_clone_sizes (lattice: &Array2<i32>) -> HashMap<i32, usize> {
   let mut counts: HashMap<i32, usize> = HashMap::new();

   // Count the occurrences of each entry in the lattice
   for &value in lattice.iter() {
       *counts.entry(value).or_insert(0) += 1;
   }

   counts
}

fn neighbor_defect(value: i32, neighbor: i32) -> i32{
   if neighbor != value {
      return 1;
   }
   
   else{
      return 0;
   }
}

fn get_lattice_cost(lattice: &Array2<i32>) -> f64 {
   let beta: f64 = 1.0;
   let mut defects: i32 = 0;

   for ((i, j), &value) in lattice.indexed_iter() {
        if i as i32 - 1 < 0 {
	   defects += neighbor_defect(value, lattice[(i - 1, j)]);
	}

        if i as i32 + 1 < lattice.nrows() as i32 {
	   defects += neighbor_defect(value, lattice[(i + 1, j)]);
        }

	if j as i32 - 1 < 0 {   
	   defects += neighbor_defect(value, lattice[(i + 1, j)]);
	}

        if j as i32 + 1 < lattice.ncols() as i32 {
	   defects += neighbor_defect(value, lattice[(i, j + 1)]);
        }
    }

   beta * (defects as f64)
}

pub fn test_potts() {
   let size = 3;
   let num_clones = 2;

   let start = Instant::now();

   let lattice = create_lattice(size, num_clones);

   println!("{:?}", lattice);

   let clone_sizes = get_clone_sizes(&lattice);
   let cost = get_lattice_cost(&lattice);

   let duration = start.elapsed();

   println!("Found {clone_sizes:?} with cost {cost:?} in {duration:?}.")
}
