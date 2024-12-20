use nalgebra::{DMatrix};
use rand::{Rng, distributions::Distribution, thread_rng};
use crate::categorical::Categorical;

fn get_transfer_matrix(K: i32) -> DMatrix<f64> {
   let t = 1.0e-5;
   let K = K.clone() as usize;

   let eye: DMatrix<f64> = DMatrix::identity(K,K);
   let ones: DMatrix<f64> = DMatrix::from_element(K, K, 1.0);

   (t * &eye) + (&ones - &eye) * (1. - t) / (K as f64 - 1.)
}

#[cfg(test)]
mod tests {
    use super::*;    
    use approx::assert_abs_diff_eq;
    use std::collections::HashMap;
    use nalgebra::{DMatrix, QR, SymmetricEigen};

    const TOLERANCE: f64 = 1e-6;

    fn assert_matrices_close(matrix1: &DMatrix<f64>, matrix2: &DMatrix<f64>, epsilon: f64) {
       assert_eq!(matrix1.shape(), matrix2.shape(), "Matrices must have the same shape");

       for (a, b) in matrix1.iter().zip(matrix2.iter()) {
       	   assert_abs_diff_eq!(a, b, epsilon = epsilon);
       }
    }

    #[test]
    fn test_transfer_matrix() {
       let T = get_transfer_matrix(5);
    }

    #[test]
    fn test_transfer_hop() {
       let PI = DMatrix::from_element(5, 1, 1.);
       let T = get_transfer_matrix(5);

       let exp = DMatrix::from_element(5, 1, 1.);
       let result = T * PI;

       assert_matrices_close(&result, &exp, TOLERANCE);
    }

    #[test]
    fn test_transfer_hop_sample() {
       let mut rng = thread_rng();

       let PI = DMatrix::from_element(5, 1, 1.);
       let T = get_transfer_matrix(5);

       let state_prob = (T * PI);

       let cat = Categorical::from_pmf(state_prob.as_slice());
       let mut counts: HashMap<usize, usize> = HashMap::new();


       for _ in (0..1_000) {
           let sample = cat.sample(&mut rng);
	   *counts.entry(sample).or_insert(0) += 1;
       }

       println!("{counts:#?}");
    }

    #[test]
    fn test_transfer_matrix_power() {
       let T = get_transfer_matrix(5);
  
       let exp = DMatrix::from_element(5, 5, 1. / 5 as f64);
       let result = T.pow(100).unwrap();

       assert_matrices_close(&result, &exp, TOLERANCE);
    }

    #[test]
    fn test_transfer_eigenvalue() {
       let T = get_transfer_matrix(5);
       let se = SymmetricEigen::new(T);

       let evs = se.eigenvalues;

       println!("{evs:?}");
    }
}