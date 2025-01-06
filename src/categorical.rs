use rand::{Rng, distributions::Distribution};
use statrs::distribution::{Categorical, Discrete};

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    use approx::assert_abs_diff_eq;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_categorical_new() {
        let pmf: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
	let mut cat = Categorical::new(&pmf);
	assert!(cat.is_ok());
    }

    #[test]
    fn test_categorical_pmf() {
        let pmf: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
	let mut cat = Categorical::new(&pmf).unwrap();

	for ii in (0..pmf.len()) {
	    let result = cat.pmf(ii.try_into().unwrap());

	    asserteq!(result, pmf[ii]);
	}
    }

    #[test]
    fn test_categorical_sample() {
        let mut rng = thread_rng();
	let pmf: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let cat = Categorical::new(&pmf).unwrap();

	for _ in 0..100 {
            let index: u64 = cat.sample(&mut rng);	    
            assert!(index < pmf.len() as u64);
        }
    }
}