use statrs::distribution::{Categorical, Discrete};

#[cfg(test)]
mod tests {
    // NB place all "parent" definitions in test scope.
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::rng;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_categorical_new() {
        // NB four categories with given probabilities.
        let pmf: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let mut cat = Categorical::new(&pmf);

        assert!(cat.is_ok());
    }

    #[test]
    fn test_categorical_pmf() {
        let pmf: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let mut cat = Categorical::new(&pmf).unwrap();

        for ii in 0..pmf.len() {
            let result = cat.pmf(ii.try_into().unwrap());

            assert_eq!(result, pmf[ii]);
        }
    }

    #[test]
    fn test_categorical_sample() {
        // NB tests sampling from given categorical.
        let mut rng = rng();
        let pmf: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let cat = Categorical::new(&pmf).unwrap();

        for _ in 0..100 {
            let index: u64 = cat.sample(&mut rng);
            assert!(index < pmf.len() as u64);
        }
    }
}
