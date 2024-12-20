use rand::{Rng, distributions::Distribution};

pub struct Categorical {
    cmf: Vec<f64>,
}

impl Categorical {
    pub fn from_pmf(pmf: &[f64]) -> Self {
        let cmf = pmf
            .iter()
            .scan(0.0, |state, x| {
                *state += x;
                Some(*state)
            })
            .collect();
        Self { cmf }
    }
}

impl Distribution<usize> for Categorical {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let uniform_sample = rng.gen::<f64>();
        let (i, _x) = self
            .cmf
            .iter()
            .enumerate()
            .find(|(_i, &x)| uniform_sample < x)
            .unwrap();
        i
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    use approx::assert_abs_diff_eq;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_categorical_from_pmf() {
        let pmf = vec![0.1, 0.2, 0.3, 0.4];
        let wc = Categorical::from_pmf(&pmf);
        let expected_cmf = vec![0.1, 0.3, 0.6, 1.0];

	for (r, e) in wc.cmf.iter().zip(expected_cmf.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = TOLERANCE);
        }
    }

    #[test]
    fn test_categorical_sample() {
        let pmf = vec![0.1, 0.2, 0.3, 0.4];
        let wc = Categorical::from_pmf(&pmf);
        let mut rng = thread_rng();

        for _ in 0..100 {
            let index = wc.sample(&mut rng);
	    
            assert!(index < pmf.len());
        }
    }
}