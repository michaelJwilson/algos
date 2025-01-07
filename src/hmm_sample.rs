use rand::{Rng, SeedableRng, rngs::StdRng, distributions::Distribution};
use crate::categorical::Categorical;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HMMSample {
    pub x: usize,
    pub y: usize,
}

// NB An iterator that returns random samples from an HMM
pub struct HMMSampleIter<'a, R: Rng + ?Sized + 'a> {
    a_categoricals: Vec<Categorical>,
    b_categoricals: Vec<Categorical>,
    c_categorical: Categorical,
    rng: &'a mut R,
    current_state: Option<usize>,
}

// NB define a lifetime a'
impl<'a, R: Rng + ?Sized> Iterator for HMMSampleIter<'a, R> {
    type Item = HMMSample;

    fn next(&mut self) -> Option<Self::Item> {
        let state = if let Some(current_state) = self.current_state {
            self.a_categoricals[current_state].sample(self.rng)
        } else {
            self.c_categorical.sample(self.rng)
        };
	
        self.current_state = Some(state);
	
        Some(HMMSample {
            x: state,
            y: self.b_categoricals[state].sample(self.rng),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::mock::StepRng;

    #[test]
    fn test_hmm_sample_creation() {
        let sample = HMMSample { x: 1, y: 2 };
        assert_eq!(sample, HMMSample { x: 1, y: 2 });
    }

    #[test]
    fn test_hmm_sample_iter() {
        // Mock RNG for deterministic results
    	let mut rng = StepRng::new(0, 1);

        let a_weights = vec![
            Categorical::from_pmf(&vec![0.1, 0.9]),
            Categorical::from_pmf(&vec![0.8, 0.2]),
        ];
	
        let b_weights = vec![
            Categorical::from_pmf(&vec![0.3, 0.7]),
            Categorical::from_pmf(&vec![0.6, 0.4]),
        ];
	
        let c_weights = Categorical::from_pmf(&vec![0.5, 0.5]);
	
        let mut iter = HMMSampleIter {
            a_categoricals: a_weights,
            b_categoricals: b_weights,
            c_categorical: c_weights,
            rng: &mut rng,
            current_state: None,
        };

        let sample1 = iter.next().unwrap();
        assert_eq!(sample1, HMMSample { x: 0, y: 0 });

        let sample2 = iter.next().unwrap();
        assert_eq!(sample2, HMMSample { x: 1, y: 0 });

        let sample3 = iter.next().unwrap();
        assert_eq!(sample3, HMMSample { x: 0, y: 0 });
    }
}