use algos::weighted_choice::WeightedChoice
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HMMSample {
    pub x: usize,
    pub y: usize,
}

/// An iterator that returns random samples from an HMM
pub struct HMMSampleIter<'a, R: Rng + ?Sized + 'a> {
    a_weighted_choices: Vec<WeightedChoice>,
    b_weighted_choices: Vec<WeightedChoice>,
    c_weighted_choice: WeightedChoice,
    rng: &'a mut R,
    current_state: Option<usize>,
}

impl<'a, R: Rng + ?Sized> Iterator for HMMSampleIter<'a, R> {
    type Item = HMMSample;

    fn next(&mut self) -> Option<Self::Item> {
        let state = if let Some(current_state) = self.current_state {
            self.a_weighted_choices[current_state].sample(self.rng)
        } else {
            self.c_weighted_choice.sample(self.rng)
        };
        self.current_state = Some(state);
        Some(HMMSample {
            x: state,
            y: self.b_weighted_choices[state].sample(self.rng),
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
        let a_weights = vec![
            WeightedChoiceFloat::new(vec![0.1, 0.9]),
            WeightedChoiceFloat::new(vec![0.8, 0.2]),
        ];
        let b_weights = vec![
            WeightedChoiceFloat::new(vec![0.3, 0.7]),
            WeightedChoiceFloat::new(vec![0.6, 0.4]),
        ];
        let c_weights = WeightedChoiceFloat::new(vec![0.5, 0.5]);

        let mut rng = StepRng::new(0, 1); // Mock RNG for deterministic results
        let mut iter = HMMSampleIter {
            a_weighted_choices: a_weights,
            b_weighted_choices: b_weights,
            c_weighted_choice: c_weights,
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