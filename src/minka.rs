use rand_distr::{Beta, Binomial, Distribution};
use rgsl::psi::trigamma::psi_1;
use statrs::function::gamma::{digamma, gamma};
use std::f64::consts::PI;

const ATOL: f64 = 1.0e-14;
const EGAMMA: f64 = 0.577_215_664_901_532_9;

pub fn initial_inverse_digamma(y: f64) -> f64 {
    // NB eqn. (149) of Minka, 2000.
    if y >= -2.22 {
        y.exp() + 0.5
    } else {
        -1. / (y + EGAMMA)
    }
}

pub fn trigamma(x: f64) -> f64 {
    psi_1(x)
}

pub fn inverse_digamma(y: f64) -> f64 {
    // NB Newton's method implementation of inverse,
    //    see Minka (2000), eqn. (146).
    let mut x = initial_inverse_digamma(y);
    let mut increment: f64 = f64::MAX;

    while increment.abs() > ATOL {
        increment = -(digamma(x) - y) / trigamma(x);

        x += increment;
    }

    x
}

pub fn sample_beta_binomial(
    alpha: f64,
    beta: f64,
    num_trials: u64,
    num_samples: u64,
) -> Vec<Vec<u64>> {
    let beta = Beta::new(alpha, beta).unwrap();
    let mut result = Vec::new();

    for _ss in 0..num_samples {
        let mut interim = Vec::new();

        for _ii in 0..num_trials {
            let pp = beta.sample(&mut rand::rng());
            let bin = Binomial::new(1, pp).unwrap();

            let sample = bin.sample(&mut rand::rng());

            interim.push(sample);
        }

        result.push(interim);
    }

    result
}

pub fn likelihood_beta_binomial(training_data: Vec<Vec<u64>>, alphas: Vec<f64>) -> f64 {
    let mut result: f64 = 1.0;
    let sum_alphas: f64 = alphas.iter().sum();

    for data in &training_data {
        let num_obs = data.len();
        let mut interim = gamma(sum_alphas) / gamma(num_obs as f64 + sum_alphas);

        for (jj, alpha) in alphas.iter().enumerate() {
            let num_in_class: f64 = data
                .iter()
                .filter(|&&xx| xx == jj as u64)
                .count() as f64;

            interim *= gamma(num_in_class + alpha);
            interim /= gamma(*alpha);
        }

        result *= interim;
    }

    result
}

pub fn polya_damped_counts(class_counts: Vec<f64>, alphas: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();

    for ii in 0..class_counts.len() {
        let interim = digamma(class_counts[ii] + alphas[ii]) - digamma(alphas[ii]);

        result.push(alphas[ii] * interim);
    }

    result
}

pub fn max_likelihood_polya_mean(training_data: Vec<Vec<u64>>, alphas: Vec<f64>) -> Vec<f64> {
    // NB max. likelihood estimate of polya mean @ fixed precision;  eqn. (118) of Minka (2000).
    let mut total_damped_counts: Vec<f64> = vec![0.; alphas.len()];

    for data in &training_data {
        let mut sample_class_counts: Vec<f64> = Vec::new();

        for jj in 0..alphas.len() {
            let num_in_class: f64 = data
                .iter()
                .filter(|&&xx| xx == jj as u64)
                .count() as f64;
            sample_class_counts.push(num_in_class);
        }

        let sample_damped_counts = polya_damped_counts(sample_class_counts, &alphas);

        for (total, damped) in total_damped_counts
            .iter_mut()
            .zip(sample_damped_counts.iter())
        {
            *total += damped;
        }
    }

    let norm: f64 = total_damped_counts.iter().sum();

    total_damped_counts
        .iter()
        .map(|&count| count / norm)
        .collect()
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_minka -- --nocapture
    use super::*;

    // TODO achieve higher precision.
    const TEST_TOL: f64 = 1.0e-14;

    #[test]
    pub fn test_minka_digamma() {
        assert!((digamma(1.) + EGAMMA).abs() < TEST_TOL);
    }

    #[test]
    pub fn test_minka_trigamma() {
        let result = trigamma(3.);
        let exp: f64 = PI * PI / 6. - 5. / 4.;

        assert!((result - exp).abs() < TEST_TOL);
    }

    #[test]
    pub fn test_minka_inverse_digamma() {
        let xs: Vec<f64> = vec![1., 2., 10., 100.];
        let exps: Vec<f64> = xs.iter().map(|xx| digamma(inverse_digamma(*xx))).collect();

        for ii in 0..xs.len() {
            assert!((xs[ii] - exps[ii]).abs() < TEST_TOL);
        }
    }

    #[test]
    pub fn test_sample_beta_binomial() {
        let num_trials: u64 = 1_000_000;

        let alpha: f64 = 20.;
        let beta: f64 = 65.;

        let result = sample_beta_binomial(alpha, beta, num_trials, 1)[0].clone();

        let num_heads: u64 = result.clone().into_iter().sum();
        let num_tails: u64 = num_trials - num_heads;

        let exp_prob: f64 = alpha / (alpha + beta);
        let obs_prob: f64 = num_heads as f64 / num_trials as f64;

        //  TODO precision?
        assert!((obs_prob - exp_prob).abs() < 1.0e-3);
    }

    #[test]
    pub fn test_likelihood_beta_binomial() {
        let mut training_data = Vec::new();
        training_data.push(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        let alphas = vec![0.5, 0.5];
        let result = likelihood_beta_binomial(training_data, alphas);

        println!("{:?}", result);
    }

    #[test]
    pub fn test_polya_damping() {
        let class_counts: Vec<f64> = vec![1., 4., 7., 10.];
        let alphas: Vec<f64> = vec![0.1, 1., 3., 10.];

        // NB sanity checked against Fig. 1 of Minka (2000).
        let exp: Vec<f64> = vec![0.9999, 2.0833, 3.9869, 7.1877];
        let result = polya_damped_counts(class_counts, &alphas);
    }

    #[test]
    pub fn test_max_likelihood_polya_mean() {
        let mut training_data = Vec::new();
        training_data.push(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        let alphas = vec![0.05, 0.95];

        // NB if alphas = vec![100., 100.], we get equivalence to ML
        //    multinomial with exp = vec![0.5, 0.5];

        let result = max_likelihood_polya_mean(training_data, alphas);

        // TODO assert for large alpha.
        println!("{:?}", result);
    }
}
