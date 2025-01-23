use statrs::function::gamma::digamma;
use std::f64::consts::PI;
use rgsl::psi::trigamma::psi_1;
use rand::prelude::*;
use rand_distr::{Distribution, Binomial, Beta};

const ATOL: f64 = 1.0e-14;
const EGAMMA: f64 = 0.577215664901532860606512090082402431;

pub fn initial_inverse_digamma(y: f64) -> f64 {
    // NB eqn. (149) of Minka, 2000.
    if y >= -2.22 {
       y.exp() + 0.5
    } else {
       - 1. / (y + EGAMMA)
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
	
        x = x + increment;
    }

    x
}   

pub fn sample_beta_binomial(alpha: f64, beta: f64, num_trials: u64) -> Vec<u64>{
    let beta = Beta::new(alpha, beta).unwrap();
    let mut result = Vec::new();

    for ii in 0..num_trials {
        let pp = beta.sample(&mut rand::thread_rng());

	let bin = Binomial::new(1, pp).unwrap();
	let sample = bin.sample(&mut rand::thread_rng());
	
    	result.push(sample);
    }
    
    result
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
	let exp: f64 = PI * PI / 6. - 5./4.;

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
	
    	let result = sample_beta_binomial(alpha, beta, num_trials);

	let num_heads: u64 = result.clone().into_iter().sum();
	let num_tails: u64 = num_trials - num_heads;

	let exp_prob: f64 = alpha / (alpha + beta);
	let obs_prob: f64 = num_heads as f64 / num_trials as f64;

	assert!((obs_prob - exp_prob).abs() < 1.0e-3);
    }
}

