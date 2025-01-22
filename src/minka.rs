use statrs::function::gamma::digamma;
use std::f64::consts::PI;

const ATOL: f64 = 1.0e-12;
const EGAMMA: f64 = 0.577215664901532860606512090082402431;

pub fn initial_inverse_digamma(y: f64) -> f64 {
    // NB eqn. (149) of Minka, 2000.
    if y >= -2.22 {
       y.exp() + 0.5
    } else {
       - 1. / (y + EGAMMA)
    }
}


// TODO available in rust gsl?  https://docs.rs/GSL/latest/rgsl/
pub fn trigamma(x: f64) -> f64 {
    let mut nn: f64 = 0.0;
    let mut result: f64 = 0.0;
    let mut increment: f64 = f64::MAX;

    while increment > ATOL {
        increment = 1. / (x + nn).powi(2);	
        result += increment;

	nn += 1.; 
    }

    result
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


#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_minka -- --nocapture
    use super::*;

    #[test]
    pub fn test_minka_digamma() {	
	assert!((digamma(1.) + EGAMMA).abs() < 1.0e-6);
    }

    #[test]
    pub fn test_minka_trigamma() {
        let result = trigamma(3.);
	let exp = PI * PI / 6. - 5./4.;

	// println!("{:?}  {:?}  {:?}", result, exp, (result - exp).abs());

	assert!((result - exp).abs() < 1.0e-6);
    }

    #[test]
    pub fn test_minka_inverse_digamma() {
    	let xs: Vec<f64> = vec![1., 2., 10., 100.];
        let exps: Vec<f64> = xs.iter().map(|xx| digamma(inverse_digamma(*xx))).collect();

	for ii in 0..xs.len() {
	    assert!((xs[ii] - exps[ii]).abs() < 1.0e-12);
	}
    }
}

