use itertools::izip;
use lru::LruCache;
use smallvec::SmallVec;
use statrs::function::factorial::ln_factorial;
use statrs::function::gamma::{digamma, ln_gamma};
use std::collections::HashMap;
use std::time::Instant;

// NB basic runtime: 9.0589 ms
pub fn basic_nbinom_logpmf(
    k: &[f64],
    r: &[f64],
    p: &[f64],
    responsibility: &[f64],
) -> Vec<Vec<f64>> {
    //
    //  Efficient negative binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Negative_binomial_distribution

    //  NB parameter-dependent only
    let gr: Vec<f64> = r.iter().map(|&x| ln_gamma(x)).collect();
    let lnp: Vec<f64> = p.iter().map(|&x| x.ln()).collect();
    let lnq: Vec<f64> = p.iter().map(|&x| (1. - x).ln()).collect();

    let result: Vec<Vec<f64>> = k
        .iter()
        .enumerate()
        .map(|(ii, &k_val)| {
            // NB data-dependent only
            let zero_point = -ln_factorial(k_val as u64);

            let row: Vec<f64> = r
                .iter()
                .enumerate()
                .map(|(ss, &r_val)| {
                    let mut interim = 0.0;

                    interim += zero_point;
                    interim += k_val * lnq[ss] + r_val * lnp[ss] - gr[ss];
                    interim += ln_gamma(k_val + r_val);
                    interim *= responsibility[ii];

                    interim
                })
                .collect();

            row
        })
        .collect();

    result
}

// NB stream runtime: 393.88 µs (22x); 11.979 µs for iterations with no checking;
pub fn stream_nbinom_logpmf(k: &[f64], r: &[f64], p: &[f64], responsibility: &[f64]) -> f64 {
    //
    //  Compute the total log-likelihood for the negative binomial distribution.
    //
    //  see: https://en.wikipedia.org/wiki/Negative_binomial_distribution

    // Precompute parameter-dependent terms
    let gr: Vec<f64> = r.iter().map(|&x| ln_gamma(x)).collect();
    let lnp: Vec<f64> = p.iter().map(|&x| x.ln()).collect();
    let lnq: Vec<f64> = p.iter().map(|&x| (1. - x).ln()).collect();

    let mut total_log_likelihood = 0.0;

    // NB array iterations allow for no per-iteration bounds checking.
    for (kk, &responsibility_val) in k.iter().zip(responsibility.iter()) {
        let zero_point = -ln_gamma(kk + 1.0);

        for (&rr, &pp, &qq, &gg) in izip!(r, &lnp, &lnq, &gr) {
            let mut interim = 0.0;

            interim += zero_point;
            interim += kk * qq + rr * pp - gg;
            interim += ln_gamma(kk + rr);
            interim *= responsibility_val;

            total_log_likelihood += interim;
        }
    }

    total_log_likelihood
}

// NB runtime:  167.73 µs
pub fn basic_betabinom_logpmf(
    k: &[f64],
    n: &[f64],
    alpha: &[f64],
    beta: &[f64],
    responsibility: &[f64],
) -> Vec<Vec<f64>> {
    //
    //  Compute the Beta-Binomial log PMF for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Beta-binomial_distribution

    let ln_beta_ab: Vec<f64> = alpha
        .iter()
        .zip(beta.iter())
        .map(|(&a, &b)| ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b))
        .collect();

    // NB there is no capacity so repeated allocation of vectors.
    let result: Vec<Vec<f64>> = k
        .iter()
        .zip(n.iter())
        .map(|(&k_val, &n_val)| {
            let zero_point =
                ln_gamma(n_val + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n_val - k_val + 1.0);

            // NB index ss access to ln_beta_ab -> bounds checking, etc.
            let row: Vec<f64> = alpha
                .iter()
                .zip(beta.iter())
                .enumerate()
                .map(|(ss, (&a, &b))| {
                    let interim = zero_point + ln_gamma(k_val + a) + ln_gamma(n_val - k_val + b)
                        - ln_gamma(n_val + a + b)
                        - ln_beta_ab[ss];

                    responsibility[ss] * interim
                })
                .collect();

            row
        })
        .collect();

    result
}

// NB runtime:  1.75 ns;  248.52 ps
pub fn stream_betabinom_logpmf(
    k: &[f64],
    n: &[f64],
    alpha: &[f64],
    beta: &[f64],
    responsibility: &[f64],
) -> f64 {
    //
    //  Compute the total log-likelihood for the Beta-Binomial distribution.
    //
    //  see: https://en.wikipedia.org/wiki/Beta-binomial_distribution

    let mut total_log_likelihood = 0.0;

    //  NB responsibility should be N X M for M states.
    for (&k_val, &n_val, &responsibility_val) in izip!(k, n, responsibility) {
        let zero_point =
            ln_gamma(n_val + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n_val - k_val + 1.0);

        //  NB index ss access to ln_beta_ab -> bounds checking, etc.
        for (&a, &b) in izip!(alpha, beta) {
            let bab = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
            let interim = zero_point + ln_gamma(k_val + a) + ln_gamma(n_val - k_val + b)
                - ln_gamma(n_val + a + b)
                - bab;

            total_log_likelihood += responsibility_val * interim;
        }
    }

    total_log_likelihood
}

// NB ludicrously slow.
pub fn cached_gamma(values: &[usize], cache_size: usize) -> f64 {
    let mut cache = LruCache::new(cache_size);
    let mut total_sum = 0.0;

    for &value in values {
        if let Some(&cached_result) = cache.get(&value) {
            total_sum += cached_result;
        } else {
            let result = ln_gamma(value as f64);

            cache.put(value, result);

            total_sum += result;
        }
    }

    total_sum
}

pub fn get_nbinom_fixture(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut k = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);

    for x in 1..=n {
        let prob = 0.1 + 0.8 * (x as f64 / n as f64);

        k.push(x as f64);
        r.push((x % 10 + 1) as f64);
        p.push(prob);
        q.push(1.0 - prob);
    }

    (k, r, p, q)
}

pub fn get_betabinom_fixture(
    num_trials: usize,
    num_states: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut k = Vec::with_capacity(num_trials);
    let mut trials = Vec::with_capacity(num_trials);
    let mut responsibility: Vec<f64> = Vec::with_capacity(num_trials);

    let mut alpha = Vec::with_capacity(num_states);
    let mut beta = Vec::with_capacity(num_states);

    for x in 1..=num_trials {
        let coverage = 10.0 + (x % 10) as f64;
        let successes = (x % 10) as f64;

        k.push(successes);
        trials.push(coverage);
        responsibility.push(1.0);
    }

    for x in 1..=num_states {
        let a = 1.0 + (x % 5) as f64;
        let b = 1.0 + ((x + 2) % 5) as f64;

        alpha.push(a);
        beta.push(b);
    }

    (k, trials, alpha, beta, responsibility)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_betabinom_fixture() {
        let (num_trials, num_states) = (1_000, 5);
        let (k, trials, alpha, beta, responsibility) =
            get_betabinom_fixture(num_trials, num_states);
    }

    #[test]
    fn test_streaming_nbinom() {
        let (k, r, p, q) = get_nbinom_fixture(1_000);

        let exp: f64 = basic_nbinom_logpmf(&k, &r, &p, &q)
            .iter()
            .flat_map(|row| row.iter())
            .sum();
        let result = stream_nbinom_logpmf(&k, &r, &p, &q);

        assert_eq!(result, exp);
    }

    #[test]
    fn test_streaming_betabinom() {
        let (k, n, alpha, beta, q) = get_betabinom_fixture(1_000, 5);

        let exp: f64 = basic_betabinom_logpmf(&k, &n, &alpha, &beta, &q)
            .iter()
            .flat_map(|row| row.iter())
            .sum();
        let result: f64 = basic_betabinom_logpmf(&k, &n, &alpha, &beta, &q)
            .iter()
            .flat_map(|row| row.iter())
            .sum();

        assert_eq!(result, exp);
    }
}
