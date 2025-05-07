use statrs::function::gamma::{ln_gamma, digamma};
use statrs::function::factorial::ln_factorial;

pub fn basic_nbinom_logpmf(
    k: Vec<f64>,
    r: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
) -> Vec<Vec<f64>> {
    //
    //  Efficient negative binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Negative_binomial_distribution

    //  NB parameter-dependent only
    let gr: Vec<f64> = r.iter().map(|&x| ln_gamma(x)).collect();
    let lnp: Vec<f64> = p.iter().map(|&x| x.ln()).collect();
    let lnq: Vec<f64> = q.iter().map(|&x| (1. - x).ln()).collect();

    let result: Vec<Vec<f64>> = k.iter().map(|&k_val| {
        // NB data-dependent only
        let zero_point = -ln_factorial(k_val as u64);

        let row: Vec<f64> = r.iter().enumerate().map(|(ss, &r_val)| {
            let mut interim = zero_point;

            interim += k_val * lnq[ss] + r_val * lnp[ss] - gr[ss];
            interim += ln_gamma(k_val + r_val);

            interim
        }).collect();

        row
    }).collect();

    result
}

pub fn get_streaming_fixture(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let k: Vec<f64> = (1..=n).map(|x| x as f64).collect();
    let r: Vec<f64> = (1..=n).map(|x| (x % 10 + 1) as f64).collect();
    let p: Vec<f64> = (1..=n).map(|x| 0.1 + 0.8 * (x as f64 / n as f64)).collect();
    let q: Vec<f64> = p.iter().map(|&x| 1.0 - x).collect();

    (k, r, p, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_nbinom() {
        let (k, r, p, q) = get_streaming_fixture(1_000);
        let result = basic_nbinom_logpmf(k, r, p, q);

        println!("Result: {:?}", result);
    }
}