#[cfg(test)]
mod tests {
    use algos::hmm::HMM;
    use ndarray::array;

    // NB array! is a macro.
    fn setup_new_hmm() -> HMM {
        let a = array![[0.7, 0.3], [0.4, 0.6]];
        let b = array![[0.5, 0.5], [0.1, 0.9]];
        let pi = array![0.6, 0.4];

        HMM::new(a, b, pi)
    }

    #[test]
    fn test_new() {
        let hmm = setup_new_hmm();

        assert_eq!(hmm.A, array![[0.7, 0.3], [0.4, 0.6]]);

        assert_eq!(hmm.B, array![[0.5, 0.5], [0.1, 0.9]]);

        assert_eq!(hmm.PI, array![0.6, 0.4]);
    }

    #[test]
    fn test_latent_states() {
        let hmm = setup_new_hmm();

        assert_eq!(hmm.n_latent_states(), 2);
    }

    #[test]
    fn test_obs_states() {
        let hmm = setup_new_hmm();

        assert_eq!(hmm.n_obs_states(), 2);
    }
}
