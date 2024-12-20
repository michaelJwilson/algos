#[cfg(test)]
mod tests {
    use algos::hmm::HMM;
    use ndarray::array;

    #[test]
    pub fn test_new() {
        let A = array![
            [0.7, 0.3],
            [0.4, 0.6]
	];

        let B = array![
            [0.5, 0.5],
            [0.1, 0.9]
        ];

        let PI = array![0.6, 0.4];

        let hmm = HMM::new(A, B, PI);

        assert_eq!(hmm.A, array![
            [0.7, 0.3],
            [0.4, 0.6]
        ]);

        assert_eq!(hmm.B, array![
            [0.5, 0.5],
            [0.1, 0.9]
        ]);

        assert_eq!(hmm.PI, array![0.6, 0.4]);
    }

    #[test]
    pub fn test_latent_states() {
    

    }
}