use ndarray::prelude::*;
use spectral::numeric::{FloatAssertions, OrderedAssertions};
use spectral::{assert_that, asserting};

#[derive(Debug)]
#[cfg_attr(feature = "serde-1", derive(Serialize, Deserialize))]
pub struct HMM {
    pub A: Array2<f64>,
    pub B: Array2<f64>,
    pub PI: Array1<f64>,
}

const TOLERANCE: f64 = 1e-6;

impl HMM {
    pub fn new(A: Array2<f64>, B: Array2<f64>, PI: Array1<f64>) -> Self {
    	// Block for dimension checks.
        {
            asserting("B must have a positive number of rows")
                .that(&B.nrows())
                .is_greater_than(0);
		
            asserting("B must have a positive number of columns")
                .that(&B.ncols())
                .is_greater_than(0);
		
            assert_eq!(
                A.nrows(),
                B.nrows(),
                "A and B must have the same number of rows"
            );
	    
            assert_eq!(A.nrows(), A.ncols(), "A must be square");
            assert_eq!(A.nrows(), PI.len(), "π must be of length N");
        }

	// Block for checking rows of A are valid probabilities.
        {
            for a in &A {
                assert_that(a).is_greater_than_or_equal_to(0.0)
            }
	    
            for row in A.rows() {
                asserting("Each row of A must sum to 1")
                    .that(&row.sum())
                    .is_close_to(1.0, TOLERANCE);
            }
        }

        // Block for checking rows of B	are valid probabilities.
        {
            for b in &B {
                assert_that(b).is_greater_than_or_equal_to(0.0)
            }

            for row in B.rows() {
                asserting("Each row of B must sum to 1")
                    .that(&row.sum())
                    .is_close_to(1.0, TOLERANCE);
            }
        }

	// Block for checking that π is a distribution.
        {
            for pi in &PI {
                assert_that(pi).is_greater_than_or_equal_to(0.0)
            }

            asserting("π must sum to 1")
                .that(&PI.sum())
                .is_close_to(1.0, TOLERANCE);
        }

        Self { A, B, PI }
    }

    /// $N$, the number of states in this HMM
    pub fn n_latent_states(&self) -> usize {
        self.B.nrows()
    }

    /// $K$, the number of possible observations that this model can emit
    pub fn n_obs_states(&self) -> usize {
        self.B.ncols()
    }
}