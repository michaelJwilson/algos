use ndarray::Array2;
use ndarray::Axis;
use rustc_hash::FxHashMap as HashMap;

#[derive(Debug)]
enum Nucleotide {
    A,
    C,
    G,
    T,
}

impl Nucleotide {
    #[inline(always)]
    fn to_index(&self) -> usize {
        match self {
            Nucleotide::A => 0,
            Nucleotide::C => 1,
            Nucleotide::G => 2,
            Nucleotide::T => 3,
        }
    }

    fn from_char(nucleotide: char) -> Self {
        match nucleotide {
            'A' => Nucleotide::A,
            'C' => Nucleotide::C,
            'G' => Nucleotide::G,
            'T' => Nucleotide::T,
            _ => panic!("Invalid nucleotide: {}", nucleotide),
        }
    }
}

// NB Box: a pointer type that uniquely owns a heap allocation of type T
#[derive(Debug)]
pub struct Node {
    //  NB A, C, G, T for leaves
    id: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    sequence: Option<Nucleotide>,
}

#[derive(Debug)]
struct Bst {
    root: Node,
}

const DEFAULT_LIKELIHOOD: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

fn get_transition_matrix(branch_length: f64) -> Array2<f64> {
    // NB transition probability matrix for nucleotide mutations (e.g., Jukes-Cantor model)
    let mutation_rate = 0.1;

    // NB prob. of no mutation
    let p_same = 0.25 + 0.75 * (-4.0 * mutation_rate * branch_length).exp();

    // NB prob. of mutation
    let p_diff = 0.25 - 0.25 * (-4.0 * mutation_rate * branch_length).exp();

    // NB A -> A, C, G, T, etc.
    Array2::from_shape_vec(
        (4, 4),
        vec![
            p_same, p_diff, p_diff, p_diff, p_diff, p_same, p_diff, p_diff, p_diff, p_diff, p_same,
            p_diff, p_diff, p_diff, p_diff, p_same,
        ],
    )
    .unwrap()
}

pub fn compute_likelihood(
    node: &Node,
    transition_matrix: &Array2<f64>,
    branch_lengths: &HashMap<usize, f64>,
) -> [f64; 4] {
    // NB Recursive function to compute the likelihood at each node using Felsenstein's algorithm.
    if let Some(sequence) = &node.sequence {
        // NB leaf node: returns a vector with 1.0 for the observed nucleotide and 0.0 for others.
        let mut likelihood = [0.0; 4];

        likelihood[sequence.to_index()] = 1.0;

        return likelihood;
    }

    // NB internal node.
    let left_likelihood = if let Some(left) = &node.left {
        compute_likelihood(left, transition_matrix, branch_lengths)
    } else {
        // NB no left child.
        DEFAULT_LIKELIHOOD
    };

    let right_likelihood = if let Some(right) = &node.right {
        compute_likelihood(right, transition_matrix, branch_lengths)
    } else {
        // NB no right child.
        DEFAULT_LIKELIHOOD
    };

    let left_branch_length = branch_lengths
        .get(&node.left.as_ref().map_or(0, |n| n.id))
        .unwrap_or(&0.0);

    let right_branch_length = branch_lengths
        .get(&node.right.as_ref().map_or(0, |n| n.id))
        .unwrap_or(&0.0);

    // TODO what are these?
    let left_exp = (-4.0 * left_branch_length).exp();
    let right_exp = (-4.0 * right_branch_length).exp();

    let mut combined_likelihood = [0.0; 4];

    combined_likelihood
        .iter_mut()
        .enumerate()
        .for_each(|(parent_state, likelihood)| {
            let mut left_sum = 0.0;
            let mut right_sum = 0.0;

            for child_state in 0..4 {
                left_sum += transition_matrix[[parent_state, child_state]]
                    * left_likelihood[child_state]
                    * left_exp;

                right_sum += transition_matrix[[parent_state, child_state]]
                    * right_likelihood[child_state]
                    * right_exp;
            }

            *likelihood = left_sum * right_sum;
        });

    combined_likelihood
}

pub fn get_felsenstein_fixture() -> (Node, Array2<f64>, HashMap<usize, f64>) {
    let left_leaf = Node {
        id: 2,
        left: None,
        right: None,
        sequence: Some(Nucleotide::from_char('A')),
    };

    let right_leaf = Node {
        id: 3,
        left: None,
        right: None,
        sequence: Some(Nucleotide::from_char('C')),
    };

    let root = Node {
        id: 1,
        left: Some(Box::new(left_leaf)),
        right: Some(Box::new(right_leaf)),
        sequence: None,
    };

    let transition_matrix = get_transition_matrix(0.5);

    // NB node ids represent keys.
    let mut branch_lengths = HashMap::default();

    branch_lengths.insert(2, 0.5);
    branch_lengths.insert(3, 0.5);

    (root, transition_matrix, branch_lengths)
}

#[cfg(test)]
mod tests {
    // cargo test felsenstein -- test_felsenstein_transition_matrix --nocapture
    use super::*;
    use ndarray::array;

    #[test]
    fn test_felsenstein_transition_matrix() {
        let branch_length = 0.5;
        let transition_matrix = get_transition_matrix(branch_length);

        assert_eq!(transition_matrix.shape(), &[4, 4]);

        for row in transition_matrix.rows() {
            let row_sum: f64 = row.sum();

            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row sum is not sufficiently close to 1.0"
            );
        }
    }

    #[test]
    fn test_felsenstein_nucleotide_to_index() {
        assert_eq!(Nucleotide::from_char('A').to_index(), 0);
        assert_eq!(Nucleotide::from_char('C').to_index(), 1);
        assert_eq!(Nucleotide::from_char('G').to_index(), 2);
        assert_eq!(Nucleotide::from_char('T').to_index(), 3);

        let result = std::panic::catch_unwind(|| Nucleotide::from_char('X').to_index());

        assert!(result.is_err(), "Expected panic for invalid nucleotide");
    }

    #[test]
    fn test_felsenstein_compute_likelihood_leaf_node() {
        //  NB unwrap assumes never None.
        let (root, transition_matrix, branch_lengths) = get_felsenstein_fixture();
        let likelihood =
            compute_likelihood(&root.left.unwrap(), &transition_matrix, &branch_lengths);

        // NB expect likelihood to be [1.0, 0.0, 0.0, 0.0] for 'A'
        assert_eq!(likelihood, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_felsenstein_compute_likelihood_internal_node() {
        let (root, transition_matrix, branch_lengths) = get_felsenstein_fixture();
        let likelihood = compute_likelihood(&root, &transition_matrix, &branch_lengths);

        assert_eq!(likelihood.len(), 4);

        for &value in &likelihood {
            assert!(value >= 0.0, "Likelihood value is negative");
        }
    }
}
