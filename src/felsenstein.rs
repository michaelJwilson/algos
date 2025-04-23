use ndarray::Axis;
use ndarray::Array2;
use rustc_hash::FxHashMap as HashMap;

/// Represents a node in the binary search tree.
#[derive(Debug)]
struct Node {
    id: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    sequence: Option<char>, // NB A, C, G, T for leaves
}

#[derive(Debug)]
struct BST {
    root: Node,
}


fn get_transition_matrix(branch_length: f64) -> Array2<f64> {
    // NB transition probability matrix for nucleotide mutations (e.g., Jukes-Cantor model)
    let mutation_rate = 0.1;

    // NB prob. of no mutation
    let p_same = 0.25 + 0.75 * (-4.0 * mutation_rate * branch_length).exp();

    // NB prob. of mutation
    let p_diff = 0.25 - 0.25 * (-4.0 * mutation_rate * branch_length).exp();

    Array2::from_shape_vec(
        (4, 4),
        vec![
            p_same, p_diff, p_diff, p_diff, // A -> A, C, G, T
            p_diff, p_same, p_diff, p_diff, // C -> A, C, G, T
            p_diff, p_diff, p_same, p_diff, // G -> A, C, G, T
            p_diff, p_diff, p_diff, p_same, // T -> A, C, G, T
        ],
    )
    .unwrap()
}

/// Map nucleotides to indices for the transition matrix.
fn nucleotide_to_index(nucleotide: char) -> usize {
    match nucleotide {
        'A' => 0,
        'C' => 1,
        'G' => 2,
        'T' => 3,
        _ => panic!("Invalid nucleotide: {}", nucleotide),
    }
}

fn compute_likelihood(
    node: &Node,
    transition_matrix: &Array2<f64>,
    branch_lengths: &HashMap<usize, f64>,
) -> Vec<f64> {
    // NB Recursive function to compute the likelihood at each node using Felsenstein's algorithm.
    if let Some(sequence) = node.sequence {
        // Leaf node: returns a vector with 1.0 for the observed nucleotide and 0.0 for others.
        let mut likelihood = vec![0.0; 4];

        likelihood[nucleotide_to_index(sequence)] = 1.0;

        return likelihood;
    }

    // Internal node: Compute likelihoods for left and right children.
    let left_likelihood = if let Some(left) = &node.left {
        compute_likelihood(left, transition_matrix, branch_lengths)
    } else {
        vec![0.25; 4] // Default likelihood if no left child
    };

    let right_likelihood = if let Some(right) = &node.right {
        compute_likelihood(right, transition_matrix, branch_lengths)
    } else {
        vec![0.25; 4] // Default likelihood if no right child
    };

    // Combine likelihoods from left and right children
    let mut combined_likelihood = vec![0.0; 4];

    for parent_state in 0..4 {
        let mut left_sum = 0.0;
        let mut right_sum = 0.0;

        for child_state in 0..4 {
            let left_branch_length = branch_lengths.get(&node.left.as_ref().map_or(0, |n| n.id)).unwrap_or(&0.0);
            let right_branch_length = branch_lengths.get(&node.right.as_ref().map_or(0, |n| n.id)).unwrap_or(&0.0);

            left_sum += transition_matrix[[parent_state, child_state]]
                * left_likelihood[child_state]
                * (-4.0 * left_branch_length).exp();

            right_sum += transition_matrix[[parent_state, child_state]]
                * right_likelihood[child_state]
                * (-4.0 * right_branch_length).exp();
        }

        combined_likelihood[parent_state] = left_sum * right_sum;
    }

    combined_likelihood
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

            assert!((row_sum - 1.0).abs() < 1e-6, "Row sum is not sufficiently close to 1.0");
        }
    }

    #[test]
    fn test_felsenstein_nucleotide_to_index() {
        assert_eq!(nucleotide_to_index('A'), 0);
        assert_eq!(nucleotide_to_index('C'), 1);
        assert_eq!(nucleotide_to_index('G'), 2);
        assert_eq!(nucleotide_to_index('T'), 3);

        let result = std::panic::catch_unwind(|| nucleotide_to_index('X'));

        assert!(result.is_err(), "Expected panic for invalid nucleotide");
    }

    #[test]
    fn test_felsenstein_compute_likelihood_leaf_node() {
        let leaf_node = Node {
            id: 1,
            left: None,
            right: None,
            sequence: Some('A'),
        };

        let transition_matrix = get_transition_matrix(0.5);
        let branch_lengths = HashMap::default();

        let likelihood = compute_likelihood(&leaf_node, &transition_matrix, &branch_lengths);

        // NB expect likelihood to be [1.0, 0.0, 0.0, 0.0] for 'A'
        assert_eq!(likelihood, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_felsenstein_compute_likelihood_internal_node() {
        let left_child = Node {
            id: 2,
            left: None,
            right: None,
            sequence: Some('A'),
        };

        let right_child = Node {
            id: 3,
            left: None,
            right: None,
            sequence: Some('C'),
        };

        let root = Node {
            id: 1,
            left: Some(Box::new(left_child)),
            right: Some(Box::new(right_child)),
            sequence: None,
        };

        let transition_matrix = get_transition_matrix(0.5);
        let mut branch_lengths = HashMap::default();

        // NB branch length for left child
        branch_lengths.insert(2, 0.5);

        // NB branch length for right child
        branch_lengths.insert(3, 0.5);

        let likelihood = compute_likelihood(&root, &transition_matrix, &branch_lengths);

        assert_eq!(likelihood.len(), 4);

        for &value in &likelihood {
            assert!(value >= 0.0, "Likelihood value is negative");
        }
    }
}