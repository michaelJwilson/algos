#[derive(Debug)]
struct TreeNode {
    val: i32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

struct BinarySearchTree {
    root: Option<Box<TreeNode>>,
}

impl BinarySearchTree {
    fn new() -> Self {
        BinarySearchTree { root: None }
    }

    // NB insertions into a binary search tree satisfy left/right cuts for all levels from root.
    //    i.e. as for a random-forest classifier, a series of less than/greater than questions is
    //         asked with one question per level starting at the root.
    fn insert(&mut self, val: i32) {
        let new_node = Box::new(TreeNode::new(val));

        if let Some(root) = self.root.as_mut() {
            Self::insert_node(root, new_node);
        } else {
            self.root = Some(new_node);
        }
    }

    fn insert_node(node: &mut Box<TreeNode>, new_node: Box<TreeNode>) {
        if new_node.val < node.val {
            // NB left must potentially be mutable.
            if let Some(left) = node.left.as_mut() {
                Self::insert_node(left, new_node);
            } else {
                node.left = Some(new_node);
            }
        } else if let Some(right) = node.right.as_mut() {
            Self::insert_node(right, new_node);
        } else {
            node.right = Some(new_node);
        }
    }

    fn search(&self, val: i32) -> bool {
        Self::search_node(&self.root, val)
    }

    fn search_node(node: &Option<Box<TreeNode>>, val: i32) -> bool {
        if let Some(node) = node {
            /*
            if node.val == val {
                true
            } else if val < node.val {
                Self::search_node(&node.left, val)
            } else {
                Self::search_node(&node.right, val)
            }*/

            match val.cmp(&node.val) {
                std::cmp::Ordering::Equal => true,
                std::cmp::Ordering::Less => Self::search_node(&node.left, val),
                std::cmp::Ordering::Greater => Self::search_node(&node.right, val),
            }
        } else {
            false
        }
    }

    fn min_diff(&self) -> i32 {
        let mut prev = None;
        let mut min_diff = i32::MAX;

        Self::in_order_traversal_min_diff(&self.root, &mut prev, &mut min_diff);

        min_diff
    }

    // NB values for an in-order traversal of a BST are in ascending order.
    fn in_order_traversal_min_diff(
        node: &Option<Box<TreeNode>>,
        prev: &mut Option<i32>,
        min_diff: &mut i32,
    ) {
        if let Some(node) = node {
            // NB delegate left first for in-order traversal.
            Self::in_order_traversal_min_diff(&node.left, prev, min_diff);

            // NB unpack prev to prev_val.
            if let Some(prev_val) = prev {
                // NB update min_diff.
                *min_diff = (*min_diff).min((node.val - *prev_val).abs());
            }

            *prev = Some(node.val);

            Self::in_order_traversal_min_diff(&node.right, prev, min_diff);
        }
    }

    fn print_in_order(&self) {
        Self::print_in_order_node(&self.root);
        println!();
    }

    fn print_in_order_node(node: &Option<Box<TreeNode>>) {
        if let Some(node) = node {
            Self::print_in_order_node(&node.left);
            print!("{} ", node.val);
            Self::print_in_order_node(&node.right);
        }
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_binary_search_tree_search -- --nocapture
    use super::*;

    fn generate_test_tree(index: i32) -> BinarySearchTree {
        let mut bst = BinarySearchTree::new();

        // NB see https://www.cs.usfca.edu/~galles/visualization/BST.html
        let elements = if index == 0 {
            vec![8, 3, 10, 1, 6, 14, 4, 7, 13]
        } else if index == 1 {
            vec![4, 2, 6, 1, 3]
        } else {
            vec![1, 0, 48, 12, 49]
        };

        for el in elements.iter() {
            bst.insert(*el);
        }

        bst
    }

    #[test]
    fn test_binary_search_tree_search() {
        let bst = generate_test_tree(0);

        assert_eq!(bst.search(6), true);
        assert_eq!(bst.search(5), false);
    }

    #[test]
    fn test_binary_search_tree_inorder() {
        let bst = generate_test_tree(0);

        // NB [1 3 4 6 7 8 10 13 14].
        bst.print_in_order();
    }

    #[test]
    fn test_binary_search_tree_min_diff() {
        assert_eq!(generate_test_tree(1).min_diff(), 1);
        assert_eq!(generate_test_tree(2).min_diff(), 1);
    }
}
