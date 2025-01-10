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
            if let Some(left) = node.left.as_mut() {
                Self::insert_node(left, new_node);
            } else {
                node.left = Some(new_node);
            }
        } else {
            if let Some(right) = node.right.as_mut() {
                Self::insert_node(right, new_node);
            } else {
                node.right = Some(new_node);
            }
        }
    }

    fn search(&self, val: i32) -> bool {
        Self::search_node(&self.root, val)
    }

    fn search_node(node: &Option<Box<TreeNode>>, val: i32) -> bool {
        if let Some(node) = node {
            if node.val == val {
                true
            } else if val < node.val {
                Self::search_node(&node.left, val)
            } else {
                Self::search_node(&node.right, val)
            }
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_binary_search_tree_search -- --nocapture
    use super::*;

    #[test]
    fn test_binary_search_tree_search() {
        let mut bst = BinarySearchTree::new();
    	bst.insert(8);
	bst.insert(3);
    	bst.insert(10);
    	bst.insert(1);
    	bst.insert(6);
    	bst.insert(14);
    	bst.insert(4);
    	bst.insert(7);
    	bst.insert(13);

	assert_eq!(bst.search(6), true);
	assert_eq!(bst.search(5), false);
    }
}