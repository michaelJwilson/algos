use std::cell::RefCell;
use std::rc::Rc;

struct Solution;

//  NB Data-class like, i.e. memory allocation.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
	    val,
	    left: None,
	    right: None,
	}
    }
}

impl Solution {
    pub fn is_same_tree(root: Option<Rc<RefCell<TreeNode>>>, other: Option<Rc<RefCell<TreeNode>>>) -> bool {
        match (root, other) {
            (Some(node), Some(other_node)) => {
	        let left = node.borrow().left.clone();
		let right = node.borrow().right.clone();

		let other_left = other_node.borrow().left.clone();
		let other_right = other_node.borrow().right.clone();

		((node.borrow().val == other_node.borrow().val)
		    && (Solution::is_same_tree(left, other_left))
		    && (Solution::is_same_tree(right, other_right)))
            }

	    (None, None) => true,

	    // NB all other cases.
	    _ => false
        }
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_binary_tree_same -- --nocapture
    //
    // See:
    //     https://leetcode.com/problems/same-tree/description/?envType=study-plan-v2&envId=top-interview-150
    //
    use super::*;

    #[test]
    fn test_binary_tree_same() {
        let root = Rc::new(RefCell::new(TreeNode::new(1)));

	root.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(2))));
        root.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(3))));

	let other = Rc::new(RefCell::new(TreeNode::new(1)));

	other.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(2))));
	other.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(3))));
    	
    	let result = Solution::is_same_tree(Some(root), Some(other));
        let exp = true;
	
        assert!(result == exp);
    }

    #[test]
    fn test_binary_tree_not_same() {
        let root = Rc::new(RefCell::new(TreeNode::new(1)));

	root.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(2))));
        root.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(3))));

	let other = Rc::new(RefCell::new(TreeNode::new(1)));

	other.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(2))));
        other.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(4))));

	let result = Solution::is_same_tree(Some(root), Some(other));
        let exp = false;

	assert!(result == exp);
    }
}