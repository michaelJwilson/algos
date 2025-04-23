use ::std::cell::RefCell;
use ::std::rc::Rc;

struct Solution;

// DEPRECATE Option<Rc<RefCell<TreeNode>>> for Option<Box<TreeNode>>
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
    pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
        Self::dfs(&root, target_sum)
    }

    // NB node arg. accepted by reference.
    pub fn dfs(node: &Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
        if let Some(node) = node {
            let node = node.borrow();
            let val = node.val;

            if node.left.is_none() && node.right.is_none() {
                return val == target_sum;
            }

            // NB recursively check the left and right subtrees
            Self::dfs(&node.left, target_sum - val) || Self::dfs(&node.right, target_sum - val)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    // cargo test test_binary_tree_path_sum_large -- --nocapture
    // 
    // See:
    //     https://leetcode.com/problems/path-sum/description/?envType=study-plan-v2&envId=top-interview-150
    //
    use super::*;

    #[test]
    fn test_binary_tree_path_sum_small() {
        let root = Rc::new(RefCell::new(TreeNode::new(1)));

        root.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(2))));
        root.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(3))));

        let target_sum = 5;

        let exp = false;
        let result = Solution::has_path_sum(Some(root), target_sum);

        assert!(result == exp);
    }

    #[test]
    fn test_binary_tree_path_sum_large() {
        let root = Rc::new(RefCell::new(TreeNode::new(5)));

        root.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(4))));
        root.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(8))));
        root.borrow_mut().left.as_ref().unwrap().borrow_mut().left =
            Some(Rc::new(RefCell::new(TreeNode::new(11))));
        root.borrow_mut()
            .left
            .as_ref()
            .unwrap()
            .borrow_mut()
            .left
            .as_ref()
            .unwrap()
            .borrow_mut()
            .left = Some(Rc::new(RefCell::new(TreeNode::new(7))));
        root.borrow_mut()
            .left
            .as_ref()
            .unwrap()
            .borrow_mut()
            .left
            .as_ref()
            .unwrap()
            .borrow_mut()
            .right = Some(Rc::new(RefCell::new(TreeNode::new(2))));
        root.borrow_mut().right.as_ref().unwrap().borrow_mut().left =
            Some(Rc::new(RefCell::new(TreeNode::new(13))));
        root.borrow_mut().right.as_ref().unwrap().borrow_mut().right =
            Some(Rc::new(RefCell::new(TreeNode::new(4))));
        root.borrow_mut()
            .right
            .as_ref()
            .unwrap()
            .borrow_mut()
            .right
            .as_ref()
            .unwrap()
            .borrow_mut()
            .right = Some(Rc::new(RefCell::new(TreeNode::new(1))));

        let target_sum = 22;

        let exp = true;
        let result = Solution::has_path_sum(Some(root), target_sum);

        assert!(result == exp);
    }

    #[test]
    fn test_binary_tree_path_sum_empty() {
        let root = None;
        let target_sum = 0;

        let exp = false;
        let result = Solution::has_path_sum(root, target_sum);

        assert!(result == exp);
    }
}
