use std::rc::Rc;
use std::cell::RefCell;

struct Solution;

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
	    val: val,
	    left: None,
	    right: None,
	}
    }
}

impl Solution {
    pub fn breadth_first_search_accumulate(node_option: Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<f64>, num_nodes: &mut Vec<i32>, level: usize) {
        match node_option {
	    Some(node) => {
	    	let node = node.borrow();    

		if level >= result.len() {
                   result.push(0.0);
                   num_nodes.push(0);
            	}

	        result[level] += node.val as f64;
		num_nodes[level] += 1;

		Self::breadth_first_search_accumulate(node.left.clone(), result, num_nodes, level + 1);
		Self::breadth_first_search_accumulate(node.right.clone(), result, num_nodes, level + 1);
	    }

	    None => (),
	}
    }

    pub fn average_of_levels(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<f64> {
	let mut num_nodes: Vec<i32> = Vec::new();
	let mut result: Vec<f64> = Vec::new();

	Self::breadth_first_search_accumulate(root, &mut result, &mut num_nodes, 0);

	result.iter().zip(num_nodes.iter()).map(|(&sum, &count)| sum / count as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_breadth_first_search_accumulate -- --nocapture
    use super::*;

    #[test]
    fn test_breadth_first_search_accumulate() {
        let exp = vec![3.0, 14.5, 11.0];
        let root = Rc::new(RefCell::new(TreeNode::new(3)));

	// NB unwrap -> result or panic.
	root.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(9))));
	root.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(20))));

	root.borrow_mut().right.as_ref().unwrap().borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(15))));
	root.borrow_mut().right.as_ref().unwrap().borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(7))));

	let level_averages = Solution::average_of_levels(Some(root));

	for ii in 0..level_averages.len() {
	    println!("{:?} \t {:?}", level_averages[ii], exp[ii]);

	    assert_eq!(level_averages[ii], exp[ii]);
	}
    }
}
