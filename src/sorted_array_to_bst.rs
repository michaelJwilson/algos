// NB Data-class like.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Box<TreeNode>>,
    pub right: Option<Box<TreeNode>>,
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

pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Box<TreeNode>> {
    // NB function wrapper with nums input as ref., not Vector.
    fn helper(nums: &[i32]) -> Option<Box<TreeNode>> {
       if nums.is_empty() {
       	  return None;
       }

       let mid = nums.len() / 2;
       let mut root = TreeNode::new(nums[mid]);

       // NB recursive call.
       root.left = helper(&nums[..mid]);
       root.right = helper(&nums[mid + 1..]);

       Some(Box::new(root))
    }

    helper(&nums)
}

pub fn print_tree(node: &Option<Box<TreeNode>>, depth: usize) {
    // NB the ref keyword is used to create a ref. for borrow instead
    //    of moving.  Here node may be used again later in the code.
    if let Some(ref n) = node {
        // NB post-order.
        print_tree(&n.right, depth + 1);

	// NB the dollary sign is used for dynamic width formatting.
        println!("{:indent$}{}", "", n.val, indent = depth * 4);
	
        print_tree(&n.left, depth + 1);
    }
}

pub fn pre_order_traversal(node: &Option<Box<TreeNode>>) -> Vec<i32> {
    if let Some(ref n) = node {
        let mut result = Vec::new();

	result.extend(pre_order_traversal(&n.left));
	result.extend(vec![n.val]);
	result.extend(pre_order_traversal(&n.right));
	result
    } else {
        Vec::new()
    }
}

pub fn post_order_traversal(node: &Option<Box<TreeNode>>) -> Vec<i32> {
    if let Some(ref n) = node {
        let mut result = Vec::new();

	result.extend(pre_order_traversal(&n.right));
	result.extend(vec![n.val]);
	result.extend(pre_order_traversal(&n.left));
        result
    } else {
        Vec::new()
    }
}

pub fn in_order_traversal(node: &Option<Box<TreeNode>>) -> Vec<i32> {
    if let Some(ref n) = node {
	let mut result = Vec::new();

	result.extend(vec![n.val]);
	result.extend(pre_order_traversal(&n.left));
	result.extend(pre_order_traversal(&n.right));
        result
    } else {
        Vec::new()
    }
}


#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_sorted_array_to_bst -- --nocapture
    use super::*;

    #[test]
    pub fn test_sorted_array_to_bst() {
    	let nums = vec![-10, -3, 0, 5, 9];
 	let bst = sorted_array_to_bst(nums);
	let pst = pre_order_traversal(&bst);
	let exp = [-10, -3, 0, 5, 9];

	for ii in 0..pst.len() {
	    assert_eq!(pst[ii], exp[ii]);
	}
    }
}