struct Solution;

//  NB enforces borrowing rules at runtime, not compile time.
//     i.e. one mutable ref., or many immutable.
use std::cell::RefCell;
use std::rc::Rc;

//  NB Data-class like, i.e. memory allocation.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

//  NB consider inlining the annotated function. Inlining is
//     an optimization technique where the compiler replaces a
//     function call with the actual body of the function. This
//     can potentially reduce the overhead of the function call
//     and improve performance, especially for small, frequently
//     called functions.
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
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        match root {
            Some(node) => {
	    	//  NB calculate the depth of each child sub-tree
                let left_depth = Solution::max_depth(node.borrow().left.clone());
                let right_depth = Solution::max_depth(node.borrow().right.clone());
		
                1 + left_depth.max(right_depth)
            }
	    
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_binary_tree_max_depth -- --nocapture
    use super::*;

    #[test]
    fn test_ref_cell() {
       let cell = RefCell::new(5);

       // NB immutable borrow
       {
           let value = cell.borrow();
  	   println!("Value: {}", value);
       }

       // NB mutable borrow
       {
           let mut value = cell.borrow_mut();
           *value += 1;
       }

       // NB immutable borrow
       {
           let value = cell.borrow();
           println!("Updated Value: {}", value);
       }
    }

    #[test]
    fn test_rc() {
       // NB Rc (Reference Counted) is a type that provides shared ownership of a value.
       //    Multiple Rc instances can point to the same value, and the value is only
       //    dropped when the last Rc instance pointing to it is dropped.
       //
       //    Allows multiple owners of the same value. Each Rc instance keeps a reference count,
       //    which is incremented when a new Rc is cloned and decremented when an Rc is dropped.
       let rc1 = Rc::new(5);
       
       println!("Reference count: {}", Rc::strong_count(&rc1));

       {
	   let rc2 = Rc::clone(&rc1);
           println!("Reference count after clone: {}", Rc::strong_count(&rc1));
       }

       println!("Reference count after rc2 goes out of scope: {}", Rc::strong_count(&rc1));
    }

    #[test]
    fn test_binary_tree_max_depth() {
        let root = Rc::new(RefCell::new(TreeNode::new(3)));

	root.borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(9))));
        root.borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(20))));

	root.borrow_mut().right.as_ref().unwrap().borrow_mut().left = Some(Rc::new(RefCell::new(TreeNode::new(15))));
	root.borrow_mut().right.as_ref().unwrap().borrow_mut().right = Some(Rc::new(RefCell::new(TreeNode::new(7))));
    	
    	let result = Solution::max_depth(Some(root));
        let exp = 3;
	
        assert!(result == exp);
    }
}