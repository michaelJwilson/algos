use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

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
            val,
            left: None,
            right: None,
        }
    }
}

/// Preamble:
///     Use a queue for breadth-first search: Instead of using recursion, we can use an
///     iterative approach with a queue to perform breadth-first search. This avoids the
///     overhead of recursive function calls and stack usage.
///
///     A double-ended queue implemented with a growable ring buffer.
///     The "default" usage of this type as a queue is to use `push_back` to add to the queue,
///     and `pop_front` to remove from the queue. `extend` and `append` push onto the back in this
///     manner, and iterating over `VecDeque` goes front to back.
///
/// See `std::collections::VecDeque`:
///     <https://doc.rust-lang.org/std/collections/struct.VecDeque.html>
impl Solution {
    pub fn average_of_levels(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<f64> {
        // NB if passed an empty tree, do nothing.
        if root.is_none() {
            return vec![];
        }

        let mut result: Vec<f64> = Vec::new();
        let mut num_nodes: Vec<i32> = Vec::new();

        // NB queue of (node, level) pairs.
        let mut queue: VecDeque<(Rc<RefCell<TreeNode>>, usize)> = VecDeque::new();

        // NB put root onto the queue.
        queue.push_back((root.unwrap(), 0));

        // NB process the node.
        while let Some((node, level)) = queue.pop_front() {
            let node = node.borrow();

            if level >= result.len() {
                result.push(0.0);
                num_nodes.push(0);
            }

            result[level] += node.val as f64;
            num_nodes[level] += 1;

            // NB place the (up to two) child nodes into the queue.
            if let Some(left) = node.left.clone() {
                queue.push_back((left, level + 1));
            }

            if let Some(right) = node.right.clone() {
                queue.push_back((right, level + 1));
            }
        }

        result
            .iter()
            .zip(num_nodes.iter())
            .map(|(&sum, &count)| sum / count as f64)
            .collect()
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

        root.borrow_mut().right.as_ref().unwrap().borrow_mut().left =
            Some(Rc::new(RefCell::new(TreeNode::new(15))));
        root.borrow_mut().right.as_ref().unwrap().borrow_mut().right =
            Some(Rc::new(RefCell::new(TreeNode::new(7))));

        let level_averages = Solution::average_of_levels(Some(root));

        for ii in 0..level_averages.len() {
            println!("{:?} \t {:?}", level_averages[ii], exp[ii]);

            assert_eq!(level_averages[ii], exp[ii]);
        }
    }
}
