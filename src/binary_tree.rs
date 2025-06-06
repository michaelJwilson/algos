use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

/// NB a node in the binary tree; accepts a generic type T as value
/// providing it has the partial order / Ordinal comparison trait defined.
#[derive(Debug)]
struct Node<T: Ord> {
    value: T,
    left: Subtree<T>,
    right: Subtree<T>,
}

// NB accepts a generic type T for which the Ordinal comparison trait is
//    defined.
impl<T: Ord> Node<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            left: Subtree::new(),
            right: Subtree::new(),
        }
    }
}

// NB Box achieves a pointer definition to memory on the heap that
//    circumvents the need to know tree size a priori, as required
//    by a stack allocation.
//
//    Subtree initialized with a node or None.  Assumes binary tree.
#[derive(Debug)]
struct Subtree<T: Ord>(Option<Box<Node<T>>>);

impl<T: Ord> Subtree<T> {
    fn new() -> Self {
        Self(None)
    }

    // NB only saves unique values to the tree.
    fn insert(&mut self, value: T) {
        match &mut self.0 {
            //  NB some is the non-None option for Option.
            None => self.0 = Some(Box::new(Node::new(value))),
            Some(n) => match value.cmp(&n.value) {
                Ordering::Less => n.left.insert(value),
                Ordering::Equal => {}
                Ordering::Greater => n.right.insert(value),
            },
        }
    }

    fn has(&self, value: &T) -> bool {
        // NB read-only reference.
        match &self.0 {
            None => false,
            Some(n) => match value.cmp(&n.value) {
                Ordering::Less => n.left.has(value),
                Ordering::Equal => true,
                Ordering::Greater => n.right.has(value),
            },
        }
    }

    // NB returns number of nodes in the tree
    fn len(&self) -> usize {
        match &self.0 {
            None => 0,
            Some(n) => 1 + n.left.len() + n.right.len(),
        }
    }
}

// NB A container storing a set of values, using a binary tree.
//    If the same value is added multiple times, it is only stored once.
#[derive(Debug)]
pub struct BinaryTree<T: Ord> {
    root: Subtree<T>,
}

// TODO all methods delegate to the equivalent in Subtree?  deprecate.
impl<T: Ord> BinaryTree<T> {
    fn new() -> Self {
        Self {
            root: Subtree::new(),
        }
    }

    fn insert(&mut self, value: T) {
        self.root.insert(value);
    }

    fn has(&self, value: &T) -> bool {
        self.root.has(value)
    }

    fn len(&self) -> usize {
        self.root.len()
    }
}

pub fn query_binary_tree() {
    // NB initialize walltime timer.
    let start = Instant::now();

    let mut tree = BinaryTree::new();

    let step: i32 = 5;
    let num_element: i32 = 1_000;

    // NB 200 elements with 0, 5, ..., etc. as unsigned ints
    //    with memory determined by 32/64 bit.
    for i in (0..num_element).step_by(step as usize) {
        tree.insert(i);
    }

    let duration = start.elapsed();

    println!("Tree construction complete in {duration:?}.");

    // NB query the tree num_queries times with multiple threads.
    let num_queries: i32 = 1_000_000;
    let queries: Vec<i32> = (0..num_queries).map(|xx| xx % num_element).collect();
    let tree = Arc::new(tree);

    let start = Instant::now();

    let results: Vec<i32> = queries
        .par_iter()
        .map(|&value| Arc::clone(&tree).has(&value) as i32)
        .collect();

    // NB sum of binary "has" result is number of queries present.
    let total_queries_found: i32 = results.iter().sum();

    let duration = start.elapsed();
    let expected = (num_element / step) * (num_queries / num_element);

    println!("Found {total_queries_found} of {num_queries:?} queries to be present in {duration:?}.  Expected {expected}.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len() {
        let mut tree = BinaryTree::new();
        assert_eq!(tree.len(), 0);

        tree.insert(2);
        assert_eq!(tree.len(), 1);

        tree.insert(1);
        assert_eq!(tree.len(), 2);

        tree.insert(2); // not a unique item
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn has_str() {
        let mut tree = BinaryTree::new();
        tree.insert("foo");

        // NB a macro (with !).
        assert_eq!(tree.len(), 1);

        tree.insert("bar");

        // NB passing foo by reference retains ownership to main.
        assert!(tree.has(&"foo"))
    }

    #[test]
    fn has_i32() {
        let mut tree = BinaryTree::new();

        fn check_has(tree: &BinaryTree<i32>, exp: &[bool]) {
            let got: Vec<bool> = (0..exp.len()).map(|i| tree.has(&(i as i32))).collect();

            assert_eq!(&got, exp);
        }

        check_has(&tree, &[false, false, false, false, false]);
        tree.insert(0);

        check_has(&tree, &[true, false, false, false, false]);
        tree.insert(4);

        check_has(&tree, &[true, false, false, false, true]);
        tree.insert(4);

        check_has(&tree, &[true, false, false, false, true]);
        tree.insert(3);

        check_has(&tree, &[true, false, false, true, true]);
    }

    #[test]
    fn unbalanced() {
        let mut tree = BinaryTree::new();

        for i in 0..100 {
            tree.insert(i);
        }

        assert_eq!(tree.len(), 100);
        assert!(tree.has(&50));
    }
}
