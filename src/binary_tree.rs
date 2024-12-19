use std::cmp::Ordering;
use std::sync::Arc;
use std::thread;
use rayon::prelude::*;

/// A node in the binary tree; accepts a generic type T as value
/// providing it has the Ordinal comparison trait defined.
#[derive(Debug)]
struct Node<T: Ord> {
    value: T,
    left: Subtree<T>,
    right: Subtree<T>,
}

impl<T: Ord> Node <T> {
    fn new(value: T) -> Self {
        Self {value, left: Subtree::new(), right: Subtree::new() }
    }
}

/// A possibly-empty (None) subtree; Box achieves a pointer definition to
/// the heap that circumvents the need to know tree size a priori.
#[derive(Debug)]
struct Subtree<T: Ord>(Option<Box<Node<T>>>);

impl<T: Ord> Subtree<T> {
    fn new() -> Self {
        Self(None)
    }

    fn insert(&mut self, value: T) {
        match &mut self.0 {
	    //  Some is the non-None instance to Option.
	    None => self.0 = Some(Box::new(Node::new(value))),
	    Some(n) => match value.cmp(&n.value) {
	        Ordering::Less => n.left.insert(value),
		Ordering::Equal => {},
		Ordering::Greater => n.right.insert(value),
	    }
	}
    }

    fn has(&self, value: &T) -> bool {
        match &self.0 {
	    None => false,
	    Some(n) => match value.cmp(&n.value) {
	        Ordering::Less => n.left.has(value),
		Ordering::Equal => true,
		Ordering::Greater => n.right.has(value),
	    }
        }
    }

    fn len(&self) -> usize {
       match &self.0 {
           None => 0,
	   Some(n) => 1 + n.left.len() + n.right.len(),
       }
    }
}

/// A container storing a set of values, using a binary tree.
///
/// If the same value is added multiple times, it is only stored once.
#[derive(Debug)]
pub struct BinaryTree<T: Ord> {
    root: Subtree<T>,
}

impl<T: Ord> BinaryTree<T> {
    fn new() -> Self {
        Self { root: Subtree::new() }
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
   let mut tree = BinaryTree::new();

   for i in 0..100 {
       tree.insert(i);
   }

   let tree = Arc::new(tree);

   let queries: Vec<i32> = (0..10).map(|i| i * 10 + 21).collect();
   let results: Vec<bool> = queries.par_iter()
        .map(|&value| {
            let tree_clone = Arc::clone(&tree);
            tree_clone.has(&value)
        })
        .collect();

    for (query, result) in queries.iter().zip(results.iter()) {
        println!("Query {} = {}", query, result);
    }
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
            let got: Vec<bool> =
                (0..exp.len()).map(|i| tree.has(&(i as i32))).collect();
		
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