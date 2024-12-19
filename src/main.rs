fn main() {
    let root = TreeNode::new(1);
    let left_child = TreeNode::new(2);
    root.borrow_mut().left = Some(left_child);

    if let Some(ref left) = root.borrow().left {
        println!("Left child value: {}", left.borrow().value);
    } else {
        println!("No left child");
    }
}