///  Cannot borrow immutable and mutable references to a simultaneously.
///
///
fn main() {
    let mut a: i32 = 10;

    // let b: &i32 = &a;

    {
        let c: &mut i32 = &mut a;
        *c = 20;
    }

    println!("a: {a}");
    // println!("b: {b}");
}