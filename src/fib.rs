fn fib(n: u32) -> u32 {
    // NB next number is the sum of the previous two.
    //    starts 0, 1, 1, 2, 3, 5, 8, ...
    //
    //    
    if n < 2 {
        return n;
    } else {
        return fib(n - 1) + fib(n - 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fib() {
       let n = 20;
       
       println!("fib({n}) = {}", fib(n));
    }
}