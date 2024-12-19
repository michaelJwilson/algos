/// Determine the length of the collatz sequence beginning at `n`.
fn collatz(n: u32) -> u32 {
  if n == 0 {
      return 0;
  }
  if n == 1 {
      return 1;
  }
  else if n % 2 == 0 {
      return n / 2;
  }
  else {
      return 3 * n + 1;
  }
}


fn collatz_length(mut n: u32) -> u32 {
    let MAX_LEN = 100;
    let mut len = 1; 

    while n != 1 {
        println!("{n}");
    
        n = collatz(n);
        len += 1;
        
        if len > MAX_LEN {
            break;
        }
    }
    
    len
}

#[test]
fn test_collatz_length() {
    assert_eq!(collatz_length(11), 15);
}

fn main() {
    println!("Length: {}", collatz_length(11));
}