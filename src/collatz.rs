/// NB  Determine the length of the collatz sequence beginning at `n`.
///     See: https://en.wikipedia.org/wiki/Collatz_conjecture
#[inline]
pub fn collatz(n: u32) -> u32 {
    if n == 0 || n == 1 {
        n
    } else if n % 2 == 0 {
        n >> 1
    } else {
        3 * n + 1
    }
}

fn collatz_length(mut n: u32) -> u32 {
    let max_len = 100;
    let mut len = 1;

    // println!("{n}\t{len}");

    while n != 1 {
        // NB a Collatz "step".
        n = collatz(n);
        len += 1;

        if len > max_len {
            return 0;
        }

        // println!("{n}\t{len}");
    }

    // NB returns sequence length for a given n.
    len
}

#[test]
fn test_collatz_length() {
    //  cargo test collatz -- --nocapture
    assert_eq!(collatz_length(11), 15);
}
