struct Solution;

//
// Climbing a staircase of n steps, how many
// ways can you climb to the top if you can
// climb either 1 or 2 steps.
//
// NB dynamic programming.  Fibonacci like.
//
pub fn init_num_states() -> Vec<usize> {
    // NB num_states for num_stairs = {0, 1, 2};
    let num_states: Vec<usize> = vec![0, 1, 2];

    num_states
}

pub fn climb_stairs(num_stairs: usize, num_states: &mut Vec<usize>) -> usize {
    if num_stairs < num_states.len() {
        return num_states[num_stairs];
    }

    // NB we always push in num_stairs order with num_stairs -2 first.
    let interim =
        climb_stairs(num_stairs - 2, num_states) + climb_stairs(num_stairs - 1, num_states);

    num_states.push(interim);

    interim
}

impl Solution {
    pub fn climb_stairs(n: i32) -> i32 {
        // NB derived from i32 to usize conversion.
        if n < 0 {
            return 0;
        }

        let mut num_states = init_num_states();

        climb_stairs(n as usize, &mut num_states) as i32
    }
}

#[cfg(test)]
mod tests {
    // cargo test test_climb_stairs -- --nocapture
    use super::*;

    #[test]
    pub fn test_climb_stairs() {
        let mut num_states = init_num_states();
        let num_stairs = 3;

        let result = climb_stairs(num_stairs, &mut num_states);
        let exp = 3;

        assert_eq!(result, exp);
    }
}
