struct Solution;

use std::collections::HashMap;

fn majority_element(nums: &[i32]) -> i32 {
    let num_elements = nums.len();
    let mut result = HashMap::new();

    for &num in nums.iter() {
        // NB a reference to the value for key num
        let count = result.entry(num).or_insert(0);

        *count += 1;

        if *count > num_elements / 2 {
            return num;
        }
    }

    // println!("{:?}", result);

    let mut max_key: i32 = -1;
    let mut max_value: i32 = -1;

    for (&key, &value) in result.iter() {
        if value as i32 > max_value {
            max_key = key;
            max_value = value as i32;
        }
    }

    max_key
}

impl Solution {
    pub fn majority_element(nums: Vec<i32>) -> i32 {
        majority_element(&nums)
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_majority_element -- --nocapture
    use super::*;

    #[test]
    fn test_majority_element_one() {
        let nums = vec![3, 2, 3];
        let exp = 3;

        let result = Solution::majority_element(nums);

        assert!(result == exp);
    }

    #[test]
    fn test_majority_element_two() {
        let nums = vec![2, 2, 1, 1, 1, 2, 2];
        let exp = 2;

        let result = Solution::majority_element(nums);

        assert!(result == exp);
    }
}
