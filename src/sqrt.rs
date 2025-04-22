struct Solution;

// NB binary search approach.
pub fn get_sqrt(input: i64) -> i64 {
    if input < 2 {
        return input;
    }

    let mut left: i64 = 1;
    let mut right: i64 = input / 2;
    let mut result: i64 = 0;

    while left <= right {
        let mid = left + (right - left) / 2;
        let mid_squared = mid * mid;

        if mid_squared == input {
            return mid;
        } else if mid_squared < input {
            left = mid + 1;
            result = mid;
        } else {
            right = mid - 1;
        }
    }

    result
}

impl Solution {
    pub fn my_sqrt(x: i32) -> i64 {
        get_sqrt(x as i64)
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_sqrt -- --nocapture
    use super::*;

    #[test]
    pub fn test_sqrt_4() {
        let result = get_sqrt(4);

        assert_eq!(result, 2);
    }

    #[test]
    pub fn test_sqrt_8() {
        let result = get_sqrt(8);

        assert_eq!(result, 2);
    }

    #[test]
    pub fn test_sqrt_digits() {
        let digits = vec![
            0,
            1,
            10,
            12,
            21,
            101,
            144,
            155,
            1_00,
            2_147_395_599,
            2_147_395_600,
        ];
        let exp: Vec<i64> = digits
            .iter()
            .map(|&xx| (xx as f64).sqrt().floor() as i64)
            .collect();

        for ii in 0..digits.len() {
            let result = Solution::my_sqrt(digits[ii]);

            assert_eq!(result, exp[ii]);
        }
    }
}
