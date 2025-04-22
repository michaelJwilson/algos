struct Solution;

pub fn get_plus_one(input: Vec<i32>, idx: usize) -> Vec<i32> {
    let mut result = input.clone();

    if input[idx] < 9 {
        result[idx] = input[idx] + 1;
        result
    } else {
        result[idx] = 0;

        if idx == 0 {
            let mut interim = vec![1];
            interim.extend(result);

            interim
        } else {
            get_plus_one(result, idx - 1)
        }
    }
}

impl Solution {
    pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
        let last_idx: usize = digits.len() - 1;
        get_plus_one(digits, last_idx)
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_plus_one -- --nocapture
    use super::*;

    #[test]
    pub fn test_plus_one_1() {
        let digits = vec![1];
        let exp: Vec<i32> = vec![2];

        let last_idx = digits.len() - 1;
        let result = get_plus_one(digits, last_idx);

        for ii in 0..result.len() {
            assert_eq!(result[ii], exp[ii]);
        }
    }

    #[test]
    pub fn test_plus_one_12() {
        let digits = vec![1, 2];
        let exp: Vec<i32> = vec![1, 3];

        let last_idx = digits.len() - 1;
        let result = get_plus_one(digits, last_idx);
        /*
        for ii in 0..result.len() {
            assert_eq!(result[ii], exp[ii]);
        }*/

        // println!("{:?}", result);
    }

    #[test]
    pub fn test_plus_one_9() {
        let digits = vec![9];
        let exp: Vec<i32> = vec![1, 0];

        let last_idx = digits.len() - 1;
        let result = get_plus_one(digits, last_idx);

        for ii in 0..result.len() {
            assert_eq!(result[ii], exp[ii]);
        }

        // print!("{:?}", result);
    }

    #[test]
    pub fn test_plus_one_123() {
        let digits: Vec<i32> = vec![1, 2, 3];
        let exp: Vec<i32> = vec![1, 2, 4];

        let last_idx = digits.len() - 1;
        let result = get_plus_one(digits, last_idx);

        for ii in 0..result.len() {
            assert_eq!(result[ii], exp[ii]);
        }

        print!("{:?}", result);
    }

    #[test]
    pub fn test_plus_one_4321() {
        let digits = vec![4, 3, 2, 1];
        let exp: Vec<i32> = vec![4, 3, 2, 2];

        let last_idx = digits.len() - 1;
        let result = get_plus_one(digits, last_idx);

        for ii in 0..result.len() {
            assert_eq!(result[ii], exp[ii]);
        }

        // print!("{:?}", result);
    }

    #[test]
    pub fn test_plus_one_8999() {
        let digits = vec![8, 9, 9, 9];
        let exp: Vec<i32> = vec![9, 0, 0, 0];

        let last_idx = digits.len() - 1;
        let result = get_plus_one(digits, last_idx);

        print!("{:?}", result);
        /*
            for ii in 0..result.len() {
                assert_eq!(result[ii], exp[ii]);
            }
        */
    }

    #[test]
    pub fn test_plus_one_solution() {
        let digits = vec![1, 2, 3];
        let exp = vec![1, 2, 4];

        let result = Solution::plus_one(digits);

        for ii in 0..result.len() {
            assert_eq!(result[ii], exp[ii]);
        }
    }
}
