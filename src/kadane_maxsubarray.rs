struct Solution;

// NB max implies most positive, sub-array implies continuous.
impl Solution {
    pub fn max_sub_array(nums: &[i32]) -> i32 {
        if nums.is_empty() {
            return 0;
        }

        // NB assignment of i32.
        let mut max_current = nums[0];
        let mut max_global = nums[0];

        for &num in nums.iter().skip(1) {
            // NB dynamic/greedy, choose the best solution for the
            //    sub-array to this point.
            max_current = std::cmp::max(num, max_current + num);

            if max_current > max_global {
                max_global = max_current;
            }
        }

        max_global
    }
}

#[cfg(test)]
mod tests {
    // cargo test kadane_maxsubarray -- --nocapture
    use super::*;

    #[test]
    fn test_max_sub_array() {
        let nums = vec![-2, 1, -3, 4, -1, 2, 1, -5, 4];
        let result = Solution::max_sub_array(&nums);

        // NB the subarray [4, -1, 2, 1] has the largest sum 6.
        assert_eq!(result, 6);
    }
}
