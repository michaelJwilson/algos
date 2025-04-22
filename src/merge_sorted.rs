struct Solution;

fn merge_sorted<'a>(m: i32, nums1: &'a mut [i32], n: i32, nums2: &'a mut [i32]) {
    assert!(nums1.len() as i32 == (m + n));
    assert!(nums2.len() as i32 == n);

    // NB nothing to do.
    if n == 0 {
        return;
    }

    let mut jj = 0;

    for (ii, val) in nums1.iter_mut().enumerate() {
        // NB we march along nums1 until we find an element > that first in num2;
        if (*val > nums2[jj]) && ((ii as i32) < m) {
            let mut to_swap = *val;

            // nums1[ii] = nums2[jj];
            // nums2[jj] = to_swap;

            // NB we swap the elements between the array and fix up the nums2 order;
            std::mem::swap(&mut *val, &mut nums2[jj]);
            std::mem::swap(&mut nums2[jj], &mut to_swap);

            for kk in 0..nums2.len() - 1 {
                if nums2[kk + 1] < nums2[kk] {
                    let mut to_swap = nums2[kk];

                    // nums2[kk] = nums2[kk + 1];
                    // nums2[kk + 1] = to_swap;

                    nums2.swap(kk, kk + 1);

                    std::mem::swap(&mut nums2[kk + 1], &mut to_swap);
                }
            }
        }

        // NB swap in a zero-pad.
        if ii as i32 >= m {
            let mut zero = *val;

            // nums1[ii] = nums2[jj];
            // nums2[jj] = zero;

            std::mem::swap(&mut *val, &mut nums2[jj]);
            std::mem::swap(&mut nums2[jj], &mut zero);

            jj += 1;
        }
    }

    // println!("{:?}, {:?}, {:?} \t {:?}", nums1.len(), jj, nums1, nums2);
}

fn merge_sorted_optimal(m: i32, nums1: &mut [i32], n: i32, nums2: &[i32]) {
    //  Use a two-pointer approach to merge the two sorted arrays in-place.
    //  The idea is to start from the end of both arrays and place the largest
    //  elements at the end of nums1. This way, you avoid overwriting elements
    //  in nums1 that have not been processed yet.
    //
    //  NB last index in nums1 that's not zero padded;
    let mut i = m as isize - 1;

    // NB last index in num2;
    let mut j = n as isize - 1;

    // NB last index in	nums1;
    let mut k = (m + n) as isize - 1;

    // NB working back through the solution array, a given element either came from
    //    the last decrement to nums1 or nums2.  Update accordingly.
    while j >= 0 {
        if i >= 0 && nums1[i as usize] > nums2[j as usize] {
            nums1[k as usize] = nums1[i as usize];
            i -= 1;
        } else {
            nums1[k as usize] = nums2[j as usize];
            j -= 1;
        }
        k -= 1;
    }
}

impl Solution {
    pub fn merge(nums1: &mut [i32], m: i32, nums2: &mut [i32], n: i32) {
        merge_sorted_optimal(m, nums1, n, nums2);
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_merge_sorted_two -- --nocapture
    use super::*;

    #[test]
    fn test_merge_sorted_one() {
        let m: i32 = 3;
        let n: i32 = 3;

        let mut nums1 = vec![1, 2, 3, 0, 0, 0];
        let mut nums2 = vec![2, 5, 6];

        let exp = vec![1, 2, 2, 3, 5, 6];

        merge_sorted(m, &mut nums1, n, &mut nums2);

        for ii in 0..nums1.len() {
            assert!(nums1[ii] == exp[ii]);
        }
    }

    #[test]
    fn test_merge_sorted_two() {
        // NB tests when all elements in second array < all in first array.
        let m: i32 = 3;
        let n: i32 = 3;

        let mut nums1 = vec![4, 5, 6, 0, 0, 0];
        let mut nums2 = vec![1, 2, 3];

        let exp = vec![1, 2, 3, 4, 5, 6];

        merge_sorted(m, &mut nums1, n, &mut nums2);

        for ii in 0..nums1.len() {
            assert!(nums1[ii] == exp[ii]);
        }
    }

    #[test]
    fn test_merge_sorted_three() {
        // NB tests non-padding zeros in nums1.
        let m: i32 = 55;
        let n: i32 = 99;

        // NB tests non-padding zeros in nums1.
        // let mut nums1 = vec![-10,-9,-9,-4,-4,-3,-3,-2,-2,-1,-1,0,0,1,1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        // let mut nums2 = vec![-10,-8,-7,-2,-2,-1,-1,-1,0,0,0,0,1,2,3];

        let mut nums1 = vec![
            -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let mut nums2 = vec![-2, -1, -1, -1, 0, 0, 0, 0, 1, 2];

        let n: i32 = nums2.len() as i32;
        let m: i32 = nums1.len() as i32 - n;

        merge_sorted(m, &mut nums1, n, &mut nums2);
    }

    #[test]
    fn test_merge_sorted_four() {
        let m: i32 = 55;
        let n: i32 = 99;

        // NB tests non-padding zeros in nums1.
        let mut nums1 = vec![
            -10, -10, -9, -9, -9, -8, -8, -7, -7, -7, -6, -6, -6, -6, -6, -6, -6, -5, -5, -5, -4,
            -4, -4, -3, -3, -2, -2, -1, -1, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6, 6, 7,
            7, 7, 7, 8, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let mut nums2 = vec![
            -10, -10, -9, -9, -9, -9, -8, -8, -8, -8, -8, -7, -7, -7, -7, -7, -7, -7, -7, -6, -6,
            -6, -6, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2,
            -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4,
            4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9,
        ];

        let exp = vec![
            -10, -10, -10, -10, -9, -9, -9, -9, -9, -9, -9, -8, -8, -8, -8, -8, -8, -8, -7, -7, -7,
            -7, -7, -7, -7, -7, -7, -7, -7, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -5, -5, -5,
            -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -2, -2, -2, -2,
            -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
            5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9,
            9, 9, 9, 9, 9,
        ];

        merge_sorted(m, &mut nums1, n, &mut nums2);

        for ii in 0..nums1.len() {
            assert!(nums1[ii] == exp[ii]);
        }
    }

    #[test]
    fn test_empty_second_merge() {
        let m: i32 = 1;
        let n: i32 = 0;

        let mut nums1 = vec![1];
        let mut nums2 = vec![];

        let exp = vec![1];

        merge_sorted(m, &mut nums1, n, &mut nums2);

        for ii in 0..n as usize {
            assert!(nums1[ii] == exp[ii]);
        }
    }

    #[test]
    fn test_empty_first_merge() {
        let m: i32 = 0;
        let n: i32 = 1;

        // NB padded only.
        let mut nums1 = vec![0];
        let mut nums2 = vec![1];

        let exp = vec![1];

        merge_sorted(m, &mut nums1, n, &mut nums2);

        for ii in 0..n as usize {
            assert!(nums1[ii] == exp[ii]);
        }
    }

    #[test]
    fn test_merge_sorted_solution() {
        let m: i32 = 3;
        let n: i32 = 3;

        let mut nums1 = vec![1, 2, 3, 0, 0, 0];
        let mut nums2 = vec![2, 5, 6];

        let exp = vec![1, 2, 2, 3, 5, 6];

        Solution::merge(&mut nums1, m, &mut nums2, n);

        for ii in 0..nums1.len() as usize {
            assert!(nums1[ii] == exp[ii]);
        }
    }
}
