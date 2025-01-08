struct Solution;

fn merge_sorted<'a>(m: i32, nums1: &'a mut Vec<i32>, n: i32, nums2: &'a mut Vec<i32>) {
  assert!(nums1.len() as i32 == (m + n));
  assert!(nums2.len() as i32 == n);

  // NB nothing to do.
  if n == 0 {
      return;
  }

  let mut jj = 0;

  for ii in 0..nums1.len() {
      #[cfg(debug_statements)]
      println!("{:?}, {:?}, {:?} \t {:?}", ii, jj, nums1, nums2);

      // NB we march along nums1 until we find an element > that first in num2;
      if (nums1[ii] > nums2[jj]) && ((ii as i32) < m) {
          let to_swap = nums1[ii];

          nums1[ii] = nums2[jj];

	  // NB we swap the elements between the array and fix up the nums2 order;
	  nums2[jj] = to_swap;

	  for kk in 0..nums2.len() - 1 {
	      if nums2[kk + 1] < nums2[kk] {
	          let to_swap = nums2[kk];

		  nums2[kk] = nums2[kk + 1];
		  nums2[kk + 1] = to_swap;
	      }
	  }
      }

      // NB swap in a zero-pad.
      if ii as i32 >= m {
          let zero = nums1[ii];

          nums1[ii] = nums2[jj];
          nums2[jj] = zero;

	  jj += 1;
      }
  }

  #[cfg(debug_statements)]
  println!("{:?}, {:?}, {:?} \t {:?}", nums1.len(), jj, nums1, nums2);
}

impl Solution {
    pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
        merge_sorted(m, nums1, n, nums2);       
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

	let mut nums1 = vec![1,2,3,0,0,0];
	let mut nums2 = vec![2,5,6];

	let exp = vec![1,2,2,3,5,6];

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
       
       let mut nums1 = vec![4,5,6,0,0,0];
       let mut nums2 = vec![1,2,3];

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

       let mut nums1 = vec![-2,-2,-1,-1,0,0,1,1,2,2,0,0,0,0,0,0,0,0,0,0];
       let mut nums2 = vec![-2,-1,-1,-1,0,0,0,0,1,2];

       let n: i32 = nums2.len() as i32;
       let m: i32 = nums1.len() as i32 - n;
      
       merge_sorted(m, &mut nums1, n, &mut nums2);
    }


    #[test]
    fn test_merge_sorted_four() {
       let m: i32 = 55;
       let n: i32 = 99;

       // NB tests non-padding zeros in nums1.
       let mut nums1 = vec![-10,-10,-9,-9,-9,-8,-8,-7,-7,-7,-6,-6,-6,-6,-6,-6,-6,-5,-5,-5,-4,-4,-4,-3,-3,-2,-2,-1,-1,0,1,1,1,2,2,2,3,3,3,4,5,5,6,6,6,6,7,7,7,7,8,9,9,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

       let mut nums2 = vec![-10,-10,-9,-9,-9,-9,-8,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-4,-4,-4,-3,-3,-3,-2,-2,-2,-2,-2,-2,-2,-1,-1,-1,0,0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,9,9,9,9];

       let exp = vec![-10,-10,-10,-10,-9,-9,-9,-9,-9,-9,-9,-8,-8,-8,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-5,-5,-5,-4,-4,-4,-4,-4,-4,-4,-4,-3,-3,-3,-3,-3,-2,-2,-2,-2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,9,9,9,9,9,9,9,9];

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

        let mut nums1 = vec![1,2,3,0,0,0];
        let mut nums2 = vec![2,5,6];

        let exp = vec![1,2,2,3,5,6];

	Solution::merge( &mut nums1, m, &mut nums2, n );

        for ii in 0..nums1.len() as usize {
            assert!(nums1[ii] == exp[ii]);
        }
    }
}