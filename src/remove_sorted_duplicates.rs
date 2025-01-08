struct Solution;

fn remove_sorted_duplicates(nums1: &mut Vec<i32>) -> i32{
   let mut this_unique_index = nums1.len() - 1;
   let mut this_unique = nums1[this_unique_index];

   let mut num_unique_elements: i32 = 1;

   let mut ii: i32 = nums1.len() as i32 - 1;

   while ii >= 0 {
       #[cfg(debug_statements)]
       println!("{:?} {:?} {:?}", nums1, this_unique_index, this_unique);

       if nums1[ii as usize] < this_unique {
	   for jj in 0..num_unique_elements as usize{
	       nums1[ii as usize + jj + 1] = nums1[this_unique_index + jj];
	   }

	   this_unique_index = ii as usize;
	   this_unique = nums1[this_unique_index];

	   num_unique_elements += 1;
       }

       ii -= 1;
   }

   for ii in 0..num_unique_elements as usize {
       nums1[ii] = nums1[this_unique_index + ii];
   }

   #[cfg(debug_statements)]
   println!("{:?} {:?}", nums1, num_unique_elements);

   num_unique_elements
}

impl Solution {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        remove_sorted_duplicates(nums)
    }
}


#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_remove_sorted_duplicates -- --nocapture
    use super::*;

    #[test]
    fn test_remove_sorted_duplicates_one() {
	let mut nums1 = vec![0,0,1,1,1,2,2,3,3,4];
	let exp = 5;

	let result = remove_sorted_duplicates(&mut nums1);

	assert!(result == exp);
    }
    
    #[test]
    fn test_remove_sorted_duplicates_two() {
        let mut nums1 = vec![1,1,2];
        let exp = 2;

        let result = remove_sorted_duplicates(&mut nums1);

        assert!(result == exp);
    }
}