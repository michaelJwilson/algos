struct Solution;

pub fn remove_element(array: &mut Vec<i32>, value: i32) -> i32 {
    if array.len() == 0 {
        return 0;
    }

    // NB usize
    let mut last_idx = array.len() - 1;

    for ii in 0..=array.len() -1 {
        if array[ii] == value {
            while last_idx >= ii {
                if array[last_idx] != value {
                    array[ii] = array[last_idx];
                    array[last_idx] = value;

                    last_idx -= 1;
                    break;
                }
                last_idx -= 1;
            }
        }
    }

    (last_idx + 1).try_into().unwrap()
}

impl Solution {
    pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
        remove_element(nums, val)       
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_remove_element_one -- --nocapture
    use super::*;

    #[test]
    pub fn test_remove_element() {
        let mut nums: Vec<i32> = vec![1, 2, 4, 3, 4, 5, 6, 4, 4, 7, 8];        
	let length: i32 = nums.len() as i32;

	println!("{:?}, {:?}", length, nums);

	let length: i32 = remove_element(&mut nums, 4);

	println!("{:?}, {:?}", length, nums);
    }
    
    #[test]
    pub fn test_remove_element_one() {
        let mut nums: Vec<i32> = vec![3, 2, 2, 3];
        let length: i32 = nums.len() as i32;

        println!("{:?}, {:?}", length, nums);

        let length: i32 = remove_element(&mut nums, 3);

        println!("{:?}, {:?}", length, nums);
    }
    
    #[test]
    pub fn test_remove_element_two() {
        let mut nums: Vec<i32> = vec![0, 1, 2, 2, 3, 0, 4, 2];
        let length: i32 = nums.len() as i32;

        println!("{:?}, {:?}", length, nums);

        let length: i32 = remove_element(&mut nums, 2);

        println!("{:?}, {:?}", length, nums);
    }

    #[test]
    pub fn test_remove_element_three() {
        let mut nums: Vec<i32> = vec![1];
        let length: i32 = nums.len() as i32;

        println!("{:?}, {:?}", length, nums);

        let length: i32 = remove_element(&mut nums, 1);

        println!("{:?}, {:?}", length, nums);
    }

    #[test]
    pub fn test_remove_element_four() {
        let mut nums: Vec<i32> = vec![3, 3];
        let length: i32 = nums.len() as i32;

        println!("{:?}, {:?}", length, nums);

        let length: i32 = remove_element(&mut nums, 3);

        println!("{:?}, {:?}", length, nums);
    }
}