struct Solution;

pub fn get_swap_idx(
    array: Vec<i32>,
    current_idx: usize,
    last_idx: usize,
    value: i32,
) -> Option<usize> {
    //  NB
    let mut new_last_idx = last_idx;

    while new_last_idx > current_idx {
        if array[new_last_idx] != value {
            return Some(new_last_idx);
        } else {
            new_last_idx -= 1;
        }
    }

    // NB here, last_idx = current_idx;
    None
}

pub fn remove_element(array: &mut [i32], value: i32) -> i32 {
    if array.is_empty() {
        return 0;
    }

    // NB usize
    let mut last_idx = array.len() - 1;

    for ii in 0..=array.len() - 1 {
        if array[ii] == value {
            match get_swap_idx(array.to_vec(), ii, last_idx, value) {
                Some(swap_idx) => {
                    array[ii] = array[swap_idx];

                    // DEPRECATE
                    array[swap_idx] = value;

                    last_idx = swap_idx - 1;
                }

                // NB no valid swaps remaining.
                None => {
                    return ii.try_into().unwrap();
                }
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
        let length: i32 = remove_element(&mut nums, 4);

        assert_eq!(length, 7);
        assert_eq!(nums, vec![1, 2, 8, 3, 7, 5, 6, 4, 4, 4, 4]);
    }

    #[test]
    pub fn test_remove_element_get_swap_idx() {
        let nums: Vec<i32> = vec![1, 4, 3, 4, 6, 4, 4, 8, 4];
        let length: i32 = nums.len() as i32;

        let swap_idx = get_swap_idx(nums, 2, 8, 4).unwrap();

        assert_eq!(swap_idx, 7);
    }

    #[test]
    pub fn test_remove_element_one() {
        let mut nums: Vec<i32> = vec![3, 2, 2, 3];
        let length: i32 = remove_element(&mut nums, 3);

        assert_eq!(length, 2);
        assert_eq!(nums, vec![2, 2, 3, 3]);
    }

    #[test]
    pub fn test_remove_element_two() {
        let mut nums: Vec<i32> = vec![0, 1, 2, 2, 3, 0, 4, 2];
        let length: i32 = remove_element(&mut nums, 2);

        assert_eq!(length, 5);
        assert_eq!(nums, vec![0, 1, 4, 0, 3, 2, 2, 2]);
    }

    #[test]
    pub fn test_remove_element_three() {
        let mut nums: Vec<i32> = vec![1];
        let length: i32 = remove_element(&mut nums, 1);

        assert_eq!(length, 0);
    }

    #[test]
    pub fn test_remove_element_four_swap_idx() {
        let mut nums: Vec<i32> = vec![3, 3];
        let swap_idx = get_swap_idx(nums, 0, 1, 3);

        assert_eq!(swap_idx, None);
    }

    #[test]
    pub fn test_remove_element_four() {
        let mut nums: Vec<i32> = vec![3, 3];
        let length: i32 = remove_element(&mut nums, 3);

        assert_eq!(length, 0);
    }
}
