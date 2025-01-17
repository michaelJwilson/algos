struct Solution;

pub fn get_needle_haystack(needle: &str, haystack: &str) -> i32 {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len == 0 {
        return 0;
    }

    if needle.len() > haystack.len() {
        return -1;
    }

    for ii in 0..=haystack_len - needle_len {
        if &haystack[ii..ii + needle_len] == needle {
            return ii as i32;
        }
    }

    -1
}

impl Solution {
    pub fn str_str(haystack: String, needle: String) -> i32 {
        get_needle_haystack(&needle, &haystack)       
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_needle_haystack -- --nocapture
    use super::*;

    #[test]
    pub fn test_needle_haystack() {
       let needle = "sad";
       let haystack = "sadbut sad";

       let index = get_needle_haystack(needle, haystack);

       assert_eq!(index, 0);
    }

    #[test]
    pub fn test_needle_haystack_invalid() {
       let needle = "leeto";
       let haystack = "leetcode";

       let index = get_needle_haystack(needle, haystack);

       assert_eq!(index, -1);
    }

    #[test]
    pub fn test_needle_haystack_tooshort() {
       let needle = "aaaa".to_string();
       let haystack = "aaa".to_string();

       let index = Solution::str_str(haystack, needle);
       assert_eq!(index, -1);
    }

    #[test]
    pub fn test_needle_haystack_solution() {
       let needle = "sad".to_string();
       let haystack = "sadbut sad".to_string();

       let index = Solution::str_str(haystack, needle);

       assert_eq!(index, 0);
    }
}