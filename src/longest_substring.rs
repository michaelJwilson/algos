use std::collections::HashSet;

fn get_longest_substring(input_str: &str) -> usize {
    let mut chars = input_str.chars();
    let mut seen = HashSet::new();
    let mut max_len = 0;
    let mut start = 0;

    for (end, char) in input_str.chars().enumerate() {
        while seen.contains(&char) {
            seen.remove(&input_str[start..].chars().next().unwrap());
            start += 1;
        }
        seen.insert(char);
        max_len = max_len.max(end - start + 1);
    }

    max_len
}

struct Solution;

impl Solution {
    pub fn length_of_longest_substring(s: String) -> i32 {
        get_longest_substring(&s) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(get_longest_substring(""), 0);
    }

    #[test]
    fn test_abcabcbb() {
        assert_eq!(get_longest_substring("abcabcbb"), 3);
    }

    #[test]
    fn test_bbbbb() {
        assert_eq!(get_longest_substring("bbbbb"), 1);
    }

    #[test]
    fn test_pwwkew() {
        assert_eq!(get_longest_substring("pwwkew"), 3);
    }

    #[test]
    fn test_Solution() {
        assert_eq!(
            Solution::length_of_longest_substring("abcabcbb".to_string()),
            3
        );
    }
}
