struct Solution;

pub fn longest_common_prefix(strs: Vec<String>) -> String {
    if strs.is_empty() {
        return String::new();
    }

    // NB shortest length of any input string.
    let mut shortest_length: usize = usize::MAX;

    for s in &strs {
        shortest_length = shortest_length.min(s.len())
    }

    let mut prefix: String = String::new();

    // NB converts String to Vec<char>?
    let first_string: Vec<char> = strs[0].chars().collect();

    // NB iterate over first_string, stopping at shortest_length if necessary.
    for (ii, current_letter) in first_string.iter().take(shortest_length).enumerate() {
        // NB check whether all other strings support this character
        //    and edit accordingly.
        for ss in &strs[1..] {
            if ss.chars().nth(ii) != Some(*current_letter) {
                return prefix;
            }
        }

        prefix.push(*current_letter);
    }

    prefix
}

impl Solution {
    pub fn longest_common_prefix(strs: Vec<String>) -> String {
        longest_common_prefix(strs)
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_longest_common_prefix -- --nocapture
    use super::*;

    #[test]
    pub fn test_longest_common_prefix_one() {
        let strs: Vec<String> = vec![
            "flower".to_string(),
            "flow".to_string(),
            "flight".to_string(),
        ];
        let result: String = longest_common_prefix(strs);

        assert_eq!(result, "fl".to_string());
    }

    #[test]
    pub fn test_longest_common_prefix_noprefix() {
        let strs: Vec<String> = vec!["dog".to_string(), "racecar".to_string(), "car".to_string()];
        let result: String = longest_common_prefix(strs);

        assert_eq!(result, "".to_string());
    }

    #[test]
    pub fn test_longest_common_prefix_one_entry() {
        let strs: Vec<String> = vec!["dog".to_string()];
        let result: String = longest_common_prefix(strs);

        assert_eq!(result, "dog".to_string());
    }

    #[test]
    pub fn test_longest_common_prefix_two() {
        let strs: Vec<String> = vec!["cir".to_string(), "car".to_string()];
        let result: String = longest_common_prefix(strs);

        assert_eq!(result, "c".to_string());
    }
}
