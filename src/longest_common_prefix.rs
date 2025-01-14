struct Solution;

pub fn longest_common_prefix(strs: Vec<String>) -> String {
    if strs.is_empty() {
        return String::new();
    }

    let mut shortest_length: usize = usize::MAX;

    for s in &strs {
    	shortest_length = shortest_length.min(s.len())
    }

    // println!("{:?}", shortest_length);

    let mut prefix: String = String::new();
    let first_string: Vec<char> = strs[0].chars().collect();

    for ii in 0..shortest_length {
    	let current_letter = first_string[ii];
        let mut to_append = true;
	
	// NB if only one string is provided, this will silently skip, as desired.
        for ss in &strs[1..] {
	    if ss.chars().nth(ii) != Some(current_letter) {
               to_append = false;
	       break;
	    }
        }

	// NB pre-pends, i.e. reverse order to what we would like.
        if to_append {
	    prefix.push(current_letter);
        } else {
	    break;
	}
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
        let strs: Vec<String> = vec!["flower".to_string(), "flow".to_string(), "flight".to_string()];
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