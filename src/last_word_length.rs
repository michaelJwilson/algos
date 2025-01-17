struct Solution;

pub fn last_word_length_naive(sentence: &str) -> i32 {
    let mut result: i32 = 0;

    // NB trim_end provides a view without changing memory/storage.
    for ss in sentence.trim_end().chars().rev() {
    	if ss.is_whitespace() {
	    return result;
	} else {
	    result += 1;
	}
    }

    // NB no white space in string, e.g. single word.
    result
}

pub fn last_word_length(sentence: &str) -> i32 {
    // TODO how do this handle "    "?
    sentence
	.split_whitespace()
	.last()
	.map_or(0, |word| word.len() as i32)
}

impl Solution {
    pub fn length_of_last_word(s: String) -> i32 {
        last_word_length(&s) 	   
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_last_word_length -- --nocapture
    use super::*;

    #[test]
    pub fn test_last_word_length_empty() {
        let result = Solution::length_of_last_word("      ".to_string());
    }

    #[test]
    pub fn test_last_word_length_hello_world() {
        let result = Solution::length_of_last_word("hello world".to_string());
	assert_eq!(result, 5);
    }

    #[test]
    pub fn test_last_word_length_hello() {
        let result = Solution::length_of_last_word("hello".to_string());
	assert_eq!(result, 5);
    }

    #[test]
    pub fn test_last_word_length_fly_me() {
        let result = Solution::length_of_last_word("    fly me   to  the moon  ".to_string());
	assert_eq!(result, 4);
    }

    #[test]
    pub fn test_last_word_length_luffy() {
        let result = Solution::length_of_last_word("luffy is still joyboy".to_string());
    	assert_eq!(result, 6);
    }

    #[test]
    pub fn test_last_word_length_solution() {
    	let result = Solution::length_of_last_word("luffy is still joyboy".to_string());
	assert_eq!(result, 6);
    }
}