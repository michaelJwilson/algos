struct Solution;

pub fn last_word_length(sentence: &str) -> i32 {
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

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_last_word_length -- --nocapture
    use super::*;

    #[test]
    pub fn test_last_word_length_hello_world() {
        let result = last_word_length("hello world");
	assert_eq!(result, 5);
    }

    #[test]
    pub fn test_last_word_length_hello() {
        let result = last_word_length("hello");
	assert_eq!(result, 5);
    }

    #[test]
    pub fn test_last_word_length_fly_me() {
        let result = last_word_length("    fly me   to  the moon  ");
	assert_eq!(result, 4);
    }

    #[test]
    pub fn test_last_word_length_luffy() {
        let result = last_word_length("luffy is still joyboy");
    	assert_eq!(result, 6);
    }
}