fn get_longest_prefix(input_str: &str) -> usize {
   let mut seen = String::new();
   let mut result = 0;

   for char in input_str.chars() {
       if !seen.contains(char) {
           seen.push(char);
	   result += 1;
       }
       else {
           return result;
       }
   }

   result
}

fn get_longest_substring(input_str: &str) -> usize {
   let mut result = vec![0; input_str.len()];

   for ii in 0..result.len() {
       result[ii] = get_longest_prefix(&input_str[ii..])
   }

   *result.iter().max().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
}