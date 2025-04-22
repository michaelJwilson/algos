struct Solution;

pub fn is_valid_palindrome(input_string: &str) -> bool {
    let input_string_vec: Vec<char> = input_string.trim().chars().collect();
    let length = input_string_vec.len();

    if length < 2 {
        return true;
    }

    let mut first_index: usize = 0;
    let mut last_index: usize = input_string_vec.len() - 1;

    while first_index < last_index {
        // NB if no valid character is found, first_index has a null value of length, i.e. an invalid index.
        while !input_string_vec[first_index].is_alphanumeric() {
            first_index += 1;

            // NB no valid characters.
            if first_index == length {
                return true;
            }
        }

        // NB if first_index is null, last_index does not move from input_string_vec.len() - 1. Otherwise, we know first_index is valid
        //    and last index will take this value if none other is found.
        while !input_string_vec[last_index].is_alphanumeric() && (last_index > first_index) {
            last_index -= 1;
        }

        if last_index <= first_index {
            return true;
        } else {
            if input_string_vec[first_index]
                .to_uppercase()
                .collect::<String>()
                != input_string_vec[last_index]
                    .to_uppercase()
                    .collect::<String>()
            {
                return false;
            }

            first_index += 1;
            last_index -= 1;
        }
    }

    true
}

impl Solution {
    pub fn is_palindrome(s: String) -> bool {
        is_valid_palindrome(&s)
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_valid_palindrome -- --nocapture
    use super::*;

    #[test]
    pub fn test_valid_palindrome_empty() {
        let result = is_valid_palindrome("        ");
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_xxxxxa() {
        let result = is_valid_palindrome("        a");
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_axxxxx() {
        let result = is_valid_palindrome("a       ");
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_dotcomma() {
        let result = is_valid_palindrome(".,");
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_adot() {
        let result = Solution::is_palindrome(String::from("a."));
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_selim() {
        let result =
            Solution::is_palindrome(String::from("\"Sue,\" Tom smiles, \"Selim smote us.\""));
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_panama() {
        let result = is_valid_palindrome("A MAN, A PLAN, A CANAL: PANAMA");
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_panamania() {
        let result = is_valid_palindrome("A MAN, A PLAN, A CANAL: PANAMANIA");
        assert_eq!(result, false);
    }

    #[test]
    pub fn test_valid_palindrome_panama_lower() {
        let result = is_valid_palindrome("a man, a plan, a canal: panama");
        assert_eq!(result, true);
    }

    #[test]
    pub fn test_valid_palindrome_racecar() {
        let result = is_valid_palindrome("RACE A CAR");
        assert_eq!(result, false);
    }

    #[test]
    pub fn test_valid_palindrome_solution() {
        let result = Solution::is_palindrome(String::from("A MAN, A PLAN, A CANAL: PANAMA"));
        assert_eq!(result, true);
    }
}
