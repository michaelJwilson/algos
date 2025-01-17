// NB see https://doc.rust-lang.org/std/collections/struct.HashMap.html
use std::collections::HashMap;

struct Solution;

fn get_rosetta_hashmap() -> HashMap<&'static str, i32> {
   let mut ros_stone = HashMap::new();

   ros_stone.insert("I", 1);
   ros_stone.insert("V", 5);
   ros_stone.insert("X", 10);
   ros_stone.insert("L", 50);
   ros_stone.insert("C", 100);
   ros_stone.insert("D", 500);
   ros_stone.insert("M", 1000);

   ros_stone.insert("IV", 4);
   ros_stone.insert("IX", 9);
   ros_stone.insert("XL", 40);
   ros_stone.insert("XC", 90);
   ros_stone.insert("CD", 400);
   ros_stone.insert("CM", 900);

   ros_stone
}

fn roman_to_int_lookup(ros_stone: &HashMap<&'static str, i32>, letter: &str) -> i32 {
   // NB method calls propagate through reference.
   match ros_stone.get(letter) {
       Some(numeral) => *numeral,
       None => {
           panic!("Numeral {:?} is invalid", letter);
       }
   }
}

fn process_roman_to_int_pair(ros_stone: &HashMap<&'static str, i32>, input_roman_vec: Vec<char>, result: &mut i32) -> Vec<char> {
    let length = input_roman_vec.len();

    match length {
        // NB empty closure;
        0 => Vec::new(),
	1 => {
            *result += roman_to_int_lookup(ros_stone, &input_roman_vec[0].to_string());
            Vec::new()
        },
        _ => {
	    let first = input_roman_vec[0].to_string();
    	    let second = input_roman_vec[1].to_string();

    	    match ros_stone.get(format!("{}{}", first, second).as_str()) {
            	Some(value) => {
                    *result += value;
		    input_roman_vec[2..].to_vec()
                } 
                None => {
	            *result += roman_to_int_lookup(ros_stone, &first);
		    input_roman_vec[1..].to_vec()
                }
            }
        }
    }
}

pub fn roman_to_int_naive(ros_stone: &HashMap<&'static str, i32>, input_roman: String, result: &mut i32) {
    let mut input_roman_vec: Vec<char> = input_roman.chars().collect();

    while !input_roman_vec.is_empty() {
    	input_roman_vec = process_roman_to_int_pair(ros_stone, input_roman_vec, result); 
    }
}

fn roman_to_int(s: &str) -> i32 {
    let mut roman_map = HashMap::new();
    
    roman_map.insert('I', 1);
    roman_map.insert('V', 5);
    roman_map.insert('X', 10);
    roman_map.insert('L', 50);
    roman_map.insert('C', 100);
    roman_map.insert('D', 500);
    roman_map.insert('M', 1000);

    let mut result = 0;
    let mut prev_value = 0;

    for c in s.chars().rev() {
    	// NB unwrap_or is useful.
        let value = *roman_map.get(&c).unwrap_or(&0);
	
        if value < prev_value {
            result -= value;
        } else {
            result += value;
        }
	
        prev_value = value;
    }

    result
}

impl Solution {
    pub fn roman_to_int_naive(s: String) -> i32 {
    	let ros_stone = get_rosetta_hashmap();
    	let mut result: i32 = 0;

        roman_to_int_naive(&ros_stone, s, &mut result);

	result
    }

    pub fn roman_to_int(s: String) -> i32 {
        roman_to_int(&s)
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS=-Awarnings cargo test test_roman_to_integer -- --nocapture
    //
    // see:
    //     https://leetcode.com/problems/roman-to-integer/?envType=study-plan-v2&envId=top-interview-150
    //
    use super::*;

    #[test]
    pub fn test_roman_to_integer_iii() {
	let result = Solution::roman_to_int(String::from("III"));
	assert_eq!(result, 3);
    }

    #[test]
    pub fn test_roman_to_integer_lviii() {
        let result = Solution::roman_to_int(String::from("LVIII"));
        assert_eq!(result, 58);
    }

    #[test]
    pub	fn test_roman_to_integer_mcmxciv() {
	let result = Solution::roman_to_int(String::from("MCMXCIV"));
        assert_eq!(result, 1994);
    }
}