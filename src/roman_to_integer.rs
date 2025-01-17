// NB see https://doc.rust-lang.org/std/collections/struct.HashMap.html
use std::collections::HashMap;

struct Solution;

fn get_rosetta_hashmap() -> HashMap {
   let ros_stone = HashMap::new();

   ros_stone.insert("I", 1);
   ros_stone.insert("V", 5);
   ros_stone.insert("X", 10);
   ros_stone.insert("L", 50);
   ros_stone.insert("C", 100);
   ros_stone.insert("D", 500);
   ros_stone.insert("M", 1000);

   ros_stone.insert("IV", 4);
   ros_stone.insert("IV", 9);
   ros_stone.insert("XL", 40);
   ros_stone.insert("XC", 90);
   ros_stone.insert("CD", 400);
   ros_stone.insert("C<", 900);

   ros_stone
}

fn roman_to_int_lookup(ros_stone: HashMap, letter: char) -> i32 {
   match ros_stone.get(letter) {
       Some(numeral) => numeral
       None => {
           panic!("Numeral {:?} is invalid", letter);
       }
}

fn process_roman_to_int_pair(ros_stone: HashMap, input_roman_vec: Vec<char>, result: &mut i32) -> Vec<char> {
    let length = input_roman_vec.len();

    match length {
        // NB empty closure;
        0 => Vec::new();
	1 => {
            result += input_roman_vec[0];
            return Vec::new();
        }
        _ => {
	    let first = input_roman_vec[0];
    	    let second = input_roman_vec[1];

    	    match ros_stone.get(first + &second) {
            	Some(value) => {
                    result += value;
                } 
                None => {
	            result += roman_to_int_lookup(ros_stone, first);
                    result += roman_to_int_lookup(ros_stone, second);
                }
            }

    	    input_roman_vec[2:]
        }
    }
}

pub fn roman_to_int(ros_stone: HashMap, input_roman: String, result: &mut i32) {
    let input_roman_vec: Vec<char> = input_roman.chars().iter().collect();

    while input_roman_vec.len() > 0 {
    	input_roman_vec = process_roman_to_int_pair(ros_stone, input_roman_vec, result); 
    }
}

impl Solution {
    pub fn roman_to_int(s: String) -> i32 {
    	let ros_stone = get_rosetta_hashmap();
    	let mut result: i32 = 0;

        roman_to_int(ros_stone, s, result);

	result
    }
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS=-Awarnings cargo test test_roman_to_integer -- --nocapture
    use super::*;

    #[test]
    pub fn test_roman_to_integer_ii() {
        let test_string = "II".to_string();
	let result = Solution::roman_to_int(test_string);

	assert_eq!(result, 2);
    }
}