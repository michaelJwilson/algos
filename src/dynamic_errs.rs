use std::fs;
use std::io::Read;
use std::error::Error;

// NB 
fn read_count(path: &str) -> Result<i32, Box<dyn Error>> {
    let mut count_str = String::new();
    fs::File::open(path)?.read_to_string(&mut count_str)?;
    let count: i32 = count_str.parse()?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_read_count() {
        fs::write("count.dat", "1i3").unwrap();
	
    	match read_count("count.dat") {
	    Ok(count) => println!("Count: {count}"),
            Err(err) => println!("Error: {err}"),
        }
    }
}

