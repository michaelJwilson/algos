use std::fs::File;
use std::io::Read;

// NB read a simple text file to a String. Remember: str is a slice view.
fn main() {
    let file: Result<File, std::io::Error> = File::open("diary.txt");
    
    match file {
        Ok(mut file) => {
            let mut contents = String::new();
	    
            if let Ok(bytes) = file.read_to_string(&mut contents) {
                println!("Dear diary: {contents} ({bytes} bytes)");
            } else {
                println!("Could not read file content");
            }
        }
        Err(err) => {
            println!("The diary could not be opened: {err}");
        }
    }
}