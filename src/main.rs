use std::thread;
use std::time::Duration;

fn main() {
    let handle = thread::spawn(|| {
        let thread_id = thread::current().id();

        for i in 0..10 {
            println!("Count in thread: {i}!");
            thread::sleep(Duration::from_millis(5));
        }

	thread_id
    });

    for i in 0..5 {
        println!("Main thread: {i}");
	
        thread::sleep(Duration::from_millis(5));
    }

    let thread_id = handle.join().unwrap();
}