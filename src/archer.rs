use std::thread;
use std::sync::{Arc, Mutex};

pub fn archer() {
    let v = Arc::new(Mutex::new(vec![10, 20, 30]));
    let v2 = Arc::clone(&v);
    
    let handle = thread::spawn(move || {
        let mut v2 = v2.lock().unwrap();	
        v2.push(10);
    });

    {
	// NB shadows v with mutable reference.
        let mut v = v.lock().unwrap();	
        v.push(1000);
    }

    // NB syncs both main and spawned thread.
    handle.join().unwrap();

    println!("v: {v:?}");
}