use std::sync::{Arc, Mutex};
use std::thread;

pub fn archer() {
    // NB atomic reference counted with thread-safe lock.
    let v = Arc::new(Mutex::new(vec![10, 20, 30]));

    // NB cloned reference
    let v2 = Arc::clone(&v);

    println!("v: {v:?}");

    //  NB spawned thread pushes to v;
    let handle = thread::spawn(move || {
        // NB mutable updated to locked v via v2 ref.
        let mut v2 = v2.lock().unwrap();

        v2.push(10);
    });

    // NB enclosed scope with block
    {
        // NB update v in main thread by shadowing v
        //    with mutable reference.
        let mut v = v.lock().unwrap();
        v.push(1000);
    }

    // NB syncs both main and spawned thread.
    handle.join().unwrap();

    println!("v: {v:?}");
}
