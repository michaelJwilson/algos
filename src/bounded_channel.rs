use std::sync::mpsc;
use std::thread;
use std::time::Duration;

//
//
//

fn bounded_channel() {
    // NB The buffer size specifies the maximum number of messages
    //    that can be stored in the channel at any given time - three
    //    in this case.
    let (tx, rx) = mpsc::sync_channel(3);

    // NB move achieves ownership of the input object in the
    //    new thread.
    thread::spawn(move || {
        let thread_id = thread::current().id();

        for i in 0..10 {
            tx.send(format!("Message {i}")).unwrap();

            println!("{thread_id:?}: sent Message {i}");
        }

        println!("{thread_id:?}: done");
    });

    // NB main thread sleeps.
    thread::sleep(Duration::from_millis(100));

    for msg in rx.iter() {
        println!("Main: got {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_channel() {
        bounded_channel();
    }
}
