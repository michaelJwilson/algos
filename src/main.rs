use pprof;
use std::fs::File;
use algos::dijkstra::{dijkstra, get_adjacencies_fixture};

fn main() {
    /*
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1_000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();
    */

    println!("Welcome to algos.");

    let adjs = get_adjacencies_fixture();
    let cost = dijkstra(adjs, 0, 3).unwrap();

    /*
    if let Ok(report) = guard.report().build() {
        let file = File::create("flamegraph.svg").unwrap();
        let mut options = pprof::flamegraph::Options::default();
        options.image_width = Some(2500);
        report.flamegraph_with_options(file, &mut options).unwrap();
    };
    */

    println!("Done.");
}
