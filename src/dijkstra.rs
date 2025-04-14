use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

/*
----  TODO  ----
- Dijkstra's algorithm for shortest distance path of a node pair, s = u -> v.
- graph as an adjacency list.
- Initialize distance of each node as inf and parent as NULL pointer: Option().
- starting from root, u:
  -  update the distance of each child if (s, v).d >= (s, u).d + w(u, v)
  -  update the ancestor.
*/

#[derive(Debug)]
struct Edge {
    // NB each edge instance is defined only for a given source node.
    to: usize,
    weight: usize,
}

// NB define a state for the priority queue, which will return the min. distance node
//    (from source) known that has not been processed, as a min-heap.
#[derive(Debug, Eq, PartialEq)]
struct State {
    cost: usize,
    position: usize,
}

// NB Rust's BinaryHeap is a max-heap by default, meaning it will prioritize elements
//    with the largest value when popping elements.
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // NB reverse order for min-heap
        other.cost.cmp(&self.cost)
    }
}

// NB allows for cases where some nodes may not be comparable.
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn dijkstra(edges: &HashMap<usize, Vec<Edge>>, start: usize, goal: usize) -> Option<usize> {
    //  NB maintain current known distance between all encountered node pairs;
    //     queried correctly, i.e. *dist.get(&next.position).unwrap_or(&usize::MAX), achieves
    //     initial distances of INF for assumed usize type.
    let mut dist: HashMap<usize, usize> = HashMap::new();

    //  NB maintain a priority queue (pop returns the min. distance node in queue) as a BinaryHeap.
    let mut heap = BinaryHeap::new();

    // NB initialize distances: start/root node is at zero distance & to be processed first.
    dist.insert(start, 0);
    heap.push(State { cost: 0, position: start });

    // NB returns the min. distance state in the queue, starting with the root.
    while let Some(State { cost, position }) = heap.pop() {
        // NB If we reach the goal node, return the distance/cost.
        if position == goal {
            return Some(cost);
        }

        // NB node on the queue has an outdated distance;
        if cost > *dist.get(&position).unwrap_or(&usize::MAX) {
            continue;
        }

        // NB explore neighbors, initially of root.
        if let Some(neighbors) = edges.get(&position) {
            for edge in neighbors {
                let next = State {
                    cost: cost + edge.weight,
                    position: edge.to,
                };

                // NB here, unwrap_or achieves the new node distance initialized as MAX uint on system.
                if next.cost < *dist.get(&next.position).unwrap_or(&usize::MAX) {
                    dist.insert(next.position, next.cost);

                    // NB place neighbors on the queue (i.e. frontier); next may have a shorter distance
                    //    defined by another path, in which case it's skipped.
                    heap.push(next);
                }
            }
        }
    }

    // NB exhausted the heap without finding the goal node.
    None
}

fn get_graph_fixture() -> HashMap<usize, Vec<Edge>> {
    // NB adjaceny list representation, with a list of edges for each node.
    let mut edges: HashMap<usize, Vec<Edge>> = HashMap::new();

    // NB or_default presumably initializes the Vec first time a node is indexed.
    edges.entry(0).or_default().push(Edge { to: 1, weight: 4 }); // 0 -> 1
    edges.entry(0).or_default().push(Edge { to: 2, weight: 1 }); // 0 -> 2
    edges.entry(2).or_default().push(Edge { to: 1, weight: 2 }); // 1 -> 2
    edges.entry(1).or_default().push(Edge { to: 3, weight: 1 }); // ...
    edges.entry(2).or_default().push(Edge { to: 3, weight: 5 });

    edges
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_dijkstra -- --nocapture
    use super::*;

    #[test]
    fn test_graph_fixture() {
        let edges = get_graph_fixture(); 
    }

    #[test]
    fn test_dijkstra() {
        let start = 0;
        let goal = 3;

        let edges = get_graph_fixture();

        match dijkstra(&edges, start, goal) {
            Some(cost) => println!("Shortest path cost from {} to {} is {}", start, goal, cost),
            None => println!("No path found from {} to {}", start, goal),
        }
    }
}