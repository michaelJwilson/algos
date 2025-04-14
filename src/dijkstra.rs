use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::algo::dijkstra as petgraph_dijkstra;
use petgraph::data::FromElements;

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

#[derive(Debug)]
struct AdjacencyList {
    edges: HashMap<usize, Vec<Edge>>,
}

impl AdjacencyList {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, weight: usize) {
       self.edges.entry(from).or_default().push(Edge { to: to, weight: weight });
    }

    fn get_endpoints(&self) -> Vec<(u32, u32)> {
        let mut endpoints_list = Vec::new();
        
        for (&from, neighbors) in &self.edges {
            for edge in neighbors {
                // NB no weights.
                endpoints_list.push((from as u32, edge.to as u32));
            }
        }
        
        endpoints_list
    }
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

fn dijkstra(adjs: AdjacencyList, start: usize, goal: usize) -> Option<usize> {
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
        if let Some(neighbors) = adjs.edges.get(&position) {
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

fn get_adjacencies_fixture() -> AdjacencyList {
    // NB adjaceny list representation, with a list of edges for each node.
    let mut adjs = AdjacencyList::new();

    adjs.add_edge(0, 1, 4);
    adjs.add_edge(0, 2, 1);
    adjs.add_edge(2, 1, 2);
    adjs.add_edge(1, 3, 1);
    adjs.add_edge(2, 3, 5);

    adjs
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_petgraph_dijkstra -- --nocapture
    use super::*;

    #[test]
    fn test_adjacencies_fixture() {
        let adjs = get_adjacencies_fixture();

        println!("Adjacencies: {:?}", adjs);
    }

    #[test]
    fn test_dijkstra() {
        let start = 0;
        let goal = 3;

        let adjs = get_adjacencies_fixture();

        match dijkstra(adjs, start, goal) {
            Some(cost) => println!("Shortest path cost from {} to {} is {}", start, goal, cost),
            None => println!("No path found from {} to {}", start, goal),
        }
    }

    #[test]
    fn test_petgraph_dijkstra() {
       // NB see:
       //    https://docs.rs/petgraph/latest/petgraph/#example

       let adjs = get_adjacencies_fixture();
       let edges = adjs.get_endpoints();

       // println!("Edges: {:?}", edges);

       // NB <usize, ()> specify the type of the node and edge weights.
       let graph = UnGraph::<usize, ()>::from_edges(&edges);

       // NB find the shortest path from `0` to `3` using `1` as the cost for every edge.
       let node_map = petgraph_dijkstra(&graph, 0.into(), Some(3.into()), |_| 1);

       // NB assumes unit weight on edges.
       let exp = 2i32;
       let result = node_map.get(&NodeIndex::new(3)).unwrap();

       assert_eq!(&exp, result);
    }
} 