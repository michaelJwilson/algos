use rand::rng;
use rand::seq::IteratorRandom;
use std::cmp::{max, min};
use std::collections::VecDeque;

use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;
use petgraph::graph::{Graph, NodeIndex, UnGraph};

//  Ford-Fulkerson/Edmonds-Karp algorithm for max. flow on a directed graph.

fn bfs(residual_graph: &[Vec<i32>], source: usize, sink: usize, parent: &mut [isize]) -> bool {
    // NB  Shortest augmenting path [number of edges] by breadth-first search, i.e. shorter time to return.
    let mut visited = vec![false; residual_graph.len()];
    let mut queue = VecDeque::new();

    visited[source] = true;
    parent[source] = -1;

    queue.push_back(source);

    // NB first-in-first-out queue.
    while let Some(u) = queue.pop_front() {
        // NB loop over all un-visited neighbours.
        for v in 0..residual_graph.len() {
            if !visited[v] && residual_graph[u][v] > 0 {
                parent[v] = u as isize;
                visited[v] = true;

                queue.push_back(v);

                if v == sink {
                    return true;
                }
            }
        }
    }

    false
}

fn edmonds_karp(graph: Vec<Vec<i32>>, source: usize, sink: usize) -> i32 {
    // NB residual graph contains the residual capacity (c_uv - f_uv) on the forward edges,
    //    and the inverse flow, f_uv on the backward edges.
    let mut residual_graph = graph.clone();

    // NB save the frontier node that discovered a node on the tree as the parent, allowing for
    //    back trace.
    let mut parent = vec![-1; graph.len()];
    let mut max_flow = 0;

    // NB while there is an 'augmenting path' by breadth-first search.
    while bfs(&residual_graph, source, sink, &mut parent) {
        let mut path_flow = i32::MAX;

        // Find the minimum residual capacity in the path filled by BFS
        let mut v = sink;
        while v != source {
            let u = parent[v] as usize;
            path_flow = path_flow.min(residual_graph[u][v]);
            v = u;
        }

        // Update residual capacities of the edges and reverse edges
        let mut v = sink;
        while v != source {
            let u = parent[v] as usize;
            residual_graph[u][v] -= path_flow;
            residual_graph[v][u] += path_flow;
            v = u;
        }

        max_flow += path_flow;
    }

    max_flow
}

pub fn get_large_graph_fixture(node_count: usize) -> (NodeIndex, NodeIndex, isize, Graph<u8, u8>) {
    let mut g = Graph::<u8, u8>::new();
    let nodes: Vec<_> = (0..node_count).map(|i| g.add_node(i as u8)).collect();

    let source = nodes[0];
    let sink = nodes[node_count - 1];

    for j in 0..node_count {
        for i in j + 1..node_count {
            g.add_edge(nodes[j], nodes[i], 1_u8);
        }
    }

    (source, sink, -1, g)
}

fn get_adjacencies_fixture() -> (usize, usize, usize, Vec<Vec<i32>>) {
    let graph = vec![
        vec![0, 16, 0, 13, 0, 0, 0], // (0, 1, 16); (0, 3, 13);
        vec![0, 0, 10, 0, 12, 0, 0], // (1, 2, 10); (1, 4, 12);
        vec![0, 0, 0, 10, 0, 0, 0],  // (2, 3, 10);
        vec![0, 4, 0, 0, 0, 14, 0],  // (3, 1, 4); (3, 5, 14);
        vec![0, 0, 0, 9, 0, 0, 20],  // (4, 3, 9); (4, 6, 20);
        vec![0, 0, 0, 0, 7, 0, 4],   // (5, 4, 7); (5, 6, 4);
        vec![0, 0, 0, 0, 0, 0, 0],   // sink
    ];

    let source = 0;
    let sink = 6;
    let max_flow = 23;

    (source, sink, max_flow, graph)
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_petgraph_ford_fulkerson_large -- --nocapture
    use super::*;

    #[test]
    fn test_adjacencies_fixture() {
        let (source, sink, max_flow, graph) = get_adjacencies_fixture();
    }

    #[test]
    fn test_edmonds_karp() {
        let (source, sink, max_flow, graph) = get_adjacencies_fixture();
        let max_flow = edmonds_karp(graph, source, sink);
    }

    #[test]
    fn test_petgraph_ford_fulkerson() {
        //  Example of Fig. 24.2 of Cormen, Leiserson, Rivest and Stein, pg. 673.
        //  Contains anti-parallel edges, which necessitates addition of side-step nodes.
        //
        //  NB see:
        //    https://docs.rs/petgraph/latest/petgraph/algo/ford_fulkerson/fn.ford_fulkerson.html
        let mut graph = Graph::<u8, u8>::new();
        let source = graph.add_node(0);
        let _ = graph.add_node(1);
        let _ = graph.add_node(2);
        let _ = graph.add_node(3);
        let _ = graph.add_node(4);

        let destination = graph.add_node(5);

        // NB contains anti-parallel edges.
        graph.extend_with_edges(&[
            (0, 1, 16),
            (0, 2, 13),
            (1, 2, 10),
            (1, 3, 12),
            (2, 1, 4),
            (2, 4, 14),
            (3, 2, 9),
            (3, 5, 20),
            (4, 3, 7),
            (4, 5, 4),
        ]);

        // NB seems to be Edmonds-Karp; accepts anti-parallel edges.
        let (max_flow, _) = petgraph_ford_fulkerson(&graph, source, destination);

        assert_eq!(23, max_flow);
    }

    #[test]
    fn test_petgraph_ford_fulkerson_large() {
        let (source, sink, _, g) = get_large_graph_fixture(200);
        let (max_flow, _) = petgraph_ford_fulkerson(&g, source, sink);

        println!(
            "Large graph fixture with {:?} edges and nodes {:?} has max. flow {:?}",
            g.edge_count(),
            g.node_count(),
            max_flow,
        );
    }
}
