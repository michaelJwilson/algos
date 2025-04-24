use ndarray::Array2;
use num_traits::cast::ToPrimitive;
use rand::rng;
use rand::seq::IteratorRandom;
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::iter::zip;

use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;
use petgraph::algo::spfa;
use petgraph::graph::{Edge, Graph, Node, NodeIndex, UnGraph};
use petgraph::visit::{Bfs, EdgeRef};

macro_rules! edge_weight {
    ($value:expr) => {
        E::from($value)
    };
}

macro_rules! node_weight {
    () => {
        N::from(0)
    };
}

//
//  Ford-Fulkerson/Edmonds-Karp algorithm for max. flow on a directed graph.
//
//  WARNING petgraph node/edge deletion shifts indices!  The last node/edge replaces the deletion.
//
//  NB see petgraph source:
//     https://docs.rs/petgraph/latest/src/petgraph/graph_impl/mod.rs.html#498-510
//
//     e.g. pub struct Node:
//          https://docs.rs/petgraph/latest/src/petgraph/graph_impl/mod.rs.html#221
//
//          pub struct Edge:
//          https://docs.rs/petgraph/latest/src/petgraph/graph_impl/mod.rs.html#165
//
//          pub struct Graph (nodes and edges are Vec<Node>, etc.):
//          https://docs.rs/petgraph/latest/src/petgraph/graph_impl/mod.rs.html#390
//
//  TODO assumes a dense, adjaceny matrix.
fn bfs(adj_matrix: &Array2<i32>, source: usize, sink: usize, parent: &mut [usize]) -> bool {
    // NB  Shortest augmenting path [number of edges] by breadth-first search, i.e. shorter time to return.
    let mut visited = vec![false; adj_matrix.nrows()];

    // NB  Proceessing of each node adds in all neighbors (that have not been visited), N.
    //     i.e. each queue interaction removes one node and adds on N.
    let mut queue = VecDeque::with_capacity(adj_matrix.nrows());

    visited[source] = true;
    parent[source] = 0;

    queue.push_back(source);

    // NB first-in-first-out queue.
    while let Some(u) = queue.pop_front() {
        // NB loop over all un-visited neighbours.
        for (v, &capacity) in adj_matrix.row(u).indexed_iter() {
            if capacity > 0 && !visited[v] {
                parent[v] = u;
                visited[v] = true;

                if v == sink {
                    return true;
                }

                queue.push_back(v);
            }
        }
    }

    false
}

// TODO assumes a dense, adjaceny matrix.
pub fn edmonds_karp(adj_matrix: Array2<i32>, source: usize, sink: usize) -> (i32, Array2<i32>) {
    // NB residual graph contains the residual capacity (c_uv - f_uv) on the forward edges,
    //    and the inverse flow, f_uv on the backward edges.
    //
    //    save the frontier node that discovered a node on the tree as the parent, allowing for
    //    back trace.
    //
    //    in-place updates of the residual graph.
    let mut residual_graph = adj_matrix.clone();

    let mut parent = vec![0; residual_graph.nrows()];
    let mut max_flow = 0;

    // NB while there is an 'augmenting path' by breadth-first search.
    while bfs(&residual_graph, source, sink, &mut parent) {
        // NB find the minimum residual capacity in the path filled by BFS
        let mut v = sink;
        let mut path_flow = i32::MAX;

        // NB minimum capacity on the augmenting path
        while v != source {
            let u = parent[v];

            path_flow = path_flow.min(residual_graph[[u, v]]);

            v = u;
        }

        // NB update residual capacities of the edges and reverse edges
        let mut v = sink;

        while v != source {
            let u = parent[v];

            // NB forwards edge: residual capacity on graph.
            residual_graph[[u, v]] -= path_flow;

            // NB backwards edge:  -flow on forward edge.
            residual_graph[[v, u]] += path_flow;

            v = u;
        }

        max_flow += path_flow;
    }

    // TODO why transpose?
    (max_flow, residual_graph.t().to_owned())
}

// TODO assumes a dense, adjaceny matrix.
pub fn min_cut_labelling(
    node_count: usize,
    edge_flows: Vec<(NodeIndex, NodeIndex, u8)>,
    source: NodeIndex,
) -> Vec<bool> {
    //  Given forward edges flows for the max. flow, assign a pixel
    //  labelling by calculating distances from the source and assigning
    //  according to whether they are reachable.
    //
    //  NB  see:
    //      https://docs.rs/petgraph/latest/petgraph/algo/spfa/fn.spfa.html
    let mut g = Graph::new();

    // NB node with no weight
    let _ = (0..node_count).map(|_| g.add_node(()));

    (0..node_count).for_each(|_| {
        g.add_node(());
    });
    /*
    for _ in 0..node_count {
        g.add_node(());
    }
    */

    // NB see petgraph::graph::Edge
    for edge in edge_flows.into_iter() {
        g.add_edge(edge.0, edge.1, edge.2);
    }

    // NB compute shortest paths from node source to all others.
    //    see:
    //        https://docs.rs/petgraph/latest/petgraph/algo/spfa/fn.spfa.html
    //
    let path = spfa(&g, source, |edge| *edge.weight()).unwrap();

    let labels: Vec<bool> = path
        .distances
        .into_iter()
        .map(|dist| dist < u8::MAX)
        .collect();

    labels
}

pub fn get_petgraph_adj_matrix<N, E>(graph: &Graph<N, E>) -> Array2<i32>
where
    E: ToPrimitive, // Ensure edge weights can be converted to i32
{
    let num_nodes = graph.node_count();
    let mut adj_matrix = Array2::<i32>::zeros((num_nodes, num_nodes));

    for edge in graph.edge_references() {
        let source = edge.source().index();
        let target = edge.target().index();

        adj_matrix[[source, target]] = edge.weight().to_i32().unwrap();
    }

    adj_matrix
}

// -- fixtures --
pub fn get_adj_matrix_fixture() -> (usize, usize, usize, Array2<i32>) {
    let mut graph = Array2::<i32>::zeros((7, 7));

    graph[[0, 1]] = 16;
    graph[[0, 3]] = 13;
    graph[[1, 2]] = 10;
    graph[[1, 4]] = 12;
    graph[[2, 3]] = 10;
    graph[[3, 1]] = 4;
    graph[[3, 5]] = 14;
    graph[[4, 3]] = 9;
    graph[[4, 6]] = 20;
    graph[[5, 4]] = 7;
    graph[[5, 6]] = 4;

    let source = 0;
    let sink = 6;
    let max_flow = 23;

    (source, sink, max_flow, graph)
}

pub fn get_clrs_graph_fixture<N, E>() -> (NodeIndex, NodeIndex, E, Graph<N, E>)
where
    N: Default + Copy + From<u8>,
    E: From<u8>,
{
    //  Example of Fig. 24.2 of Cormen, Leiserson, Rivest and Stein, pg. 673.
    //  Contains 6 nodes and 10 edges - including anti-parallel, which necessitates addition
    //  of "side-step" nodes.
    //
    let mut graph = Graph::<N, E>::with_capacity(6, 10);

    // TODO define x = N::from(0)?
    let source = graph.add_node(node_weight!());
    let _ = graph.add_node(node_weight!());
    let _ = graph.add_node(node_weight!());
    let _ = graph.add_node(node_weight!());
    let _ = graph.add_node(node_weight!());

    let sink = graph.add_node(node_weight!());

    // NB contains 10 edges (including anti-parallel).
    graph.extend_with_edges([
        (0, 1, edge_weight!(16)),
        (0, 2, edge_weight!(13)),
        (1, 2, edge_weight!(10)),
        (1, 3, edge_weight!(12)),
        (2, 1, edge_weight!(4)),
        (2, 4, edge_weight!(14)),
        (3, 2, edge_weight!(9)),
        (3, 5, edge_weight!(20)),
        (4, 3, edge_weight!(7)),
        (4, 5, edge_weight!(4)),
    ]);

    (source, sink, 23.into(), graph)
}

pub fn get_large_graph_fixture(node_count: usize) -> (NodeIndex, NodeIndex, usize, Graph<u8, u8>) {
    let mut g = Graph::<u8, u8>::with_capacity(node_count, node_count * (node_count - 1) / 2);
    let nodes: Vec<_> = (0..node_count).map(|i| g.add_node(i as u8)).collect();

    let source = nodes[0];
    let sink = nodes[node_count - 1];

    for j in 0..node_count {
        for i in j + 1..node_count {
            g.add_edge(nodes[j], nodes[i], 1_u8);
        }
    }

    (source, sink, 0, g)
}

#[cfg(test)]
mod tests {
    //  cargo test max_flow -- --nocapture
    use super::*;

    #[test]
    fn test_max_flow_adjacencies_fixture() {
        let (source, sink, max_flow, graph) = get_adj_matrix_fixture();
    }

    #[test]
    fn test_max_flow_edmonds_karp() {
        let (source, sink, max_flow, graph) = get_adj_matrix_fixture();
        let max_flow = edmonds_karp(graph, source, sink);
    }

    #[test]
    fn test_max_flow_clrs_graph_fixture() {
        // NB correspoding max. flow is two-channel: s, v1, v3, t == 23, 12, 12, 19, 23.
        //                                           s, v2, v4, t == 23, 11, 11,  4, 23.
        //
        //    saturated, min. cut edges are (1 -> 3), (4 -> 3) and (4 -> sink).
        //    implied pixel labelling: {s/0, 1, 2, 4}, {3, t/5}.
        //
        let (source, sink, exp_max_flow, graph) = get_clrs_graph_fixture::<u8, u8>();

        // NB O(1) access
        let num_nodes = graph.node_count();
        let num_edges = graph.edge_count();

        assert_eq!(num_nodes, 6);
        assert_eq!(num_edges, 10);

        let adj_matrix = get_petgraph_adj_matrix(&graph);

        // NB exp given by-hand sum of clrs edge weights.
        assert_eq!(adj_matrix.sum(), 109);

        // NB in-place update of adj_matrix to residual graph.
        let (max_flow, res_graph) = edmonds_karp(adj_matrix, 0, 5);

        assert_eq!(exp_max_flow as i32, max_flow);

        // println!("{:?}", min_cut_edges);
        // println!("{:?}", visited);

        /*
        // NB returns an iterator of all nodes with an edge starting from a, respecting direction.
        let num_neighbors = graph.neighbors(sink).count();
        let num_edges = graph.edges(sink).count();

        // TODO BUG 0, 0??
        // println!("{:?}", num_neighbors);
        // println!("{:?}", num_edges);

        let node_weight = graph.node_weight(sink).unwrap();
        */
    }

    #[test]
    fn test_max_flow_petgraph_bfs() {
        let mut graph = Graph::<_, ()>::new();
        let a = graph.add_node(0);

        let mut bfs = Bfs::new(&graph, a);
        while let Some(nx) = bfs.next(&graph) {
            // we can access `graph` mutably here still
            graph[nx] += 1;
        }
    }

    #[test]
    fn test_max_flow_petgraph_map() {
        let mut graph = Graph::<u8, u8>::new();

        let a = graph.add_node(1);
        let b = graph.add_node(2);
        let c = graph.add_node(3);

        graph.add_edge(a, b, 10);
        graph.add_edge(b, c, 20);

        let new_graph = graph.map(
            |node_idx, node_weight| { node_weight * 2 },
            |edge_idx, edge_weight| { edge_weight + 5 },
        );
    }

    #[test]
    fn test_max_flow_petgraph_ford_fulkerson() {
        //  NB see:
        //    https://docs.rs/petgraph/latest/petgraph/algo/ford_fulkerson/fn.ford_fulkerson.html
        //
        let (source, sink, exp_max_flow, graph) = get_clrs_graph_fixture::<u8, u8>();

        // NB seems to be Edmonds-Karp; accepts anti-parallel edges.
        let (max_flow, edge_flows) = petgraph_ford_fulkerson(&graph, source, sink);

        assert_eq!(exp_max_flow, max_flow);
        assert_eq!(edge_flows.len(), graph.edge_count());

        // NB edge flows is ordered as for clrs fixture, i.e. (0, 1, 16), (0, 2, 13), etc.
        //    edge_flow[0] == (0, 1, 12), edge_flow[1] == (0, 2, 11); sum is source outflow == max_flow.
        //
        //    similarly, sink_inflow == max_flow.
        assert_eq!(edge_flows[0] + edge_flows[1], max_flow);
    }

    #[test]
    fn test_max_flow_petgraph_ford_fulkerson_large() {
        let (source, sink, _, g) = get_large_graph_fixture(200);
        let (max_flow, edge_flows) = petgraph_ford_fulkerson(&g, source, sink);

        println!(
            "Large graph fixture with {:?} edges and nodes {:?} has max. flow {:?}",
            g.edge_count(),
            g.node_count(),
            max_flow,
        );
    }

    #[test]
    fn test_max_flow_min_cut_labelling() {
        let (source, sink, exp_max_flow, graph) = get_clrs_graph_fixture::<u8, u8>();
        let (max_flow, flows_on_edges) = petgraph_ford_fulkerson(&graph, source, sink);

        let num_nodes = graph.node_count();
        let mut edge_flows = Vec::new();

        for (ii, (edge, weight)) in zip(graph.edge_references(), graph.edge_weights()).enumerate() {
            let flow = flows_on_edges[ii];

            // TODO
            if flow > 0 && !(flow == *weight) {
                let new_edge = (edge.source(), edge.target(), flow);
                edge_flows.push(new_edge);
            }
        }

        let labels = min_cut_labelling(num_nodes, edge_flows, source);

        // NB min-cut edges are (1, 3), (2, 3), (4, 3), (4, 5/sink); i.e. separating 3 & 5 from sink.
        assert_eq!(labels, [true, true, true, false, true, false]);
    }

    #[test]
    fn test_max_flow_min_cut_labelling_large() {
        let (source, sink, _, g) = get_large_graph_fixture(10);
        let (max_flow, flows_on_edges) = petgraph_ford_fulkerson(&g, source, sink);

        let num_nodes = g.node_count();
        let mut edge_flows = Vec::new();

        for (ii, (edge, weight)) in zip(g.edge_references(), g.edge_weights()).enumerate() {
            let flow = flows_on_edges[ii];

            // NB non-saturated flow on edge, i.e. not on the min. cut.
            if flow > 0 && !(flow == *weight) {
                let new_edge = (edge.source(), edge.target(), flow);

                edge_flows.push(new_edge);
            }
        }

        let labels = min_cut_labelling(num_nodes, edge_flows, source);
        let label_count = labels.iter().count();

        println!("{:?}\t{:?}", num_nodes, label_count);

        // NB min-cut edges are (1, 3), (2, 3), (4, 3), (4, 5/sink); i.e. separating 3 & 5 from sink.
        // assert_eq!(labels, [true, true, true, false, true, false]);
    }
}
