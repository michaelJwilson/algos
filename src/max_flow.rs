use image::imageops::FilterType;
use image::{ImageBuffer, Luma};
use ndarray::Array2;
use num_traits::cast::ToPrimitive;
use rand::seq::IteratorRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::iter::zip;

use petgraph::algo::ford_fulkerson as petgraph_ford_fulkerson;
use petgraph::algo::spfa;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{Edge, Graph, Node, NodeIndex, UnGraph};
use petgraph::visit::{Bfs, EdgeRef};

macro_rules! edge_weight {
    ($value:expr) => {
        E::from($value)
    };
}

macro_rules! node_weight {
    ($type:ty) => {
        <$type>::from(0)
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
pub fn edmonds_karp(adj_matrix: Array2<u32>, source: usize, sink: usize) -> (i32, Array2<i32>) {
    // NB residual graph contains the residual capacity (c_uv - f_uv) on the forward edges,
    //    and the inverse flow, f_uv on the backward edges.
    //
    //    save the frontier node that discovered a node on the tree as the parent, allowing for
    //    back trace.
    //
    //    in-place updates of the residual graph.
    let mut residual_graph = adj_matrix.clone().mapv(|x| x as i32);

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

pub fn min_cut_labelling(graph: &Graph<u32, u32>, source: NodeIndex, sink: NodeIndex) -> Vec<bool> {
    //  Given forward edges flows for the max. flow, assign a pixel
    //  labelling by calculating distances from the source and assigning
    //  according to whether they are reachable.
    //
    //  NB  see:
    //      https://docs.rs/petgraph/latest/petgraph/algo/spfa/fn.spfa.html
    //
    let (max_flow, max_flow_on_edges) = petgraph_ford_fulkerson(&graph, source, sink);

    let mut edge_flows = Vec::new();

    for (ii, (edge, weight)) in zip(graph.edge_references(), graph.edge_weights()).enumerate() {
        let flow = max_flow_on_edges[ii];

        // NB non-saturated (!min. cut) edges on the max. flow graph.
        if flow > 0 && flow != *weight {
            edge_flows.push((edge.source(), edge.target(), flow));
        }
    }

    // TODO more efficient?
    let mut g = graph.clone();
    g.clear_edges();

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
        .map(|dist| dist < u32::MAX)
        .collect();

    labels
}

pub fn get_petgraph_adj_matrix<N, E>(graph: &Graph<N, E>) -> Array2<E>
where
    E: ToPrimitive + std::clone::Clone + num_traits::Zero,
{
    let num_nodes = graph.node_count();
    let mut adj_matrix = Array2::<E>::zeros((num_nodes, num_nodes));

    for edge in graph.edge_references() {
        let source = edge.source().index();
        let target = edge.target().index();

        adj_matrix[[source, target]] = edge.weight().clone();
    }

    adj_matrix
}

// -- fixtures --
pub fn get_adj_matrix_fixture<E>() -> (usize, usize, usize, Array2<E>)
where
    E: From<u32> + num_traits::Zero + std::clone::Clone,
{
    let mut adj_matrix = Array2::<E>::zeros((7, 7));

    adj_matrix[[0, 1]] = 16.into();
    adj_matrix[[0, 3]] = 13.into();
    adj_matrix[[1, 2]] = 10.into();
    adj_matrix[[1, 4]] = 12.into();
    adj_matrix[[2, 3]] = 10.into();
    adj_matrix[[3, 1]] = 4.into();
    adj_matrix[[3, 5]] = 14.into();
    adj_matrix[[4, 3]] = 9.into();
    adj_matrix[[4, 6]] = 20.into();
    adj_matrix[[5, 4]] = 7.into();
    adj_matrix[[5, 6]] = 4.into();

    let source = 0;
    let sink = 6;
    let max_flow = 23;

    (source, sink, max_flow, adj_matrix)
}

pub fn get_clrs_graph_fixture<N, E>() -> (NodeIndex, NodeIndex, E, Graph<N, E>)
where
    N: Default + Copy + From<u32>,
    E: From<u32>,
{
    //  Example of Fig. 24.2 of Cormen, Leiserson, Rivest and Stein, pg. 673.
    //  Contains 6 nodes and 10 edges - including anti-parallel, which necessitates addition
    //  of "side-step" nodes.
    //
    let mut graph = Graph::<N, E>::with_capacity(6, 10);

    let source = graph.add_node(node_weight!(N));
    let _ = graph.add_node(node_weight!(N));
    let _ = graph.add_node(node_weight!(N));
    let _ = graph.add_node(node_weight!(N));
    let _ = graph.add_node(node_weight!(N));

    let sink = graph.add_node(node_weight!(N));

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

pub fn get_large_graph_fixture<N, E>(
    node_count: usize,
    sparsity: f64,
) -> (Vec<NodeIndex>, Graph<N, E>)
where
    N: Default + Copy + From<u32>,
    E: From<u32>,
{
    assert!(sparsity <= 1., "sparsity argument must be [0, 1].");

    let mut graph = Graph::<N, E>::new();

    // TODO no allocation.
    let nodes: Vec<NodeIndex> = (0..node_count)
        .map(|i| graph.add_node(N::from(i as u32)))
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for i in 0..node_count {
        for j in i + 1..min(i + 3, node_count) {
            if i == j {
                continue;
            }

            if rng.random::<f64>() < sparsity {
                let edge_weight = rng.random_range(1..=32);

                graph.add_edge(nodes[i], nodes[j], E::from(edge_weight));
            }
        }
    }

    (nodes, graph)
}

pub fn get_checkerboard_fixture(N: usize, sampling: usize, error_rate: f64) -> Array2<u32> {
    let mut result = Array2::<u32>::zeros((N, N));

    // TODO efficiency.
    for i in 0..N {
        for j in 0..N {
            if (i + j) % 2 == 0 {
                result[[i, j]] = 1_u32;
            }
        }
    }

    let fine_size = N * sampling;
    let mut fine_result = Array2::<u32>::zeros((fine_size, fine_size));

    for i in 0..N {
        for j in 0..N {
            let value = result[[i, j]];

            for di in 0..sampling {
                for dj in 0..sampling {
                    fine_result[[i * sampling + di, j * sampling + dj]] = value;
                }
            }
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let num_pixels = fine_size * fine_size;
    let num_noisy_pixels = (num_pixels as f64 * error_rate) as usize;

    for _ in 0..num_noisy_pixels {
        let x = rng.random_range(0..fine_size);
        let y = rng.random_range(0..fine_size);

        fine_result[[x, y]] = if rng.random_bool(0.5) { 1 } else { 0 };
    }

    fine_result
}

#[inline]
fn valid_indices(num_rows: usize, num_cols: usize, row: i32, col: i32) -> bool {
    (row >= 0) && (row < num_rows as i32) && (col >= 0) && (col < num_cols as i32)
}

pub fn binary_image_map_graph(binary_image: Array2<u32>) -> ( Vec<NodeIndex>, Graph<u32, u32> ) { 
    //
    //  See pg. 237 of Computer Vision, Prince.
    //
    //  NB below, left, right, above.
    let NEIGHBOR_OFFSETS: [(i32, i32); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

    let num_pixels = binary_image.len();

    let num_rows = binary_image.nrows();
    let num_cols = binary_image.ncols();

    // NB source + sink + one-per-pixel.
    let num_nodes = 2 + num_pixels;

    let mut graph =
        Graph::<u32, u32>::with_capacity(num_nodes, num_pixels + num_pixels + 2 * (num_pixels - 1));

    let nodes: Vec<NodeIndex> = (0..num_nodes)
        .map(|i| graph.add_node(0))
        .collect();

    let source = nodes[0];
    let sink = nodes[nodes.len() - 1];

    for (ii, ((row, col), value_ref)) in binary_image.indexed_iter().enumerate() {
        let value = *value_ref;

        // NB zero point shift due to source node.
        let node_idx: NodeIndex = NodeIndex::new(1 + col + row * num_cols);

        // NB 0 maps to source; 1 maps to sink.
        // TODO efficient?
        if value == 0 {
            graph.add_edge(source, node_idx, 0);
            graph.add_edge(node_idx, sink, 1);
        } else {
            graph.add_edge(source, node_idx, 1);
            graph.add_edge(node_idx, sink, 0);
        }

        for (di, dj) in NEIGHBOR_OFFSETS {
            // NB i32, not usize, required for subtraction.
            let new_row_index = row as i32 + di;
            let new_col_index = col as i32 + dj;

            if valid_indices(num_rows, num_cols, new_row_index, new_col_index) {
                let neighbor_value = binary_image[[
                    new_row_index.try_into().unwrap(),
                    new_col_index.try_into().unwrap(),
                ]];

                let neighbor_idx: NodeIndex = NodeIndex::new(
                    (1 + new_col_index + new_row_index * num_cols as i32)
                        .try_into()
                        .unwrap(),
                );

                // TODO [pair_cost]
                let pair_cost = (value != neighbor_value) as u32;

                // NB P_ab(1,0)
                graph.add_edge(node_idx, neighbor_idx, pair_cost);

                // NB P_ab(0, 1)
                graph.add_edge(neighbor_idx, node_idx, pair_cost);
            }
        }
    }

    (nodes, graph)
}

#[cfg(test)]
mod tests {
    //  cargo test max_flow -- --nocapture
    use super::*;

    #[test]
    fn test_max_flow_adjacencies_fixture() {
        let (source, sink, max_flow, graph) = get_adj_matrix_fixture::<u32>();
    }

    #[test]
    fn test_max_flow_edmonds_karp() {
        let (source, sink, max_flow, graph) = get_adj_matrix_fixture::<u32>();
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
        let (source, sink, exp_max_flow, graph) = get_clrs_graph_fixture::<u32, u32>();

        // NB O(1) access
        let num_nodes = graph.node_count();
        let num_edges = graph.edge_count();

        assert_eq!(num_nodes, 6);
        assert_eq!(num_edges, 10);

        let adj_matrix = get_petgraph_adj_matrix(&graph);

        // NB exp given by-hand sum of clrs edge weights.
        assert_eq!(adj_matrix.sum(), 109);
    }

    #[test]
    fn test_max_flow_petgraph_bfs() {
        let (source, sink, _, mut graph) = get_clrs_graph_fixture::<u32, u32>();

        let mut bfs = Bfs::new(&graph, source);

        while let Some(nx) = bfs.next(&graph) {
            // NB we can mutably access 'graph'
            graph[nx] += 1;
        }
    }

    #[test]
    fn test_max_flow_petgraph_map() {
        let (source, sink, _, graph) = get_clrs_graph_fixture::<u32, u32>();
        let new_graph = graph.map(
            |node_idx, node_weight| node_weight * 2,
            |edge_idx, edge_weight| edge_weight + 5,
        );
    }

    #[test]
    fn test_max_flow_petgraph_ford_fulkerson() {
        //  NB see:
        //    https://docs.rs/petgraph/latest/petgraph/algo/ford_fulkerson/fn.ford_fulkerson.html
        //
        let (source, sink, exp_max_flow, graph) = get_clrs_graph_fixture::<u32, u32>();

        // NB seems to be Edmonds-Karp; accepts anti-parallel edges.
        let (max_flow, max_flow_on_edges) = petgraph_ford_fulkerson(&graph, source, sink);

        assert_eq!(exp_max_flow, max_flow);
        assert_eq!(max_flow_on_edges.len(), graph.edge_count());

        // NB edge flows is ordered as for clrs fixture, i.e. (0, 1, 16), (0, 2, 13), etc.
        //    edge_flow[0] == (0, 1, 12), edge_flow[1] == (0, 2, 11); sum is source outflow == max_flow.
        //
        //    similarly, sink_inflow == max_flow.
        assert_eq!(max_flow_on_edges[0] + max_flow_on_edges[1], max_flow);
    }

    #[test]
    fn test_max_flow_petgraph_ford_fulkerson_large() {
        let (nodes, graph) = get_large_graph_fixture::<u32, u32>(200, 0.1);

        let (source, sink) = (nodes[0], nodes[nodes.len() - 1]);
        let (max_flow, max_flow_on_edges) = petgraph_ford_fulkerson(&graph, source, sink);

        println!(
            "Large graph fixture with {:?} edges and nodes {:?} has max. flow {:?}",
            graph.edge_count(),
            graph.node_count(),
            max_flow,
        );
    }

    #[test]
    fn test_max_flow_min_cut_labelling() {
        let (source, sink, exp_max_flow, graph) = get_clrs_graph_fixture::<u32, u32>();
        let labels = min_cut_labelling(&graph, source, sink);

        // NB min-cut edges are (1, 3), (2, 3), (4, 3), (4, 5/sink); i.e. separating 3 & 5 from sink.
        assert_eq!(labels, [true, true, true, false, true, false]);
    }

    #[test]
    fn test_max_flow_min_cut_labelling_large() {
        let (nodes, graph) = get_large_graph_fixture::<u32, u32>(5, 1.);
        let (source, sink) = (nodes[0], nodes[nodes.len() - 2]);

        // NB max. flow saturates (0, 1), (2,3) and has zero flow on (1,2) and (2,4).
        //    implies labelling: {0, 2} and {1, 3, 4} as reachable from source.
        let labels = min_cut_labelling(&graph, source, sink);
        let num_source_labelled = labels.iter().filter(|&&x| x).count() as i32;

        println!("{:?}", labels);
        println!("{:?}", num_source_labelled);

        assert_eq!(labels, vec![true, false, true, false, false]);
    }

    #[test]
    fn test_max_flow_min_cut_labelling_scale() {
        let (nodes, graph) = get_large_graph_fixture::<u32, u32>(100, 1.);
        let (source, sink) = (nodes[0], nodes[nodes.len() - 2]);

        let labels = min_cut_labelling(&graph, source, sink);
        let num_source_labelled = labels.iter().filter(|&&x| x).count() as i32;

        println!("{:?}", num_source_labelled);

        // NB regression test.
        assert_eq!(num_source_labelled, 24);
    }

    #[test]
    fn test_max_flow_checkerboard_fixture() {
        let N = 8;
        let sampling = 4;
        let error_rate = 0.25;
        let checkerboard = get_checkerboard_fixture(N, sampling, error_rate);

        //  println!("{:?}", checkerboard);

        let mut image = ImageBuffer::new((N * sampling) as u32, (N * sampling) as u32);

        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Luma([255 * (1_u32 - checkerboard[[y as usize, x as usize]])]);
        }

        let image = image::imageops::resize(&image, 1200, 1200, FilterType::Nearest);

        /*
        // NB free the written image.
        if false {
            image
                .save("checkerboard.png")
                .expect("Failed to save image");
        }
        */
    }

    #[test]
    fn test_max_flow_binary_image_map_graph() {
        let N = 2;
        let sampling = 2;
        let error_rate = 0.25;

        let checkerboard = get_checkerboard_fixture(N, sampling, error_rate);
        let exp_pixel_count = checkerboard.len();

        let (nodes, graph) = binary_image_map_graph(checkerboard);
        let (source, sink) = (nodes[0], nodes[nodes.len() - 1]);

        // println!("{:?}", map_graph);

        assert_eq!(1 + exp_pixel_count + 1, graph.node_count());

        let (max_flow, max_flow_on_edges) = petgraph_ford_fulkerson(&graph, source, sink);

        println!("{:?}", max_flow_on_edges);
    }
}
