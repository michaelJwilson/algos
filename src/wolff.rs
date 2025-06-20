use std::collections::HashMap;

use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::IteratorRandom;
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct PottsLattice {
    pub width: usize,
    pub height: usize,
    pub ncolor: usize,
    pub h: Vec<Vec<f64>>, // External field, shape: (nnode x ncolor)
    pub j: f64,           // Max Potts coupling, assuming edges normalized to max. of unity.
    pub edges: Vec<Edge>,
}

impl PottsLattice {
    pub fn new(width: usize, height: usize, ncolor: usize, h: Vec<Vec<f64>>, j: f64) -> Self {
        let mut edges = Vec::new();
        let idx = |x, y| y * width + x;
        let mut rng = rand::thread_rng();
        for y in 0..height {
            for x in 0..width {
                let site = idx(x, y);

                if x + 1 < width {
                    edges.push(Edge {
                        from: site,
                        to: idx(x + 1, y),
                        weight: rng.gen(),
                    });
                }

                if y + 1 < height {
                    edges.push(Edge {
                        from: site,
                        to: idx(x, y + 1),
                        weight: rng.gen(),
                    });
                }
            }
        }

        Self {
            width,
            height,
            ncolor,
            h,
            j,
            edges,
        }
    }

    pub fn weighted_random<R: Rng>(
        &self,
        rng: &mut R,
        weights: &[f64],
        exclude: Option<usize>,
    ) -> (usize, f64) {
        let mut new_weights: Vec<f64> = weights.to_vec();

        if let Some(ex) = exclude {
            new_weights[ex] = 0.0;
        }

        let sum: f64 = new_weights.iter().sum();

        let new_weights = new_weights.iter().map(|&w| w / sum).collect::<Vec<f64>>();

        let dist =
            WeightedIndex::new(new_weights).expect("Weights must be non-negative and not all zero");

        let sample = dist.sample(rng);

        (sample, new_weights[sample])
    }

    pub fn potts_cost(&self, assignment: &[usize], beta: f64) -> f64 {
        let mut total_cost = 0.0;

        for (i, &color) in assignment.iter().enumerate() {
            total_cost += self.h[i][color] * beta;
        }

        for edge in &self.edges {
            if assignment[edge.from] != assignment[edge.to] {
                total_cost += beta * self.j * edge.weight;
            }
        }

        total_cost
    }

    pub fn wolff_step<R: Rng>(&self, assignment: &mut [usize], beta: f64, rng: &mut R) {
        let nnode = self.width * self.height;

        let seed = rng.random_range(0..nnode);
        let seed_color = assignment[seed];

        let mut in_cluster = vec![false; nnode];
        let mut queue = std::collections::VecDeque::new();

        in_cluster[seed] = true;
        queue.push_back(seed);

        // NB  Wolff is designed such that acceptance according to edge updates is unity.
        while let Some(site) = queue.pop_front() {
            for edge in self.edges.iter().filter(|e| e.from == site || e.to == site) {
                let neighbor = if edge.from == site {
                    edge.to
                } else {
                    edge.from
                };

                if !in_cluster[neighbor] && assignment[neighbor] == seed_color {
                    let p: f64 = 1.0 - (-beta * self.j * edge.weight).exp();

                    if rng.random::<f64>() < p {
                        in_cluster[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        let mut possible_colors: Vec<usize> =
            (0..self.ncolor).filter(|&c| c != seed_color).collect();

        let &new_color = possible_colors.choose(rng).unwrap();

        // NB iterate over nodes and calculate the product of (-beta * (h[new_color] - h[seed_color])).exp()
        let acceptance: f64 = self
            .h
            .iter()
            .map(|h_c| (-beta * (h_c[new_color] - h_c[seed_color])).exp())
            .product();

        // NB we can update the assignment with the new color.
        if rng.random()::<f64>() < acceptance {
            for (i, &inc) in in_cluster.iter().enumerate() {
                if inc {
                    assignment[i] = new_color;
                }
            }
        }
    }

    pub fn sample_node_color(
        node: usize,
        assignment: &mut [usize],
        rng: &mut R,
        exclude: Option<usize>,
    ) {
        let node_color: usize = assignment[node];

        let node_field_energy: Vec<f64> = self.h[node]
            .iter()
            .map(|&h_c| h_c - self.h[node][node_color])
            .collect();

        let weights = node_field_energy
            .iter()
            .map(|&E| (-beta * E).exp())
            .collect::<Vec<f64>>();

        let (new_color, prob) = self::weighted_random(rng, weights, exclude);

        (new_color, prob)
    }

    pub fn tree_step<R: Rng>(&self, assignment: &mut [usize], beta: f64, rng: &mut R) {
        let nnode: usize = self.width * self.height;

        // NB define the seed
        let root: usize = rng.random_range(0..nnode);
        let root_color: usize = assignment[root];

        let mut in_cluster: Vec<bool> = vec![false; nnode];
        let mut queue = std::collections::VecDeque::new();

        in_cluster[seed] = true;
        queue.push_back(seed);

        let (new_root_color, _) = self.sample_node_color(root, assignment, rng);

        let new_assignment: HashMap<usize, usize> = HashMap::new();
        new_assignment.insert(root, new_root_color);

        // NB by proposing a new root color according to Boltzmann, acceptance remains unity.
        let acceptance: f64 = 1.0;

        // NB when draw is greater than acceptance, we can reject.  I.e. we can stop building
        //    as soon as acceptance < draw, as it is monotonically decreasing.
        let draw: f64 = rng.random::<f64>();
        let valid: bool = true;

        'outer: while let Some(site) = queue.pop_front() {
            for edge in self.edges.iter().filter(|e| e.from == site || e.to == site) {
                let neighbor = if edge.from == site {
                    edge.to
                } else {
                    edge.from
                };

                // NB we remove loops by not considering (same color) children already in the cluster.
                //    However, nodes can have multiple attempts to join the cluster depending on the edge set.
                if !in_cluster[neighbor] && assignment[neighbor] == seed_color {
                    // NB by proposing a new bond according to Boltzmann, acceptance remains unity.
                    let pbond: f64 = 1.0 - (-beta * self.j * edge.weight).exp();

                    // NB sample new color and apply importance sampling.
                    if rng.random::<f64>() < pbond {
                        //  NB proposed move has unit probability.
                        let new_color: usize = assignment[site];
                    } else {
                        let (new_color, prob) =
                            self.sample_node_color(neighbor, assignment, rng, Some(seed_color));

                        acceptance /= prob;
                    }

                    acceptance *= (-beta
                        * (self.h[neighbor][new_color] - self.h[neighbor][seed_color]))
                        .exp();

                    in_cluster[neighbor] = true;
                    queue.push_back(neighbor);
                }

                if acceptance < draw {
                    // NB we can stop building the cluster as acceptance is monotonically decreasing.
                    valid = false;

                    // NB we can break out of (inner + ) outer loop.
                    break 'outer;
                }
            }
        }

        if valid {
            // NB we can update the assignment with the new colors.
            for (node, &new_color) in new_assignment.iter() {
                assignment[*node] = new_color;
            }
        }

        valid
    }
}