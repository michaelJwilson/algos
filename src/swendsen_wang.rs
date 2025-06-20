use rand::seq::SliceRandom;
use rand::Rng;
use rand::prelude::IteratorRandom;

pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    // NB initializes with n clusters.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        let mut root: usize = x;

        while self.parent[root] != root {
            root = self.parent[root];
        }

        // NB path compression / update parents.
        while self.parent[x] != x {
            let next: usize = self.parent[x];
            self.parent[x] = root;
            x = next;
        }

        root
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let xr: usize = self.find(x);
        let yr: usize = self.find(y);

        // NB joins right tree to left tree.
        if xr != yr {
            self.parent[yr] = xr;
        }
    }

    pub fn clusters(&mut self) -> Vec<Vec<usize>> {
        // NB returns a list of clusters, where each cluster is a vector of node indices.
        let n = self.parent.len();

        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            let root = self.find(i);

            clusters[root].push(i);
        }

        clusters.into_iter().filter(|c| !c.is_empty()).collect()
    }
}
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
    pub h: Vec<Vec<f64>>, // External field, shape: nnode x ncolor
    pub j: f64,           // Max Potts coupling
    pub edges: Vec<Edge>, // List of edges
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

    pub fn swendsen_wang_step<R: Rng>(&self, assignment: &mut [usize], beta: f64, rng: &mut R) {
        let nnode: usize = self.width * self.height;
        let mut uf: UnionFind = UnionFind::new(nnode);

        // 1. Bond formation
        for edge in &self.edges {
            if assignment[edge.from] == assignment[edge.to] {
                let p = 1.0 - (-beta * self.j * edge.weight).exp();

                // NB joins child node with cluster of parent.
                if rng.gen::<f64>() < p {
                    uf.union(edge.from, edge.to);
                }
            }
        }

        // 2. Cluster relabeling
        let clusters = uf.clusters();
        for cluster in clusters {
            // Choose a new color at random
            let new_color = (0..self.ncolor).choose(rng).unwrap();
            for &site in &cluster {
                assignment[site] = new_color;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_field(nnode: usize, ncolor: usize, max_h: f64) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        (0..nnode)
            .map(|_| (0..ncolor).map(|_| rng.gen::<f64>() * max_h).collect())
            .collect()
    }

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);

        // Initially, each element is its own parent
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }

        // Union some elements
        uf.union(0, 1);
        uf.union(3, 4);

        // Now 0 and 1 should have the same root, as should 3 and 4
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(3), uf.find(4));
        assert_ne!(uf.find(0), uf.find(3));

        // Union across groups
        uf.union(1, 3);

        // Now all 0,1,3,4 should have the same root
        let root = uf.find(0);
        for &i in &[1, 3, 4] {
            assert_eq!(uf.find(i), root);
        }

        // Only 2 should be separate
        assert_ne!(uf.find(2), root);
    }

    #[test]
    fn test_swendsen_wang() {
        let width = 8;
        let height = 8;
        let ncolor = 3;

        let J = 1.0;
        let beta = 2.0;

        // h: nnode x ncolor
        let h = random_field(width * height, ncolor, 5.);

        let lattice = PottsLattice::new(width, height, ncolor, h.clone(), J);

        // Total cost for a configuration
        let assignment = vec![0; width * height]; // All nodes assigned to color 0
        let total = lattice.potts_cost(&assignment, beta);

        println!("Total cost for all nodes assigned to color 0: {}", total);
    }
}
