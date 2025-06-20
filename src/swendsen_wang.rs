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
    pub h: Vec<Vec<f64>>,        // External field, shape: nnode x ncolor
    pub j: f64,                  // Max Potts coupling
    pub edges: Vec<Edge>,        // List of edges
}

impl PottsLattice {
    pub fn new(
        width: usize,
        height: usize,
        ncolor: usize,
        h: Vec<Vec<f64>>,
        j: f64,
    ) -> Self {
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

        // External field contribution
        for (i, &color) in assignment.iter().enumerate() {
            total_cost += self.h[i][color] * beta;
        }

        // Coupling contribution
        for edge in &self.edges {
            if assignment[edge.from] != assignment[edge.to] {
                total_cost += beta * self.j * edge.weight;
            }
        }

        total_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_field(nnode: usize, ncolor: usize, max_h: f64) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        (0..nnode)
        .map(|_| {
            (0..ncolor)
                .map(|_| rng.gen::<f64>() * max_h)
                .collect()
        })
        .collect()
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