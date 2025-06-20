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