use rand::thread_rng;
use rand::Rng;
use rustc_hash::FxHashMap as HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Latent,
    Observed,
}

#[derive(Clone)]
pub struct Variable {
    pub id: usize,
    pub domain: usize,
    pub var_type: VariableType,
    pub pos: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType {
    Emission,
    Transition,
    Start_Prior,
    Custom,
}

pub struct Factor {
    pub id: usize,
    pub variables: Vec<usize>,   // variable ids
    pub table: Vec<f64>, // flattened table, row-major, probabilities for all clique assignments.
    pub factor_type: FactorType, // label for the type of factor
}

/// NB represents a bipartite graph of variables < -- > factors.
pub struct FactorGraph {
    pub variables: Vec<Variable>,
    pub factors: Vec<Factor>,
    pub var_to_factors: HashMap<usize, Vec<usize>>, // variable id -> factor ids
    pub factor_to_vars: HashMap<usize, Vec<usize>>, // factor id -> variable ids
}

fn next_assignment(assignment: &mut [usize], domains: &[usize], skip: usize) -> bool {
    for (j, dom) in domains.iter().enumerate() {
        if j == skip {
            continue;
        }
        assignment[j] += 1;

        if assignment[j] < *dom {
            return true;
        } else {
            assignment[j] = 0;
        }
    }

    false
}

/// Message from variable to factor or vice versa.
/// Indexed by (from, to, assignment).
type Message = HashMap<(usize, usize, usize), f64>;

/// Returns a vector of marginal distributions for each variable in the factor graph.
/// Each entry is a vector of probabilities (length = variable domain size) representing
/// the estimated marginal probability of each assignment for that variable after belief propagation.
pub fn ls_belief_propagation(
    fg: &FactorGraph,
    max_iters: usize,
    tol: f64,
    beta: Option<f64>,
) -> Vec<Vec<f64>> {
    let beta = beta.unwrap_or(1.0);

    println!("Solving belief propagation");

    let mut messages: Message = HashMap::default();

    // NB initialize var -> factor as 1/domain size.
    for var in &fg.variables {
        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..var.domain {
                messages.insert((var.id, fid, s), 1.0 / var.domain as f64);
            }
        }
    }

    for factor in &fg.factors {
        for &vid in &factor.variables {
            let vdom = fg.variables.iter().find(|v| v.id == vid).unwrap().domain;

            for s in 0..vdom {
                messages.insert((factor.id, vid, s), 1.0 / vdom as f64);
            }
        }
    }

    // NB converges in t*, diameter of the tree (max. node to node distance),
    //    i.e. 2log_2 num. leaves for a fully balanced (ultrametric) binary tree.
    for iter in 0..max_iters {
        let mut new_messages = messages.clone();

        // NB  passes on incoming messages to var (except output factor),
        //	   see eqn. (14.14) of Information, Physics & Computation, Mezard.
        for var in &fg.variables {
            for &fid in fg.var_to_factors.get(&var.id).unwrap() {
                let vdom = var.domain;

                for s in 0..vdom {
                    let mut prod = 1.0;

                    for &other_fid in fg.var_to_factors.get(&var.id).unwrap() {
                        if other_fid != fid {
                            prod *= messages[&(other_fid, var.id, s)];
                        }
                    }

                    new_messages.insert((var.id, fid, s), prod);
                }
            }
        }

        // NB passes on incoming messages to factor (except output variable),
        //    weighted by factor marginalized over all other variables,
        //    see eqn. (14.15) of Information, Physics & Computation, Mezard.
        for factor in &fg.factors {
            let fvars = &factor.variables;
            let ftable = &factor.table;

            for (i, &vid) in fvars.iter().enumerate() {
                let vdom = fg.variables.iter().find(|v| v.id == vid).unwrap().domain;

                for s in 0..vdom {
                    // NB sum over all assignments to other variables in the factor
                    let mut sum = 0.0;
                    let num_vars = fvars.len();

                    let mut assignment = vec![0; num_vars];

                    let domains: Vec<usize> = fvars
                        .iter()
                        .map(|vid| fg.variables.iter().find(|v| v.id == *vid).unwrap().domain)
                        .collect();

                    assignment[i] = s;

                    loop {
                        // Compute index into factor table
                        let mut idx = 0;
                        let mut stride = 1;

                        // NB row-major: last index first.
                        for (j, &a) in assignment.iter().rev().enumerate() {
                            idx += a * stride;
                            stride *= domains[domains.len() - 1 - j];
                        }

                        let mut prod = 1.0;

                        for (j, &other_vid) in fvars.iter().enumerate() {
                            if j != i {
                                prod *= messages[&(other_vid, factor.id, assignment[j])];
                            }
                        }

                        sum += ftable[idx].powf(beta) * prod;

                        if !next_assignment(&mut assignment, &domains, i) {
                            break;
                        }
                    }

                    new_messages.insert((factor.id, vid, s), sum);
                }
            }
        }

        // NB messages are probability distributions, normalize.
        for ((from, to, _), _) in new_messages.clone().iter() {
            let vdom = if let Some(var) = fg.variables.iter().find(|v| v.id == *from) {
                var.domain
            } else {
                fg.variables.iter().find(|v| v.id == *to).unwrap().domain
            };

            let norm: f64 = (0..vdom).map(|s| new_messages[&(*from, *to, s)]).sum();

            for s in 0..vdom {
                let val = new_messages[&(*from, *to, s)] / norm;
                new_messages.insert((*from, *to, s), val);
            }
        }

        let max_diff = new_messages
            .iter()
            .map(|(k, &v)| (v - messages.get(k).copied().unwrap_or(0.0)).abs())
            .fold(0.0, f64::max);

        println!("Belief propagation iteration {iter}: max_diff={max_diff:.3e}");

        if max_diff < tol {
            println!("Converged at iteration {iter} with tolerance {max_diff:.3e}");

            break;
        }

        messages = new_messages;
    }

    // NB compute marginals for each variable: product of incoming messages.
    let mut marginals = Vec::new();

    for var in &fg.variables {
        let vdom = var.domain;
        let mut marginal = vec![1.0; vdom];

        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..vdom {
                marginal[s] *= messages[&(fid, var.id, s)];
            }
        }

        let norm: f64 = marginal.iter().sum();

        for s in 0..vdom {
            marginal[s] /= norm;
        }

        marginals.push(marginal);
    }

    marginals
}

pub fn random_one_hot_H(nleaves: usize, ncolor: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    (0..nleaves)
        .map(|_| {
            let mut row = vec![0.0; ncolor];
            let idx = rng.gen_range(0..ncolor);
            row[idx] = 1.0;
            row
        })
        .collect()
}

pub fn linear_one_hot_H(nleaves: usize, ncolor: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();

    let weights: Vec<f64> = (1..=ncolor).map(|i| i as f64).collect();
    let total: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();

    (0..nleaves)
        .map(|_| {
            let mut r = rng.gen::<f64>();
            let mut idx = 0;

            for (i, &p) in probs.iter().enumerate() {
                if r < p {
                    idx = i;
                    break;
                }

                r -= p;
            }

            let mut row = vec![0.0; ncolor];

            row[idx] = 1.0;
            row
        })
        .collect()
}

pub fn random_normalized_H(nleaves: usize, ncolor: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();

    (0..nleaves)
        .map(|_| {
            let mut row: Vec<f64> = (0..ncolor).map(|_| rng.gen::<f64>()).collect();
            let norm: f64 = row.iter().sum();
            if norm > 0.0 {
                for v in &mut row {
                    *v /= norm;
                }
            }
            row
        })
        .collect()
}

fn felsensteins(
    nleaves: usize,
    nancestors: usize,
    ncolor: usize,
    emission_factors: &[Vec<f64>],
    pairwise_table: &[f64],
) -> Vec<Vec<f64>> {
    let nnodes = nleaves + nancestors;

    let mut likelihoods = vec![vec![1.0; ncolor]; nnodes];

    for leaf in 0..nleaves {
        likelihoods[leaf] = emission_factors[leaf].clone();
    }

    for p in nleaves..nnodes {
        let left = 2 * (p - nleaves);
        let right = 2 * (p - nleaves) + 1;

        let mut lk = vec![0.0; ncolor];

        for parent_state in 0..ncolor {
            let mut left_sum = 0.0;
            let mut right_sum = 0.0;

            for child_state in 0..ncolor {
                let trans = pairwise_table[parent_state * ncolor + child_state];

                left_sum += trans * likelihoods[left][child_state];
                right_sum += trans * likelihoods[right][child_state];
            }

            lk[parent_state] = left_sum * right_sum;
        }

        likelihoods[p] = lk;
    }

    // Downward pass: outs[node][state]
    let mut outs = vec![vec![1.0; ncolor]; nnodes];

    let root = nnodes - 1;
    // outs[root] is all 1.0 (no parent)
    // Traverse from root downward
    for p in nleaves..nnodes {
        let left = 2 * (p - nleaves);
        let right = 2 * (p - nleaves) + 1;

        if left >= nnodes {
            continue;
        }
        // For each child, compute outs[child][child_state]
        for &child in &[left, right] {
            for child_state in 0..ncolor {
                let mut sum = 0.0;

                for parent_state in 0..ncolor {
                    let trans = pairwise_table[parent_state * ncolor + child_state];

                    // For the sibling, use the upward message
                    let sibling = if child == left { right } else { left };

                    sum += outs[p][parent_state] * trans * likelihoods[sibling][parent_state];
                }

                outs[child][child_state] = sum;
            }
        }
    }

    let mut marginals = vec![vec![0.0; ncolor]; nnodes];

    for node in 0..nnodes {
        for state in 0..ncolor {
            marginals[node][state] = likelihoods[node][state] * outs[node][state];
        }

        let norm: f64 = marginals[node].iter().sum();

        if norm > 0.0 {
            for v in &mut marginals[node] {
                *v /= norm;
            }
        }
    }

    marginals
}

pub fn compute_tree_positions(nleaves: usize, nancestors: usize) -> Vec<(f64, f64)> {
    let nnodes = nleaves + nancestors;
    let mut pos = vec![(0.0, 0.0); nnodes];

    // Leaves: evenly spaced along x at y=0
    for i in 0..nleaves {
        pos[i] = (i as f64, 0.0);
    }

    // Ancestors: recursively place at the midpoint of their children, y increases with depth
    let mut depth = 1;
    let mut nodes_in_level = nleaves;

    // NB first parent
    let mut start_idx = nleaves;

    while start_idx < nnodes {
        let parents_in_level = nodes_in_level / 2;

        for i in 0..parents_in_level {
            let left = 2 * (start_idx + i - nleaves);
            let right = left + 1;

            let parent = start_idx + i;

            let x = (pos[left].0 + pos[right].0) / 2.0;
            let y = depth as f64;

            pos[parent] = (x, y);
        }

        nodes_in_level /= 2;

        start_idx += parents_in_level;

        depth += 1;
    }

    pos
}

pub fn save_node_marginals(
    filename: &str,
    variables: &[Variable],
    marginals: &[Vec<f64>],
    felsenstein: &[Vec<f64>],
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# id,x,y,bp_marginal,felsenstein_marginal")?;

    for (var, (bp, fel)) in variables
        .iter()
        .zip(marginals.iter().zip(felsenstein.iter()))
    {
        let (x, y) = var.pos.unwrap_or((f64::NAN, f64::NAN));

        writeln!(writer, "{}\t{}\t{}\t{:?}\t{:?}", var.id, x, y, bp, fel)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultrametric_binary_tree_belief_propagation() {
        let ncolor = 3;
        let nleaves = 16;

        env_logger::init();

        // NB  assumes ultrametric (fully balanced) binary tree, N=(2n -1).
        let nancestors = nleaves - 1;

        let leaves: Vec<Variable> = (0..nleaves)
            .map(|id| Variable {
                id,
                domain: ncolor,
                var_type: VariableType::Observed,
                pos: None,
            })
            .collect();

        let ancestors: Vec<Variable> = (0..nancestors)
            .map(|id| Variable {
                id: id + nleaves,
                domain: ncolor,
                var_type: VariableType::Latent,
                pos: None,
            })
            .collect();

        let mut variables: Vec<Variable> = leaves.into_iter().chain(ancestors).collect();

        println!("Solving for node positions.");

        let pos = compute_tree_positions(nleaves, nancestors);

        for (i, v) in variables.iter_mut().enumerate() {
            v.pos = Some(pos[i]);
        }

        // H: emission weights for each leaf
        let H = linear_one_hot_H(nleaves, ncolor);

        // Emission factors for each leaf: exp(H.s)
        let mut emission_factors = Vec::new();

        for h in &H {
            let mut table: Vec<f64> = (0..ncolor).map(|si| h[si].exp()).collect();

            let norm: f64 = table.iter().sum();

            for v in &mut table {
                *v /= norm;
            }

            // println!("{:?}", table);

            emission_factors.push(table);
        }

        // Factors: one for each leaf emission, and internal soft constraints.
        let mut factors = Vec::new();

        // TODO avoid cloning table.
        for (leaf_idx, table) in emission_factors.iter().enumerate() {
            factors.push(Factor {
                id: leaf_idx,
                variables: vec![leaf_idx], // vector of variable ids.
                table: table.clone(),
                factor_type: FactorType::Emission,
            });
        }

        // var_to_factors: map each variable to its emission factor(s)
        let mut var_to_factors: HashMap<usize, Vec<usize>> = HashMap::default();

        for leaf_idx in 0..nleaves {
            var_to_factors.insert(leaf_idx, vec![leaf_idx]);
        }

        let mut factor_to_vars: HashMap<usize, Vec<usize>> = HashMap::default();

        for (leaf_idx, factor) in factors.iter().enumerate() {
            factor_to_vars.insert(leaf_idx, factor.variables.clone());
        }

        let pairwise_strength = 5.0;
        let mut pairwise_table = vec![0.0; ncolor * ncolor];

        //  TODO normalized?
        for i in 0..ncolor {
            for j in 0..ncolor {
                pairwise_table[i * ncolor + j] = if i == j { pairwise_strength } else { 1.0 };
            }
        }

        // Normalize each row of the pairwise_table to sum to 1
        for i in 0..ncolor {
            let row_start = i * ncolor;
            let row_end = row_start + ncolor;
            let row_sum: f64 = pairwise_table[row_start..row_end].iter().sum();

            if row_sum > 0.0 {
                for j in row_start..row_end {
                    pairwise_table[j] /= row_sum;
                }
            }
        }

        let mut next_factor_id = nleaves;

        // For a full binary tree, parent indices: nleaves..(nleaves + nancestors)
        // Children of parent p: left = 2*(p-nleaves)+0, right = 2*(p-nleaves)+1
        for p in 0..nancestors {
            let parent_id = nleaves + p;

            let left_child = 2 * p;
            let right_child = 2 * p + 1;

            factors.push(Factor {
                id: next_factor_id,
                variables: vec![parent_id, left_child],
                table: pairwise_table.clone(),
                factor_type: FactorType::Transition,
            });

            var_to_factors
                .entry(parent_id)
                .or_default()
                .push(next_factor_id);

            var_to_factors
                .entry(left_child)
                .or_default()
                .push(next_factor_id);

            factor_to_vars.insert(next_factor_id, vec![parent_id, left_child]);

            next_factor_id += 1;

            factors.push(Factor {
                id: next_factor_id,
                variables: vec![parent_id, right_child],
                table: pairwise_table.clone(),
                factor_type: FactorType::Transition,
            });

            var_to_factors
                .entry(parent_id)
                .or_default()
                .push(next_factor_id);

            var_to_factors
                .entry(right_child)
                .or_default()
                .push(next_factor_id);

            factor_to_vars.insert(next_factor_id, vec![parent_id, right_child]);

            next_factor_id += 1;
        }

        let fg = FactorGraph {
            variables: variables.clone(),
            factors,
            var_to_factors,
            factor_to_vars,
        };

        // NB converges in at most the diameter == 2 log2 n iterations for a balanced BT.
        let mut max_iters = (2.0 * (nleaves as f64).log2()).ceil() as usize;

        let tol = 1e-6;
        let beta: Option<f64> = None;

        let marginals = ls_belief_propagation(&fg, max_iters, tol, beta);

        let exp = felsensteins(
            nleaves,
            nancestors,
            ncolor,
            &emission_factors,
            &pairwise_table,
        );

        println!(
            "{:>10} {:>30} {:>30}",
            "Leaf Index", "BP Marginal", "Felsenstein"
        );

        for i in 0..10 {
            print!("{:>10} ", i);

            for j in 0..ncolor {
                print!("{:>10.6} ", marginals[i][j]);
            }
            println!();
            print!("{:>10} ", "");

            for j in 0..ncolor {
                print!("{:>10.6} ", exp[i][j]);
            }

            println!("\n");
        }

        println!(
            "{:>10} {:>30} {:>30}",
            "Latent Index", "BP Marginal", "Felsenstein"
        );

        for i in nleaves..nleaves + 10 {
            print!("{:>10} ", i);

            for j in 0..ncolor {
                print!("{:>10.6} ", marginals[i][j]);
            }

            println!();

            print!("{:>10} ", "");

            for j in 0..ncolor {
                print!("{:>10.6} ", exp[i][j]);
            }

            println!("\n");
        }

        println!(
            "{:>10} {:>30} {:>30}",
            "Latent Index", "BP Marginal", "Felsenstein"
        );

        // TODO why is the root uniform?
        let start = marginals.len().saturating_sub(10);
        for i in start..marginals.len() {
            print!("{:>10} ", i);

            for j in 0..ncolor {
                print!("{:>10.6} ", marginals[i][j]);
            }
            println!();
            print!("{:>10} ", "");

            for j in 0..ncolor {
                print!("{:>10.6} ", exp[i][j]);
            }

            println!("\n");
        }

        save_node_marginals("data/node_marginals.csv", &variables, &marginals, &exp).unwrap();
    }
}
