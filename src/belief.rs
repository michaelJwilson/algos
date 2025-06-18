use std::collections::HashMap;

/// Represents a variable node in the factor graph.
pub struct Variable {
    pub id: usize,
    pub domain: usize, // number of possible states
}

/// Represents a factor node in the factor graph.
pub struct Factor {
    pub id: usize,
    pub variables: Vec<usize>, // variable ids
    pub table: Vec<f64>,       // flattened table, row-major
}

/// Represents a bipartite factor graph.
pub struct FactorGraph {
    pub variables: Vec<Variable>,
    pub factors: Vec<Factor>,
    pub var_to_factors: HashMap<usize, Vec<usize>>, // variable id -> factor ids
    pub factor_to_vars: HashMap<usize, Vec<usize>>, // factor id -> variable ids
}

/// Message from variable to factor or vice versa.
/// Indexed by (from, to, assignment).
type Message = HashMap<(usize, usize, usize), f64>;

/// Lauritzen-Spiegelhalter (belief propagation) for a general factor graph.
/// Returns: marginal distributions for each variable (as Vec<Vec<f64>>)
pub fn ls_belief_propagation(
    fg: &FactorGraph,
    max_iters: usize,
    tol: f64,
) -> Vec<Vec<f64>> {
    let mut messages: Message = HashMap::new();

    // Initialize messages to uniform
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

    // Iterative message passing
    for _ in 0..max_iters {
        let mut new_messages = messages.clone();

        // Variable to factor messages
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

        // Factor to variable messages
        for factor in &fg.factors {
            let fvars = &factor.variables;
            let ftable = &factor.table;
            for (i, &vid) in fvars.iter().enumerate() {
                let vdom = fg.variables.iter().find(|v| v.id == vid).unwrap().domain;
                for s in 0..vdom {
                    // Sum over all assignments to other variables in the factor
                    let mut sum = 0.0;
                    let num_vars = fvars.len();
                    let mut assignment = vec![0; num_vars];
                    // Fix assignment[i] = s, sum over others
                    fn next_assignment(assignment: &mut [usize], domains: &[usize], skip: usize) -> bool {
                        for (j, dom) in domains.iter().enumerate() {
                            if j == skip { continue; }
                            assignment[j] += 1;
                            if assignment[j] < *dom {
                                return true;
                            } else {
                                assignment[j] = 0;
                            }
                        }
                        false
                    }
                    let domains: Vec<usize> = fvars.iter()
                        .map(|vid| fg.variables.iter().find(|v| v.id == *vid).unwrap().domain)
                        .collect();
                    assignment[i] = s;
                    loop {
                        // Compute index into factor table
                        let mut idx = 0;
                        let mut stride = 1;
                        for (j, &a) in assignment.iter().rev().enumerate() {
                            idx += a * stride;
                            stride *= domains[domains.len() - 1 - j];
                        }
                        // Product of incoming messages for other variables
                        let mut prod = 1.0;
                        for (j, &other_vid) in fvars.iter().enumerate() {
                            if j != i {
                                prod *= messages[&(other_vid, factor.id, assignment[j])];
                            }
                        }
                        sum += ftable[idx] * prod;
                        if !next_assignment(&mut assignment, &domains, i) {
                            break;
                        }
                    }
                    new_messages.insert((factor.id, vid, s), sum);
                }
            }
        }

        // Normalize messages
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

        // Check for convergence (optional, not implemented here)
        messages = new_messages;
    }

    // Compute marginals for each variable
    let mut marginals = Vec::new();
    for var in &fg.variables {
        let vdom = var.domain;
        let mut marginal = vec![1.0; vdom];
        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..vdom {
                marginal[s] *= messages[&(fid, var.id, s)];
            }
        }
        // Normalize
        let norm: f64 = marginal.iter().sum();
        for s in 0..vdom {
            marginal[s] /= norm;
        }
        marginals.push(marginal);
    }
    marginals
}