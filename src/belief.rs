use rustc_hash::FxHashMap as HashMap;
use rand::thread_rng;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Latent,
    Observed,
}

pub struct Variable {
    pub id: usize,
    pub domain: usize,
    pub var_type: VariableType,
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
    pub variables: Vec<usize>, // variable ids
    pub table: Vec<f64>,       // flattened table, row-major, probabilities for all clique assignments.
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
    for _ in 0..max_iters {
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
                                        
                    let domains: Vec<usize> = fvars.iter()
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

        let max_diff = new_messages.iter()
            .map(|(k, &v)| (v - messages.get(k).copied().unwrap_or(0.0)).abs())
            .fold(0.0, f64::max);

        if max_diff < tol {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultrametric_binary_tree_belief_propagation() {
        let ncolor = 5;
        let nleaves = 50;

        // NB  assumes ultrametric (fully balanced) binary tree, N=(2n -1).
        let nancestors = nleaves - 1;

        let leaves: Vec<Variable> = (0..nleaves)
            .map(|id| Variable { id, domain: ncolor, var_type: VariableType::Observed })
            .collect();

        let ancestors: Vec<Variable> = (0..nancestors)
            .map(|id| Variable { id: id + nleaves, domain: ncolor, var_type: VariableType::Latent })
            .collect();

        let variables: Vec<Variable> = leaves.into_iter().chain(ancestors).collect();

        // H: emission weights for each leaf
        let H = random_one_hot_H(nleaves, ncolor);

        // Emission factors for each leaf: exp(H.s)
        let mut emission_factors = Vec::new();

        for h in &H {
            let table: Vec<f64> = (0..ncolor).map(|si| h[si].exp()).collect();
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

        let fg = FactorGraph {
            variables,
            factors,
            var_to_factors,
            factor_to_vars,
        };

        // NB converges in at most the diameter == 2 log2 n iterations for a balanced BT.
        let max_iters = (2.0 * (nleaves as f64).log2()).ceil() as usize;
        let tol = 1e-6;
        let beta = None;

        /*
        let marginals = ls_belief_propagation(&fg, max_iters, tol, beta);

        for marginal in marginals {
            let sum: f64 = marginal.iter().sum();
            assert!((sum - 1.0).abs() < 1e-8, "Marginal not normalized: {:?}", marginal);
        }
        */
    }
}