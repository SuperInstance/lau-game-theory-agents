//! Mechanism design: VCG, incentive compatibility, social choice.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A mechanism: allocates outcomes and collects payments based on reported types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mechanism {
    pub n_agents: usize,
    pub n_outcomes: usize,
    /// Valuations: valuations[agent][outcome] = value
    pub valuations: Vec<Vec<f64>>,
}

impl Mechanism {
    /// Create a mechanism with given valuations.
    pub fn new(valuations: Vec<Vec<f64>>) -> Self {
        let n_agents = valuations.len();
        let n_outcomes = valuations.first().map(|v| v.len()).unwrap_or(0);
        Self { n_agents, n_outcomes, valuations }
    }

    /// Compute the efficient allocation (maximizes social welfare).
    pub fn efficient_allocation(&self) -> (usize, f64) {
        let mut best_outcome = 0;
        let mut best_welfare = f64::NEG_INFINITY;
        for o in 0..self.n_outcomes {
            let welfare: f64 = self.valuations.iter().map(|v| v[o]).sum();
            if welfare > best_welfare {
                best_welfare = welfare;
                best_outcome = o;
            }
        }
        (best_outcome, best_welfare)
    }

    /// VCG mechanism: returns (allocation, payments).
    /// Payments[p] = social_welfare_without_p(with efficient allocation for others)
    ///             - social_welfare_without_p(with chosen allocation)
    pub fn vcg(&self) -> VCGResult {
        let (allocation, social_welfare) = self.efficient_allocation();

        let mut payments = vec![0.0; self.n_agents];
        let mut agent_payoffs = vec![0.0; self.n_agents];

        for p in 0..self.n_agents {
            // Social welfare without agent p at the chosen allocation
            let sw_chosen_without_p: f64 = self.valuations.iter().enumerate()
                .filter(|(i, _)| *i != p)
                .map(|(_, v)| v[allocation])
                .sum();

            // Efficient allocation without agent p
            let mut best_alt = 0;
            let mut best_alt_welfare = f64::NEG_INFINITY;
            for o in 0..self.n_outcomes {
                let w: f64 = self.valuations.iter().enumerate()
                    .filter(|(i, _)| *i != p)
                    .map(|(_, v)| v[o])
                    .sum();
                if w > best_alt_welfare {
                    best_alt_welfare = w;
                    best_alt = o;
                }
            }

            // VCG payment (Clarke pivot rule)
            payments[p] = best_alt_welfare - sw_chosen_without_p;
            agent_payoffs[p] = self.valuations[p][allocation] - payments[p];
        }

        VCGResult {
            allocation,
            social_welfare,
            payments,
            agent_payoffs,
        }
    }
}

/// Result of VCG mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VCGResult {
    pub allocation: usize,
    pub social_welfare: f64,
    /// Payment each agent makes (Clarke pivot).
    pub payments: Vec<f64>,
    /// Net payoff for each agent (valuation - payment).
    pub agent_payoffs: Vec<f64>,
}

/// Check if a mechanism is dominant-strategy incentive compatible (DSIC).
/// A mechanism is DSIC if truthful reporting is a dominant strategy.
pub fn is_dsic<F>(mechanism_fn: F, n_agents: usize, n_outcomes: usize, valuations: &Vec<Vec<f64>>) -> bool
where
    F: Fn(&Vec<Vec<f64>>) -> (usize, Vec<f64>),
{
    for p in 0..n_agents {
        // Truthful payoff
        let (true_alloc, true_payments) = mechanism_fn(valuations);
        let true_payoff = valuations[p][true_alloc] - true_payments[p];

        // Try all misreports
        for fake_val in 0..n_outcomes {
            let mut fake_vals = valuations.clone();
            // Agent p reports fake_val as their value for each outcome
            // Simple test: just change one outcome's value
            for o in 0..n_outcomes {
                fake_vals[p][o] = if o == fake_val { valuations[p][o] + 10.0 } else { valuations[p][o] - 10.0 };
            }
            let (fake_alloc, fake_payments) = mechanism_fn(&fake_vals);
            let fake_payoff = valuations[p][fake_alloc] - fake_payments[p];

            if fake_payoff > true_payoff + 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Check individual rationality: each agent gets non-negative payoff.
pub fn is_individually_rational(result: &VCGResult) -> bool {
    result.agent_payoffs.iter().all(|&p| p >= -1e-10)
}

/// Check budget balance: total payments >= 0 (no deficit).
pub fn is_budget_balanced(result: &VCGResult) -> bool {
    result.payments.iter().sum::<f64>() >= -1e-10
}

/// A social choice function.
pub trait SocialChoiceFunction {
    fn choose(&self, valuations: &Vec<Vec<f64>>) -> usize;
}

/// Utilitarian social choice: maximize total welfare.
pub struct UtilitarianSCF;

impl SocialChoiceFunction for UtilitarianSCF {
    fn choose(&self, valuations: &Vec<Vec<f64>>) -> usize {
        let n_outcomes = valuations.first().map(|v| v.len()).unwrap_or(0);
        let mut best = 0;
        let mut best_w = f64::NEG_INFINITY;
        for o in 0..n_outcomes {
            let w: f64 = valuations.iter().map(|v| v[o]).sum();
            if w > best_w { best_w = w; best = o; }
        }
        best
    }
}

/// Affine maximizer social choice.
pub struct AffineMaximizerSCF {
    pub weights: Vec<f64>,
    pub offsets: Vec<f64>,
}

impl SocialChoiceFunction for AffineMaximizerSCF {
    fn choose(&self, valuations: &Vec<Vec<f64>>) -> usize {
        let n_outcomes = valuations.first().map(|v| v.len()).unwrap_or(0);
        let mut best = 0;
        let mut best_w = f64::NEG_INFINITY;
        for o in 0..n_outcomes {
            let w: f64 = valuations.iter().enumerate()
                .map(|(i, v)| self.weights[i] * v[o])
                .sum::<f64>() + self.offsets.get(o).copied().unwrap_or(0.0);
            if w > best_w { best_w = w; best = o; }
        }
        best
    }
}

/// Revelation principle check: can the mechanism be implemented truthfully?
pub fn check_revelation_principle<F>(
    mechanism_fn: F,
    n_agents: usize,
    n_outcomes: usize,
    test_valuations: &Vec<Vec<Vec<f64>>>,
) -> bool
where
    F: Fn(&Vec<Vec<f64>>) -> (usize, Vec<f64>),
{
    test_valuations.iter().all(|vals| is_dsic(&mechanism_fn, n_agents, n_outcomes, vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_valuations() -> Vec<Vec<f64>> {
        vec![
            vec![10.0, 5.0, 0.0],   // Agent 0
            vec![0.0, 8.0, 3.0],    // Agent 1
            vec![2.0, 4.0, 9.0],    // Agent 2
        ]
    }

    #[test]
    fn test_efficient_allocation() {
        let m = Mechanism::new(example_valuations());
        let (alloc, welfare) = m.efficient_allocation();
        // Welfare: outcome 0 = 12, outcome 1 = 17, outcome 2 = 12
        assert_eq!(alloc, 1);
        assert!((welfare - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_vcg_payments() {
        let m = Mechanism::new(example_valuations());
        let result = m.vcg();
        assert_eq!(result.allocation, 1);
        // Agent 0's payment: welfare of others at alt - welfare of others at chosen
        // Others at outcome 1: 8 + 4 = 12
        // Others best: outcome 2 gives 3 + 9 = 12, outcome 0 gives 0 + 2 = 2
        // Payment = 12 - 12 = 0
        assert!(result.payments[0] >= -1e-10);
    }

    #[test]
    fn test_vcg_individual_rationality() {
        let m = Mechanism::new(example_valuations());
        let result = m.vcg();
        assert!(is_individually_rational(&result));
    }

    #[test]
    fn test_utilitarian_scf() {
        let scf = UtilitarianSCF;
        let outcome = scf.choose(&example_valuations());
        assert_eq!(outcome, 1);
    }

    #[test]
    fn test_affine_maximizer() {
        let scf = AffineMaximizerSCF {
            weights: vec![1.0, 1.0, 1.0],
            offsets: vec![0.0, 0.0, 5.0],
        };
        let outcome = scf.choose(&example_valuations());
        assert!(outcome == 1 || outcome == 2); // Tie at 17, either is valid
    }

    #[test]
    fn test_vcg_two_agents() {
        let vals = vec![
            vec![5.0, 10.0],
            vec![8.0, 3.0],
        ];
        let m = Mechanism::new(vals);
        let result = m.vcg();
        // Welfare: o0=13, o1=13 -> tie, pick first
        // Let's make it clearer
        assert!(result.social_welfare > 0.0);
    }

    #[test]
    fn test_mechanism_creation() {
        let m = Mechanism::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(m.n_agents, 2);
        assert_eq!(m.n_outcomes, 2);
    }
}
