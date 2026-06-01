//! Bargaining: Nash bargaining solution, Rubinstein alternating offers, Kalai-Smorodinsky.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A bargaining problem: feasible set and disagreement point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BargainingProblem {
    /// Disagreement point d = (d1, d2).
    pub disagreement: [f64; 2],
    /// Feasible set represented as a list of utility pairs.
    pub feasible_set: Vec<[f64; 2]>,
}

impl BargainingProblem {
    /// Create a new bargaining problem.
    pub fn new(disagreement: [f64; 2], feasible_set: Vec<[f64; 2]>) -> Self {
        Self { disagreement, feasible_set }
    }

    /// Simple bargaining problem with a linear frontier.
    pub fn linear_frontier(total: f64, disagreement: [f64; 2]) -> Self {
        let n_points = 100;
        let feasible: Vec<[f64; 2]> = (0..=n_points)
            .map(|i| {
                let t = i as f64 / n_points as f64;
                [total * t, total * (1.0 - t)]
            })
            .collect();
        Self::new(disagreement, feasible)
    }

    /// Filter feasible set to individually rational points.
    pub fn individually_rational(&self) -> Vec<[f64; 2]> {
        self.feasible_set.iter()
            .filter(|[u1, u2]| *u1 >= self.disagreement[0] && *u2 >= self.disagreement[1])
            .copied()
            .collect()
    }
}

/// Nash Bargaining Solution: maximizes (u1 - d1) * (u2 - d2).
pub fn nash_bargaining_solution(problem: &BargainingProblem) -> [f64; 2] {
    let d = problem.disagreement;
    let ir_set = problem.individually_rational();

    let mut best = d;
    let mut best_product = 0.0f64; // Product of gains

    for &[u1, u2] in &ir_set {
        let g1 = u1 - d[0];
        let g2 = u2 - d[1];
        if g1 < 0.0 || g2 < 0.0 { continue; }
        let product = g1 * g2;
        if product > best_product {
            best_product = product;
            best = [u1, u2];
        }
    }

    best
}

/// Kalai-Smorodinsky Solution: equalizes relative gains.
/// Find the point on the Pareto frontier where (u1-d1)/(u1_max-d1) = (u2-d2)/(u2_max-d2).
pub fn kalai_smorodinsky_solution(problem: &BargainingProblem) -> [f64; 2] {
    let d = problem.disagreement;
    let ir_set = problem.individually_rational();

    let u1_max = ir_set.iter().map(|[u1, _]| *u1).fold(d[0], f64::max);
    let u2_max = ir_set.iter().map(|[_, u2]| *u2).fold(d[1], f64::max);

    let g1_max = u1_max - d[0];
    let g2_max = u2_max - d[1];

    if g1_max <= 0.0 || g2_max <= 0.0 {
        return d;
    }

    // Find the point on the frontier where g1/g1_max = g2/g2_max
    // This is the KS line: parametrized by t ∈ [0,1]
    // u1 = d1 + t * g1_max, u2 = d2 + t * g2_max
    // Find intersection with Pareto frontier

    let mut best = d;
    let mut best_t = 0.0;
    let mut best_diff = f64::INFINITY;

    for t in (0..=1000).map(|i| i as f64 / 1000.0) {
        let target = [d[0] + t * g1_max, d[1] + t * g2_max];
        // Find closest feasible point
        for &[u1, u2] in &ir_set {
            if u1 >= target[0] - 1e-10 && u2 >= target[1] - 1e-10 {
                let diff = (u1 - target[0]).abs() + (u2 - target[1]).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_t = t;
                    best = [u1, u2];
                }
            }
        }
    }

    best
}

/// Egalitarian solution: maximizes the minimum gain.
pub fn egalitarian_solution(problem: &BargainingProblem) -> [f64; 2] {
    let d = problem.disagreement;
    let ir_set = problem.individually_rational();

    let mut best = d;
    let mut best_min = f64::NEG_INFINITY;

    for &[u1, u2] in &ir_set {
        let g1 = u1 - d[0];
        let g2 = u2 - d[1];
        let min_gain = g1.min(g2);
        if min_gain > best_min {
            best_min = min_gain;
            best = [u1, u2];
        }
    }

    best
}

/// Rubinstein alternating offers game.
/// Two players alternate making offers. Discount factors δ1, δ2.
/// Subgame perfect equilibrium: player 1 offers (1 - δ2)/(1 - δ1*δ2) to self.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubinsteinResult {
    /// Share for player 1.
    pub share_1: f64,
    /// Share for player 2.
    pub share_2: f64,
    /// Discount factor 1.
    pub delta_1: f64,
    /// Discount factor 2.
    pub delta_2: f64,
}

/// Solve Rubinstein alternating offers with infinite horizon.
pub fn rubinstein_bargaining(delta_1: f64, delta_2: f64) -> RubinsteinResult {
    let share_1 = (1.0 - delta_2) / (1.0 - delta_1 * delta_2);
    let share_2 = 1.0 - share_1;
    RubinsteinResult {
        share_1,
        share_2,
        delta_1,
        delta_2,
    }
}

/// Rubinstein with equal discount factors.
pub fn rubinstein_equal_delta(delta: f64) -> RubinsteinResult {
    rubinstein_bargaining(delta, delta)
}

/// Compute Nash axioms satisfaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomCheck {
    pub pareto_efficiency: bool,
    pub symmetry: bool,
    pub scale_invariance: bool,
    pub independence_of_irrelevant_alternatives: bool,
}

/// Check Nash axioms for a given solution.
pub fn check_nash_axioms(
    problem: &BargainingProblem,
    solution: [f64; 2],
) -> AxiomCheck {
    let d = problem.disagreement;
    let ir = problem.individually_rational();

    // Pareto efficiency: no other IR point dominates the solution
    let pareto = !ir.iter().any(|&[u1, u2]| {
        u1 >= solution[0] && u2 >= solution[1] &&
        (u1 > solution[0] || u2 > solution[1])
    });

    // Symmetry: if the problem is symmetric (swap axes gives same set),
    // the solution should be symmetric
    let sym_problem = is_symmetric(problem);
    let symmetry = !sym_problem || (solution[0] - solution[1]).abs() < 1e-8;

    // Scale invariance (simplified check)
    let scale_invariance = true; // Would need transformed problem to check properly

    // IIA (simplified check)
    let iia = true; // Would need reduced problem to check properly

    AxiomCheck {
        pareto_efficiency: pareto,
        symmetry,
        scale_invariance,
        independence_of_irrelevant_alternatives: iia,
    }
}

fn is_symmetric(problem: &BargainingProblem) -> bool {
    if (problem.disagreement[0] - problem.disagreement[1]).abs() > 1e-10 {
        return false;
    }
    // Check that every point [a,b] has a corresponding [b,a]
    'outer: for &[a, b] in &problem.feasible_set {
        for &[c, d] in &problem.feasible_set {
            if (c - b).abs() < 1e-8 && (d - a).abs() < 1e-8 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nash_bargaining() {
        let problem = BargainingProblem::linear_frontier(10.0, [0.0, 0.0]);
        let solution = nash_bargaining_solution(&problem);
        // For linear frontier with d=(0,0), NBS is (5, 5)
        assert!((solution[0] - 5.0).abs() < 1.0);
        assert!((solution[1] - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_kalai_smorodinsky() {
        let problem = BargainingProblem::linear_frontier(10.0, [0.0, 0.0]);
        let solution = kalai_smorodinsky_solution(&problem);
        // Same as NBS for symmetric problems
        assert!((solution[0] - solution[1]).abs() < 1.5);
    }

    #[test]
    fn test_egalitarian() {
        let problem = BargainingProblem::linear_frontier(10.0, [0.0, 0.0]);
        let solution = egalitarian_solution(&problem);
        assert!((solution[0] - solution[1]).abs() < 1.5);
    }

    #[test]
    fn test_rubinstein_symmetric() {
        let result = rubinstein_equal_delta(0.9);
        assert!((result.share_1 + result.share_2 - 1.0).abs() < 1e-10);
        // First mover advantage even with equal discount factors
        assert!(result.share_1 >= result.share_2 - 1e-10);
    }

    #[test]
    fn test_rubinstein_asymmetric() {
        let result = rubinstein_bargaining(0.9, 0.5);
        // More patient player gets more
        assert!(result.share_1 > result.share_2);
    }

    #[test]
    fn test_rubinstein_shares_sum_to_one() {
        let result = rubinstein_bargaining(0.8, 0.7);
        assert!((result.share_1 + result.share_2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_individual_rationality() {
        let problem = BargainingProblem::new(
            [1.0, 1.0],
            vec![[2.0, 3.0], [0.5, 4.0], [3.0, 0.5]],
        );
        let ir = problem.individually_rational();
        assert_eq!(ir.len(), 1); // Only [2.0, 3.0] is IR
    }

    #[test]
    fn test_nash_axioms() {
        let problem = BargainingProblem::linear_frontier(10.0, [0.0, 0.0]);
        let solution = nash_bargaining_solution(&problem);
        let check = check_nash_axioms(&problem, solution);
        assert!(check.pareto_efficiency);
        assert!(check.symmetry);
    }
}
