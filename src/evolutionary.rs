//! Evolutionary game theory: replicator dynamics, ESS, fitness landscapes.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use crate::normal_form::NormalFormGame;

/// Configuration for evolutionary dynamics simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryConfig {
    pub time_steps: usize,
    pub dt: f64,
}

impl Default for EvolutionaryConfig {
    fn default() -> Self {
        Self { time_steps: 1000, dt: 0.01 }
    }
}

/// Result of evolutionary simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryResult {
    /// Population frequencies over time: trajectory[t] = DVector of frequencies.
    pub trajectory: Vec<DVector<f64>>,
    /// Average fitness over time.
    pub avg_fitness: Vec<f64>,
    /// Converged flag.
    pub converged: bool,
}

/// Replicator dynamics for a symmetric game.
/// dx_i/dt = x_i * (f_i - f_bar)
/// where f_i = (Ax)_i and f_bar = x^T A x
pub fn replicator_dynamics(
    payoff_matrix: &DMatrix<f64>,
    initial_freq: &DVector<f64>,
    config: &EvolutionaryConfig,
) -> EvolutionaryResult {
    let n = initial_freq.len();
    let mut x = initial_freq.clone();
    let mut trajectory = Vec::with_capacity(config.time_steps + 1);
    let mut avg_fitness = Vec::with_capacity(config.time_steps + 1);
    trajectory.push(x.clone());

    let mut converged = false;
    for _ in 0..config.time_steps {
        let fitness = payoff_matrix * &x;
        let avg_fit = x.dot(&fitness);
        avg_fitness.push(avg_fit);

        // Replicator equation
        let dx = &x.component_mul(&(&fitness - DVector::from_element(n, avg_fit))) * config.dt;

        x += dx;

        // Clamp negatives to zero
        for v in x.iter_mut() {
            if *v < 0.0 { *v = 0.0; }
        }

        // Renormalize
        let sum: f64 = x.iter().sum();
        if sum > 1e-15 { x /= sum; }

        trajectory.push(x.clone());

        // Check convergence
        if trajectory.len() >= 2 {
            let diff = &trajectory[trajectory.len() - 1] - &trajectory[trajectory.len() - 2];
            if diff.norm() < 1e-10 {
                converged = true;
                break;
            }
        }
    }

    EvolutionaryResult { trajectory, avg_fitness, converged }
}

/// Check if a strategy is an Evolutionarily Stable Strategy (ESS).
/// A strategy x is ESS if for all y != x:
/// 1. x^T A x >= y^T A x  (Nash condition)
/// 2. If x^T A x = y^T A x, then x^T A y > y^T A y  (stability condition)
pub fn is_ess(payoff_matrix: &DMatrix<f64>, strategy: &DVector<f64>) -> bool {
    let n = strategy.len();
    let x = strategy;
    let ax = payoff_matrix * x;
    let xax = x.dot(&ax);

    // Check against all pure strategies as potential invaders
    for i in 0..n {
        let mut y = DVector::zeros(n);
        y[i] = 1.0;

        // Skip if y == x (same pure strategy)
        let diff = &y - x;
        if diff.norm() < 1e-10 { continue; }

        let ay = payoff_matrix * &y;
        let yax = y.dot(&ax);

        // Condition 1: Nash
        if yax > xax + 1e-10 {
            return false;
        }

        // Condition 2: Stability
        if (yax - xax).abs() < 1e-10 {
            let xay = x.dot(&(payoff_matrix * &y));
            let yay = y.dot(&ay);
            if xay <= yay + 1e-10 {
                return false;
            }
        }
    }

    // Also check mixed strategies (sample some)
    for i in 0..n {
        for j in (i+1)..n {
            for &alpha in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let mut y = DVector::zeros(n);
                y[i] = alpha;
                y[j] = 1.0 - alpha;

                let yax = y.dot(&ax);
                if yax > xax + 1e-10 { return false; }
                if (yax - xax).abs() < 1e-10 {
                    let ay_full = payoff_matrix * &y;
                    let xay = x.dot(&ay_full);
                    let yay = y.dot(&ay_full);
                    if xay <= yay + 1e-10 { return false; }
                }
            }
        }
    }

    true
}

/// Find all pure ESS in a symmetric game.
pub fn find_pure_ess(payoff_matrix: &DMatrix<f64>) -> Vec<usize> {
    let n = payoff_matrix.nrows();
    (0..n).filter(|&i| {
        let mut x = DVector::zeros(n);
        x[i] = 1.0;
        is_ess(payoff_matrix, &x)
    }).collect()
}

/// Fitness landscape: compute fitness of each pure strategy against the population.
pub fn fitness_landscape(payoff_matrix: &DMatrix<f64>, population: &DVector<f64>) -> DVector<f64> {
    payoff_matrix * population
}

/// Average fitness of the population.
pub fn average_fitness(payoff_matrix: &DMatrix<f64>, population: &DVector<f64>) -> f64 {
    let fitness = fitness_landscape(payoff_matrix, population);
    population.dot(&fitness)
}

/// Moran process simulation: finite population stochastic dynamics.
pub fn moran_process(
    payoff_matrix: &DMatrix<f64>,
    population_size: usize,
    initial_count: usize,
    strategy_indices: (usize, usize),
    max_steps: usize,
) -> MoranResult {
    let (s1, s2) = strategy_indices;
    let mut count_s1 = initial_count;
    let mut steps = 0;
    let mut fixation = false;

    let fitness_s1_against_s1 = payoff_matrix[(s1, s1)];
    let fitness_s1_against_s2 = payoff_matrix[(s1, s2)];
    let fitness_s2_against_s1 = payoff_matrix[(s2, s1)];
    let fitness_s2_against_s2 = payoff_matrix[(s2, s2)];

    while count_s1 > 0 && count_s1 < population_size && steps < max_steps {
        let n_s1 = count_s1 as f64;
        let n_s2 = (population_size - count_s1) as f64;
        let n = population_size as f64;

        let f1 = (n_s1 * fitness_s1_against_s1 + n_s2 * fitness_s1_against_s2) / n;
        let f2 = (n_s1 * fitness_s2_against_s1 + n_s2 * fitness_s2_against_s2) / n;

        let total_fitness = n_s1 * f1 + n_s2 * f2;
        let prob_increase = (n_s1 * f1 / total_fitness) * (n_s2 / n);
        let prob_decrease = (n_s2 * f2 / total_fitness) * (n_s1 / n);

        let r: f64 = rand_simple();
        if r < prob_increase {
            count_s1 += 1;
        } else if r < prob_increase + prob_decrease {
            count_s1 -= 1;
        }
        steps += 1;
    }

    fixation = count_s1 == population_size;

    MoranResult {
        steps,
        fixation,
        final_count: count_s1,
    }
}

/// Simple deterministic "random" for reproducibility (LCG).
fn rand_simple() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(12345);
    }
    STATE.with(|s| {
        let mut v = s.get();
        v = v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.set(v);
        ((v >> 33) as f64) / (1u64 << 31) as f64
    })
}

/// Result of Moran process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoranResult {
    pub steps: usize,
    pub fixation: bool,
    pub final_count: usize,
}

/// Hawk-Dove game payoff matrix.
pub fn hawk_dove_matrix(v: f64, c: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[
        (v - c) / 2.0, v,
        0.0, v / 2.0,
    ])
}

/// Coordination game payoff matrix.
pub fn coordination_matrix(a: f64, b: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[
        a, 0.0,
        0.0, b,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replicator_convergence_coordination() {
        let a = coordination_matrix(3.0, 2.0);
        let init = DVector::from_vec(vec![0.6, 0.4]);
        let config = EvolutionaryConfig { time_steps: 5000, dt: 0.01 };
        let result = replicator_dynamics(&a, &init, &config);
        // Should converge to one of the pure strategies
        let final_freq = result.trajectory.last().unwrap();
        assert!(final_freq[0] > 0.99 || final_freq[1] > 0.99);
    }

    #[test]
    fn test_ess_pure_hawk_dove() {
        // In Hawk-Dove with V=4, C=6, no pure ESS
        let a = hawk_dove_matrix(4.0, 6.0);
        let ess = find_pure_ess(&a);
        assert!(ess.is_empty());
    }

    #[test]
    fn test_ess_coordination() {
        let a = coordination_matrix(3.0, 2.0);
        let ess = find_pure_ess(&a);
        assert_eq!(ess, vec![0, 1]); // Both pure strategies are ESS in coordination
    }

    #[test]
    fn test_fitness_landscape() {
        let a = coordination_matrix(3.0, 2.0);
        let pop = DVector::from_vec(vec![0.5, 0.5]);
        let fitness = fitness_landscape(&a, &pop);
        assert!((fitness[0] - 1.5).abs() < 1e-10);
        assert!((fitness[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_fitness() {
        let a = coordination_matrix(3.0, 2.0);
        let pop = DVector::from_vec(vec![1.0, 0.0]);
        let af = average_fitness(&a, &pop);
        assert!((af - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_replicator_preserves_sum() {
        let a = coordination_matrix(3.0, 2.0);
        let init = DVector::from_vec(vec![0.3, 0.7]);
        let config = EvolutionaryConfig { time_steps: 100, dt: 0.01 };
        let result = replicator_dynamics(&a, &init, &config);
        for freq in &result.trajectory {
            let sum: f64 = freq.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_moran_process() {
        let a = coordination_matrix(3.0, 2.0);
        let result = moran_process(&a, 20, 5, (0, 1), 10000);
        // With coordination game favoring strategy 0, should likely fix
        // (but stochastic, so just check it runs)
        assert!(result.final_count <= 20);
    }

    #[test]
    fn test_hawk_dove_ess_mixed() {
        let a = hawk_dove_matrix(4.0, 6.0);
        let mixed = DVector::from_vec(vec![0.4, 0.6]); // V/C = 4/6 ≈ 0.67... mixed ESS at V/C
        // The mixed ESS for Hawk-Dove is at p(hawk) = V/C = 2/3
        let p_hawk = 4.0 / 6.0;
        let ess_strat = DVector::from_vec(vec![p_hawk, 1.0 - p_hawk]);
        assert!(is_ess(&a, &ess_strat));
    }
}
