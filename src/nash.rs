//! Nash equilibrium computation: support enumeration, best response dynamics, and existence verification.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::normal_form::NormalFormGame;

/// A Nash equilibrium: mixed strategies for each player.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibrium {
    pub strategies: Vec<DVector<f64>>,
    pub expected_payoffs: Vec<f64>,
}

/// Compute all Nash equilibria of a 2-player normal form game via support enumeration.
pub fn support_enumeration(game: &NormalFormGame) -> Vec<NashEquilibrium> {
    assert_eq!(game.n_players, 2, "Support enumeration only for 2-player games");
    let n1 = game.n_strategies[0];
    let n2 = game.n_strategies[1];
    let mut equilibria = Vec::new();

    for sup1_size in 1..=n1 {
        for sup2_size in 1..=n2 {
            for sup1 in supports(n1, sup1_size) {
                for sup2 in supports(n2, sup2_size) {
                    if let Some(ne) = solve_support(game, &sup1, &sup2) {
                        equilibria.push(ne);
                    }
                }
            }
        }
    }
    equilibria
}

fn supports(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    generate_supports(0, n, k, &mut current, &mut result);
    result
}

fn generate_supports(start: usize, n: usize, k: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    for i in start..n {
        current.push(i);
        generate_supports(i + 1, n, k, current, result);
        current.pop();
    }
}

fn solve_support(game: &NormalFormGame, sup1: &[usize], sup2: &[usize]) -> Option<NashEquilibrium> {
    let a = &game.payoffs[0];
    let b = &game.payoffs[1];
    let n1 = game.n_strategies[0];
    let n2 = game.n_strategies[1];

    // Solve for player 2's mixed strategy q that makes player 1 indifferent over sup1
    // A[si, :] * q = v for all si in sup1, where v is the value
    // q[j] = 0 for j not in sup2, sum(q[sup2]) = 1
    //
    // This means: for all si in sup1: sum_j_in_sup2 A[si,j] * q[j] = v
    // With constraint: sum(q[sup2]) = 1
    //
    // Variables: q[sup2[0]], q[sup2[1]], ..., q[sup2[k2-1]], v
    // Equations: |sup1| indifference + 1 normalization = |sup1| + 1
    // Variables: |sup2| + 1 (q's + v)
    
    let k1 = sup1.len();
    let k2 = sup2.len();
    let n_vars = k2 + 1; // q[0..k2] and v
    let n_eq = k1 + 1;   // indifference for each si + normalization

    // Build the system for q and v:
    // For each si in sup1: sum_{j_idx} A[si, sup2[j_idx]] * q[j_idx] - v = 0
    // Normalization: sum q[j_idx] = 1

    let mut mat = vec![0.0f64; n_eq * n_vars];
    let mut rhs = vec![0.0f64; n_eq];

    for (eq_i, &si) in sup1.iter().enumerate() {
        for (j_idx, &j) in sup2.iter().enumerate() {
            mat[eq_i * n_vars + j_idx] = a[(si, j)];
        }
        mat[eq_i * n_vars + k2] = -1.0; // -v
        rhs[eq_i] = 0.0;
    }

    // Normalization row
    for j_idx in 0..k2 {
        mat[k1 * n_vars + j_idx] = 1.0;
    }
    mat[k1 * n_vars + k2] = 0.0; // v not involved
    rhs[k1] = 1.0;

    let m = nalgebra::DMatrix::from_row_slice(n_eq, n_vars, &mat);
    let r = nalgebra::DVector::from_vec(rhs.to_vec());

    let sol = if n_eq == n_vars {
        m.clone().lu().solve(&r)?
    } else if n_eq > n_vars {
        // Overdetermined: least squares
        let ata = &m.transpose() * &m;
        let atr = m.transpose() * &r;
        ata.lu().solve(&atr)?
    } else {
        // Underdetermined: try pseudo-inverse approach
        let ata = &m.transpose() * &m;
        let atr = m.transpose() * &r;
        ata.lu().solve(&atr)?
    };

    // Extract q
    let mut q = DVector::zeros(n2);
    for (j_idx, &j) in sup2.iter().enumerate() {
        q[j] = sol[j_idx];
    }

    // Validate q
    for j in 0..n2 {
        if q[j] < -1e-8 { return None; }
    }
    for j in 0..n2 {
        if !sup2.contains(&j) && q[j] > 1e-8 { return None; }
    }
    let q_sum: f64 = q.iter().sum();
    if q_sum < 1e-10 { return None; }
    q /= q_sum;

    // Now solve for player 1's mixed strategy p that makes player 2 indifferent over sup2
    // For each sj in sup2: sum_{i_idx} B[sup1[i_idx], sj] * p[i_idx] = w
    let n_vars_p = k1 + 1;
    let n_eq_p = k2 + 1;

    let mut mat_p = vec![0.0f64; n_eq_p * n_vars_p];
    let mut rhs_p = vec![0.0f64; n_eq_p];

    for (eq_j, &sj) in sup2.iter().enumerate() {
        for (i_idx, &i) in sup1.iter().enumerate() {
            mat_p[eq_j * n_vars_p + i_idx] = b[(i, sj)];
        }
        mat_p[eq_j * n_vars_p + k1] = -1.0;
        rhs_p[eq_j] = 0.0;
    }
    for i_idx in 0..k1 {
        mat_p[k2 * n_vars_p + i_idx] = 1.0;
    }
    rhs_p[k2] = 1.0;

    let mp = nalgebra::DMatrix::from_row_slice(n_eq_p, n_vars_p, &mat_p);
    let rp = nalgebra::DVector::from_vec(rhs_p.to_vec());

    let sol_p = if n_eq_p == n_vars_p {
        mp.clone().lu().solve(&rp)?
    } else {
        let ata = &mp.transpose() * &mp;
        let atr = mp.transpose() * &rp;
        ata.lu().solve(&atr)?
    };

    let mut p = DVector::zeros(n1);
    for (i_idx, &i) in sup1.iter().enumerate() {
        p[i] = sol_p[i_idx];
    }

    for i in 0..n1 {
        if p[i] < -1e-8 { return None; }
    }
    for i in 0..n1 {
        if !sup1.contains(&i) && p[i] > 1e-8 { return None; }
    }
    let p_sum: f64 = p.iter().sum();
    if p_sum < 1e-10 { return None; }
    p /= p_sum;

    // Verify best response conditions
    // Player 1: strategies in sup1 all give same payoff against q; strategies outside give <=
    let v1 = payoff_against_mixed(a, &p, &q, 0, sup1[0]);
    for &si in sup1 {
        let val = row_against_q(a, si, &q);
        if (val - v1).abs() > 1e-6 { return None; }
    }
    for i in 0..n1 {
        if sup1.contains(&i) { continue; }
        if row_against_q(a, i, &q) > v1 + 1e-8 { return None; }
    }

    let v2 = col_against_p(b, sup2[0], &p);
    for &sj in sup2 {
        let val = col_against_p(b, sj, &p);
        if (val - v2).abs() > 1e-6 { return None; }
    }
    for j in 0..n2 {
        if sup2.contains(&j) { continue; }
        if col_against_p(b, j, &p) > v2 + 1e-8 { return None; }
    }

    let ep1 = expected_payoff_2p(a, &p, &q);
    let ep2 = expected_payoff_2p(b, &p, &q);

    Some(NashEquilibrium {
        strategies: vec![p, q],
        expected_payoffs: vec![ep1, ep2],
    })
}

fn row_against_q(a: &nalgebra::DMatrix<f64>, row: usize, q: &DVector<f64>) -> f64 {
    let mut val = 0.0;
    for j in 0..q.len() {
        val += a[(row, j)] * q[j];
    }
    val
}

fn col_against_p(b: &nalgebra::DMatrix<f64>, col: usize, p: &DVector<f64>) -> f64 {
    let mut val = 0.0;
    for i in 0..p.len() {
        val += b[(i, col)] * p[i];
    }
    val
}

fn payoff_against_mixed(a: &nalgebra::DMatrix<f64>, _p: &DVector<f64>, q: &DVector<f64>, _player: usize, row: usize) -> f64 {
    row_against_q(a, row, q)
}

fn expected_payoff_2p(a: &nalgebra::DMatrix<f64>, p: &DVector<f64>, q: &DVector<f64>) -> f64 {
    let mut val = 0.0;
    for i in 0..p.len() {
        for j in 0..q.len() {
            val += p[i] * q[j] * a[(i, j)];
        }
    }
    val
}

/// Best response dynamics: iterate best responses until convergence.
pub fn best_response_dynamics(game: &NormalFormGame, max_iter: usize) -> Option<NashEquilibrium> {
    assert_eq!(game.n_players, 2);
    let n1 = game.n_strategies[0];
    let n2 = game.n_strategies[1];
    let mut p = DVector::from_element(n1, 1.0 / n1 as f64);
    let mut q = DVector::from_element(n2, 1.0 / n2 as f64);

    for _ in 0..max_iter {
        let br1 = game.best_responses(0, &q);
        let mut new_p = DVector::zeros(n1);
        for &i in &br1 { new_p[i] = 1.0 / br1.len() as f64; }

        let br2 = game.best_responses(1, &p);
        let mut new_q = DVector::zeros(n2);
        for &j in &br2 { new_q[j] = 1.0 / br2.len() as f64; }

        let converged = (&new_p - &p).norm() < 1e-10 && (&new_q - &q).norm() < 1e-10;
        p = new_p;
        q = new_q;
        if converged {
            let a = &game.payoffs[0];
            let b = &game.payoffs[1];
            return Some(NashEquilibrium {
                strategies: vec![p.clone(), q.clone()],
                expected_payoffs: vec![
                    expected_payoff_2p(a, &p, &q),
                    expected_payoff_2p(b, &p, &q),
                ],
            });
        }
    }
    None
}

/// Verify Nash equilibrium conditions for a 2-player game.
pub fn verify_nash(game: &NormalFormGame, ne: &NashEquilibrium) -> bool {
    let p = &ne.strategies[0];
    let q = &ne.strategies[1];
    let a = &game.payoffs[0];
    let b = &game.payoffs[1];

    // Check probabilities are valid
    for i in 0..p.len() { if p[i] < -1e-8 { return false; } }
    for j in 0..q.len() { if q[j] < -1e-8 { return false; } }
    if (p.iter().sum::<f64>() - 1.0).abs() > 1e-6 { return false; }
    if (q.iter().sum::<f64>() - 1.0).abs() > 1e-6 { return false; }

    // Compute payoffs for player 1
    let mut best1 = f64::NEG_INFINITY;
    let mut cur1 = 0.0;
    for i in 0..p.len() {
        let val = row_against_q(a, i, q);
        if val > best1 { best1 = val; }
        cur1 += p[i] * val;
    }
    if cur1 < best1 - 1e-6 { return false; }

    // Compute payoffs for player 2
    let mut best2 = f64::NEG_INFINITY;
    let mut cur2 = 0.0;
    for j in 0..q.len() {
        let val = col_against_p(b, j, p);
        if val > best2 { best2 = val; }
        cur2 += q[j] * val;
    }
    if cur2 < best2 - 1e-6 { return false; }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nash_prisoners_dilemma() {
        let g = NormalFormGame::prisoners_dilemma();
        let equilibria = support_enumeration(&g);
        assert!(!equilibria.is_empty());
        let ne = &equilibria[0];
        assert!(verify_nash(&g, ne));
    }

    #[test]
    fn test_nash_matching_pennies() {
        let g = NormalFormGame::matching_pennies();
        let equilibria = support_enumeration(&g);
        assert!(!equilibria.is_empty());
        let ne = &equilibria[0];
        assert!((ne.strategies[0][0] - 0.5).abs() < 0.1);
        assert!((ne.strategies[1][0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_nash_battle_of_the_sexes() {
        let g = NormalFormGame::battle_of_the_sexes();
        let equilibria = support_enumeration(&g);
        assert!(equilibria.len() >= 2);
    }

    #[test]
    fn test_best_response_dynamics() {
        let g = NormalFormGame::prisoners_dilemma();
        let ne = best_response_dynamics(&g, 100);
        assert!(ne.is_some());
        let ne = ne.unwrap();
        assert!(verify_nash(&g, &ne));
    }

    #[test]
    fn test_nash_stag_hunt() {
        let g = NormalFormGame::stag_hunt();
        let equilibria = support_enumeration(&g);
        assert!(equilibria.len() >= 2);
    }

    #[test]
    fn test_verify_valid_nash() {
        let g = NormalFormGame::prisoners_dilemma();
        let ne = NashEquilibrium {
            strategies: vec![
                DVector::from_vec(vec![0.0, 1.0]),
                DVector::from_vec(vec![0.0, 1.0]),
            ],
            expected_payoffs: vec![1.0, 1.0],
        };
        assert!(verify_nash(&g, &ne));
    }

    #[test]
    fn test_verify_invalid_nash() {
        let g = NormalFormGame::prisoners_dilemma();
        let ne = NashEquilibrium {
            strategies: vec![
                DVector::from_vec(vec![1.0, 0.0]),
                DVector::from_vec(vec![1.0, 0.0]),
            ],
            expected_payoffs: vec![3.0, 3.0],
        };
        assert!(!verify_nash(&g, &ne));
    }

    #[test]
    fn test_rock_paper_scissors_mixed_ne() {
        let g = NormalFormGame::rock_paper_scissors();
        let equilibria = support_enumeration(&g);
        assert!(!equilibria.is_empty());
        let ne = &equilibria[0];
        for p in &ne.strategies {
            for &v in p.iter() {
                assert!((v - 1.0/3.0).abs() < 0.15, "Expected ~1/3, got {}", v);
            }
        }
    }
}
