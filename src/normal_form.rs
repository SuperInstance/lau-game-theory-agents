//! Normal form games: payoff matrices, dominance, best response analysis.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A normal-form (strategic-form) game for N players.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalFormGame {
    /// Number of players.
    pub n_players: usize,
    /// Number of strategies per player.
    pub n_strategies: Vec<usize>,
    /// Payoff tensors: payoffs[player] is a matrix where rows are joint-strategy profiles.
    /// Stored as flat matrices for simplicity. For 2-player games, payoffs[p] is n_strat x n_strat.
    pub payoffs: Vec<DMatrix<f64>>,
    /// Strategy labels per player.
    pub strategy_labels: Vec<Vec<String>>,
}

impl NormalFormGame {
    /// Create a 2-player normal-form game from payoff matrices.
    pub fn two_player(row_payoffs: DMatrix<f64>, col_payoffs: DMatrix<f64>) -> Self {
        let n_row = row_payoffs.nrows();
        let n_col = row_payoffs.ncols();
        assert_eq!(col_payoffs.nrows(), n_row);
        assert_eq!(col_payoffs.ncols(), n_col);
        Self {
            n_players: 2,
            n_strategies: vec![n_row, n_col],
            payoffs: vec![row_payoffs, col_payoffs],
            strategy_labels: vec![
                (0..n_row).map(|i| format!("R{i}")).collect(),
                (0..n_col).map(|j| format!("C{j}")).collect(),
            ],
        }
    }

    /// Create a 2-player game from arrays (row-major).
    pub fn from_arrays(n_row: usize, n_col: usize, row_payoffs: &[f64], col_payoffs: &[f64]) -> Self {
        Self::two_player(
            DMatrix::from_row_slice(n_row, n_col, row_payoffs),
            DMatrix::from_row_slice(n_row, n_col, col_payoffs),
        )
    }

    /// Classic Prisoner's Dilemma.
    pub fn prisoners_dilemma() -> Self {
        Self::from_arrays(2, 2,
            &[3.0, 0.0, 5.0, 1.0],  // player 1
            &[3.0, 5.0, 0.0, 1.0],  // player 2
        )
    }

    /// Matching Pennies.
    pub fn matching_pennies() -> Self {
        Self::from_arrays(2, 2,
            &[1.0, -1.0, -1.0, 1.0],
            &[-1.0, 1.0, 1.0, -1.0],
        )
    }

    /// Battle of the Sexes.
    pub fn battle_of_the_sexes() -> Self {
        Self::from_arrays(2, 2,
            &[3.0, 0.0, 0.0, 2.0],
            &[2.0, 0.0, 0.0, 3.0],
        )
    }

    /// Stag Hunt.
    pub fn stag_hunt() -> Self {
        Self::from_arrays(2, 2,
            &[4.0, 0.0, 3.0, 2.0],
            &[4.0, 3.0, 0.0, 2.0],
        )
    }

    /// Chicken (Hawk-Dove).
    pub fn chicken() -> Self {
        Self::from_arrays(2, 2,
            &[0.0, -1.0, 3.0, -5.0],
            &[0.0, 3.0, -1.0, -5.0],
        )
    }

    /// Rock-Paper-Scissors.
    pub fn rock_paper_scissors() -> Self {
        Self::from_arrays(3, 3,
            &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0],
            &[0.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0],
        )
    }

    /// Get payoff for player at a pure strategy profile.
    pub fn payoff(&self, player: usize, strategies: &[usize]) -> f64 {
        let p = &self.payoffs[player];
        match self.n_players {
            2 => p[(strategies[0], strategies[1])],
            _ => panic!("Payoff lookup only implemented for 2-player games via this method"),
        }
    }

    /// Check if strategy `s` for `player` is strictly dominated.
    pub fn is_strictly_dominated(&self, player: usize, s: usize) -> bool {
        let n_strats = self.n_strategies[player];
        for other in 0..n_strats {
            if other == s { continue; }
            if self.strictly_dominates(player, other, s) {
                return true;
            }
        }
        false
    }

    /// Check if strategy `s1` strictly dominates `s2` for `player`.
    pub fn strictly_dominates(&self, player: usize, s1: usize, s2: usize) -> bool {
        if self.n_players != 2 { return false; }
        let opponent = 1 - player;
        let n_opp = self.n_strategies[opponent];
        let mut all_better = true;
        for opp_s in 0..n_opp {
            let profile_a = if player == 0 { [s1, opp_s] } else { [opp_s, s1] };
            let profile_b = if player == 0 { [s2, opp_s] } else { [opp_s, s2] };
            if self.payoff(player, &profile_a) <= self.payoff(player, &profile_b) {
                all_better = false;
                break;
            }
        }
        all_better
    }

    /// Find all strictly dominated strategies for a player.
    pub fn dominated_strategies(&self, player: usize) -> Vec<usize> {
        (0..self.n_strategies[player])
            .filter(|&s| self.is_strictly_dominated(player, s))
            .collect()
    }

    /// Compute best response for `player` given opponent's mixed strategy.
    pub fn best_responses(&self, player: usize, opponent_mixed: &DVector<f64>) -> Vec<usize> {
        let mut expected: Vec<f64> = Vec::new();
        let n_strats = self.n_strategies[player];
        for s in 0..n_strats {
            let mut exp_val = 0.0;
            for (opp_s, &prob) in opponent_mixed.iter().enumerate() {
                let profile = if player == 0 { [s, opp_s] } else { [opp_s, s] };
                exp_val += prob * self.payoff(player, &profile);
            }
            expected.push(exp_val);
        }
        let max_val = expected.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        expected.iter().enumerate()
            .filter(|(_, &v)| (v - max_val).abs() < 1e-10)
            .map(|(i, _)| i)
            .collect()
    }

    /// Iterated elimination of strictly dominated strategies (returns surviving strategy indices per player).
    pub fn iesds(&self) -> Vec<Vec<usize>> {
        let mut surviving: Vec<Vec<usize>> = (0..self.n_players)
            .map(|p| (0..self.n_strategies[p]).collect())
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            for player in 0..self.n_players {
                let mut dominated = Vec::new();
                for &s in &surviving[player] {
                    let mut is_dom = false;
                    for &other in &surviving[player] {
                        if other == s { continue; }
                        // Check domination against surviving opponent strategies
                        if self.n_players == 2 {
                            let opp = 1 - player;
                            let mut all_better = true;
                            for &opp_s in &surviving[opp] {
                                let prof_a = if player == 0 { [other, opp_s] } else { [opp_s, other] };
                                let prof_b = if player == 0 { [s, opp_s] } else { [opp_s, s] };
                                if self.payoff(player, &prof_a) <= self.payoff(player, &prof_b) {
                                    all_better = false;
                                    break;
                                }
                            }
                            if all_better { is_dom = true; break; }
                        }
                    }
                    if is_dom { dominated.push(s); }
                }
                if !dominated.is_empty() {
                    surviving[player].retain(|s| !dominated.contains(s));
                    changed = true;
                }
            }
        }
        surviving
    }

    /// Compute expected payoff for `player` given both players' mixed strategies.
    pub fn expected_payoff(&self, player: usize, mixed: &[DVector<f64>]) -> f64 {
        if self.n_players != 2 { return 0.0; }
        let (p0, p1) = (&mixed[0], &mixed[1]);
        let mut val = 0.0;
        for i in 0..p0.len() {
            for j in 0..p1.len() {
                let profile = if player == 0 { [i, j] } else { [i, j] };
                val += p0[i] * p1[j] * self.payoff(player, &profile);
            }
        }
        val
    }
}

impl fmt::Display for NormalFormGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.n_players != 2 { return write!(f, "Multi-player game ({} players)", self.n_players); }
        let (nr, nc) = (self.n_strategies[0], self.n_strategies[1]);
        writeln!(f, "2-Player Normal Form Game ({}×{})\n", nr, nc)?;
        for i in 0..nr {
            for j in 0..nc {
                write!(f, "({:.1},{:.1}) ", self.payoffs[0][(i,j)], self.payoffs[1][(i,j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prisoners_dilemma_payoffs() {
        let g = NormalFormGame::prisoners_dilemma();
        assert_eq!(g.payoff(0, &[0, 0]), 3.0); // Both cooperate
        assert_eq!(g.payoff(0, &[1, 0]), 5.0); // P1 defects, P2 cooperates
        assert_eq!(g.payoff(1, &[0, 1]), 5.0); // P2 defects, P1 cooperates
        assert_eq!(g.payoff(0, &[1, 1]), 1.0); // Both defect
    }

    #[test]
    fn test_strict_dominance() {
        let g = NormalFormGame::prisoners_dilemma();
        assert!(g.is_strictly_dominated(0, 0)); // Cooperate is dominated by Defect
        assert!(!g.is_strictly_dominated(0, 1)); // Defect is not dominated
    }

    #[test]
    fn test_best_response() {
        let g = NormalFormGame::prisoners_dilemma();
        let opp_mixed = DVector::from_vec(vec![0.5, 0.5]);
        let br = g.best_responses(0, &opp_mixed);
        assert_eq!(br, vec![1]); // Always best to defect
    }

    #[test]
    fn test_iesds() {
        let g = NormalFormGame::prisoners_dilemma();
        let surviving = g.iesds();
        assert_eq!(surviving[0], vec![1]); // Only defect survives
        assert_eq!(surviving[1], vec![1]);
    }

    #[test]
    fn test_matching_pennies_no_dominance() {
        let g = NormalFormGame::matching_pennies();
        assert!(g.dominated_strategies(0).is_empty());
        assert!(g.dominated_strategies(1).is_empty());
    }

    #[test]
    fn test_expected_payoff() {
        let g = NormalFormGame::prisoners_dilemma();
        let mixed = vec![
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let val = g.expected_payoff(0, &mixed);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_game_display() {
        let g = NormalFormGame::prisoners_dilemma();
        let s = format!("{}", g);
        assert!(s.contains("2-Player"));
    }

    #[test]
    fn test_rock_paper_scissors() {
        let g = NormalFormGame::rock_paper_scissors();
        assert_eq!(g.n_strategies[0], 3);
        assert_eq!(g.payoff(0, &[0, 1]), -1.0); // Rock vs Paper
        assert_eq!(g.payoff(0, &[0, 2]), 1.0);  // Rock vs Scissors
    }
}
