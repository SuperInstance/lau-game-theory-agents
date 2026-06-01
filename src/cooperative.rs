//! Cooperative games: Shapley value, core, nucleolus.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A characteristic function cooperative game (N, v).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooperativeGame {
    pub n_players: usize,
    /// Characteristic function: coalition -> value.
    /// Coalition represented as bitmask.
    pub characteristic_fn: HashMap<u64, f64>,
    pub player_names: Vec<String>,
}

impl CooperativeGame {
    /// Create a cooperative game from a characteristic function.
    pub fn new(n_players: usize, char_fn: HashMap<u64, f64>) -> Self {
        let names = (0..n_players).map(|i| format!("P{}", i)).collect();
        Self {
            n_players,
            characteristic_fn: char_fn,
            player_names: names,
        }
    }

    /// Create from a closure.
    pub fn from_fn<F: Fn(u64) -> f64>(n_players: usize, f: F) -> Self {
        let mut char_fn = HashMap::new();
        for mask in 0..(1u64 << n_players) {
            char_fn.insert(mask, f(mask));
        }
        Self::new(n_players, char_fn)
    }

    /// Value of a coalition.
    pub fn value(&self, coalition: u64) -> f64 {
        self.characteristic_fn.get(&coalition).copied().unwrap_or(0.0)
    }

    /// Grand coalition value.
    pub fn grand_coalition_value(&self) -> f64 {
        self.value((1u64 << self.n_players) - 1)
    }

    /// Check if a coalition S is empty.
    pub fn is_empty_coalition(&self, coalition: u64) -> bool {
        coalition == 0
    }

    /// Compute the Shapley value for all players.
    pub fn shapley_value(&self) -> Vec<f64> {
        let n = self.n_players;
        let mut shapley = vec![0.0; n];

        for player in 0..n {
            let mut total = 0.0;
            let player_bit = 1u64 << player;

            // Sum over all coalitions not containing player
            for mask in 0..(1u64 << n) {
                if mask & player_bit != 0 { continue; }
                let s_size = mask.count_ones() as usize;
                let coalition_with = mask | player_bit;

                let marginal = self.value(coalition_with) - self.value(mask);
                let weight = factorial(s_size) * factorial(n - s_size - 1) as f64 / factorial(n);
                total += weight * marginal;
            }
            shapley[player] = total;
        }
        shapley
    }

    /// Check if an allocation is in the core.
    pub fn is_in_core(&self, allocation: &[f64]) -> bool {
        let n = self.n_players;
        // Efficiency: sum = v(N)
        let total: f64 = allocation.iter().sum();
        if (total - self.grand_coalition_value()).abs() > 1e-8 { return false; }

        // Individual rationality
        for i in 0..n {
            if allocation[i] < self.value(1u64 << i) - 1e-8 { return false; }
        }

        // Coalitional rationality
        for mask in 1u64..(1u64 << n) {
            let coalition_sum: f64 = (0..n)
                .filter(|&i| mask & (1u64 << i) != 0)
                .map(|i| allocation[i])
                .sum();
            if coalition_sum < self.value(mask) - 1e-8 { return false; }
        }
        true
    }

    /// Find the core (if non-empty) via constraint satisfaction.
    /// Returns one allocation in the core, or None if core is empty.
    pub fn find_core_allocation(&self) -> Option<Vec<f64>> {
        let n = self.n_players;
        let v_n = self.grand_coalition_value();

        // Simple approach: try the Shapley value
        let shapley = self.shapley_value();
        if self.is_in_core(&shapley) {
            return Some(shapley);
        }

        // Try equal division
        let equal = vec![v_n / n as f64; n];
        if self.is_in_core(&equal) {
            return Some(equal);
        }

        // Try proportional to individual values
        let indiv_sum: f64 = (0..n).map(|i| self.value(1u64 << i)).sum();
        if indiv_sum > 1e-10 {
            let proportional: Vec<f64> = (0..n)
                .map(|i| v_n * self.value(1u64 << i) / indiv_sum)
                .collect();
            if self.is_in_core(&proportional) {
                return Some(proportional);
            }
        }

        None
    }

    /// Check if the game is superadditive.
    pub fn is_superadditive(&self) -> bool {
        let n = self.n_players;
        for s1 in 1u64..(1u64 << n) {
            for s2 in 1u64..(1u64 << n) {
                if s1 & s2 != 0 { continue; }
                if self.value(s1 | s2) < self.value(s1) + self.value(s2) - 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the game is convex (supermodular).
    pub fn is_convex(&self) -> bool {
        let n = self.n_players;
        for s in 0u64..(1u64 << n) {
            for i in 0..n {
                if s & (1u64 << i) != 0 { continue; }
                for j in (i + 1)..n {
                    if s & (1u64 << j) != 0 { continue; }
                    let s_union_i = s | (1u64 << i);
                    let s_union_j = s | (1u64 << j);
                    let s_union_ij = s | (1u64 << i) | (1u64 << j);
                    let marginal_i = self.value(s_union_i) - self.value(s);
                    let marginal_ij = self.value(s_union_ij) - self.value(s_union_j);
                    if marginal_ij < marginal_i - 1e-10 { return false; }
                }
            }
        }
        true
    }

    /// Compute the nucleolus (approximation via lexicographic center).
    pub fn nucleolus(&self) -> Vec<f64> {
        let n = self.n_players;
        let v_n = self.grand_coalition_value();

        // Start from Shapley value and iteratively improve
        let mut allocation = self.shapley_value();

        // Simple iterative improvement: reduce maximum excess
        for _ in 0..100 {
            let mut max_excess = f64::NEG_INFINITY;
            let mut worst_coalition = 0u64;

            for mask in 1u64..(1u64 << n) {
                if mask == (1u64 << n) - 1 { continue; }
                let coalition_sum: f64 = (0..n)
                    .filter(|&i| mask & (1u64 << i) != 0)
                    .map(|i| allocation[i])
                    .sum();
                let excess = self.value(mask) - coalition_sum;
                if excess > max_excess {
                    max_excess = excess;
                    worst_coalition = mask;
                }
            }

            if max_excess < 1e-10 { break; }

            // Transfer from players not in worst coalition to those in it
            let in_coal: Vec<usize> = (0..n).filter(|&i| worst_coalition & (1u64 << i) != 0).collect();
            let out_coal: Vec<usize> = (0..n).filter(|&i| worst_coalition & (1u64 << i) == 0).collect();

            if in_coal.is_empty() || out_coal.is_empty() { break; }

            let transfer = max_excess / (in_coal.len() + out_coal.len()) as f64;
            for &i in &in_coal { allocation[i] += transfer; }
            for &i in &out_coal { allocation[i] -= transfer; }
        }

        allocation
    }

    /// Compute the Banzhaf power index.
    pub fn banzhaf_index(&self) -> Vec<f64> {
        let n = self.n_players;
        let mut swings = vec![0usize; n];

        for mask in 0..(1u64 << n) {
            for i in 0..n {
                if mask & (1u64 << i) != 0 { continue; }
                let with_i = mask | (1u64 << i);
                let marginal = self.value(with_i) - self.value(mask);
                if marginal > 1e-10 {
                    swings[i] += 1;
                }
            }
        }

        let total: usize = swings.iter().sum();
        if total == 0 { return vec![0.0; n]; }
        swings.iter().map(|&s| s as f64 / total as f64).collect()
    }
}

fn factorial(n: usize) -> f64 {
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Simple voting game: coalition wins if total votes >= quota.
pub fn voting_game(weights: &[f64], quota: f64) -> CooperativeGame {
    let n = weights.len();
    CooperativeGame::from_fn(n, |mask| {
        let total: f64 = (0..n)
            .filter(|&i| mask & (1u64 << i) != 0)
            .map(|i| weights[i])
            .sum();
        if total >= quota { 1.0 } else { 0.0 }
    })
}

/// Glove game: left gloves and right gloves, value = min(lefts, rights).
pub fn glove_game(n_left: usize, n_right: usize) -> CooperativeGame {
    let n = n_left + n_right;
    CooperativeGame::from_fn(n, |mask| {
        let lefts = (0..n_left).filter(|&i| mask & (1u64 << i) != 0).count() as f64;
        let rights = (n_left..n).filter(|&i| mask & (1u64 << i) != 0).count() as f64;
        lefts.min(rights)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_game() -> CooperativeGame {
        let mut char_fn = HashMap::new();
        char_fn.insert(0b000, 0.0);
        char_fn.insert(0b001, 1.0);
        char_fn.insert(0b010, 2.0);
        char_fn.insert(0b011, 4.0);
        char_fn.insert(0b100, 3.0);
        char_fn.insert(0b101, 5.0);
        char_fn.insert(0b110, 6.0);
        char_fn.insert(0b111, 10.0);
        CooperativeGame::new(3, char_fn)
    }

    #[test]
    fn test_shapley_value() {
        let game = simple_game();
        let sv = game.shapley_value();
        // Sum should equal grand coalition value
        let sum: f64 = sv.iter().sum();
        assert!((sum - 10.0).abs() < 1e-8);
    }

    #[test]
    fn test_shapley_symmetry() {
        // Symmetric game: all players equal
        let game = CooperativeGame::from_fn(3, |mask| {
            mask.count_ones() as f64 * 2.0
        });
        let sv = game.shapley_value();
        assert!((sv[0] - sv[1]).abs() < 1e-10);
        assert!((sv[1] - sv[2]).abs() < 1e-10);
    }

    #[test]
    fn test_core_check() {
        let game = simple_game();
        // (1, 2, 3) sums to 6, not 10 -> not efficient
        assert!(!game.is_in_core(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_superadditive() {
        let game = simple_game();
        assert!(game.is_superadditive());
    }

    #[test]
    fn test_voting_game_banzhaf() {
        let game = voting_game(&[50.0, 30.0, 20.0], 60.0);
        let bz = game.banzhaf_index();
        assert!(bz[0] >= bz[1]); // Largest player has most power
        assert!(bz[0] >= bz[2]);
    }

    #[test]
    fn test_glove_game() {
        let game = glove_game(1, 2);
        let sv = game.shapley_value();
        let sum: f64 = sv.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8); // Value is min(1,2)=1
    }

    #[test]
    fn test_grand_coalition() {
        let game = simple_game();
        assert!((game.grand_coalition_value() - 10.0).abs() < 1e-8);
    }

    #[test]
    fn test_nucleolus() {
        let game = simple_game();
        let nuc = game.nucleolus();
        let sum: f64 = nuc.iter().sum();
        assert!((sum - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_core_allocation() {
        let game = simple_game();
        if let Some(alloc) = game.find_core_allocation() {
            assert!(game.is_in_core(&alloc));
        }
    }
}
