//! Bayesian games: games with incomplete information, Harsanyi transformation.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A type for a player in a Bayesian game.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlayerType {
    pub id: String,
    pub prior_probability: f64,
}

/// A Bayesian game (game of incomplete information).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianGame {
    pub n_players: usize,
    /// Possible types for each player.
    pub type_spaces: Vec<Vec<PlayerType>>,
    /// Payoff function: payoffs[player][(type_profile, strategy_profile)] = payoff.
    /// For simplicity, stored as a function evaluator.
    pub payoff_fn: PayoffTable,
    /// Strategy sets per player per type.
    pub n_strategies: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffTable {
    /// payoffs[player][type_key][strategy_key] = f64
    /// type_key = "t1,t2,...", strategy_key = "s1,s2,..."
    pub entries: Vec<HashMap<String, HashMap<String, f64>>>,
}

impl BayesianGame {
    /// Create a simple 2-player Bayesian game.
    pub fn new(
        n_players: usize,
        type_spaces: Vec<Vec<PlayerType>>,
        n_strategies: Vec<usize>,
    ) -> Self {
        let entries = (0..n_players).map(|_| HashMap::new()).collect();
        Self {
            n_players,
            type_spaces,
            payoff_fn: PayoffTable { entries },
            n_strategies,
        }
    }

    /// Set payoff for a specific player, type profile, and strategy profile.
    pub fn set_payoff(&mut self, player: usize, types: &[usize], strategies: &[usize], payoff: f64) {
        let type_key = types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
        let strat_key = strategies.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
        self.payoff_fn.entries[player]
            .entry(type_key)
            .or_insert_with(HashMap::new)
            .insert(strat_key, payoff);
    }

    /// Get payoff for a specific player, type profile, and strategy profile.
    pub fn get_payoff(&self, player: usize, types: &[usize], strategies: &[usize]) -> f64 {
        let type_key = types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
        let strat_key = strategies.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
        self.payoff_fn.entries[player]
            .get(&type_key)
            .and_then(|m| m.get(&strat_key))
            .copied()
            .unwrap_or(0.0)
    }

    /// Compute the probability of a type profile (independent types).
    pub fn type_profile_probability(&self, types: &[usize]) -> f64 {
        types.iter().enumerate()
            .map(|(p, &t)| self.type_spaces[p][t].prior_probability)
            .product()
    }

    /// Compute expected payoff for a player given strategy rules (one per type).
    /// strategy_rules[player][type] = mixed strategy vector
    pub fn expected_payoff(
        &self,
        player: usize,
        strategy_rules: &[Vec<DVector<f64>>],
        player_type: usize,
    ) -> f64 {
        let mut total = 0.0;
        let n_types_other: Vec<usize> = self.type_spaces.iter().map(|ts| ts.len()).collect();

        // Enumerate all type profiles
        let mut type_profiles = vec![0usize; self.n_players];
        type_profiles[player] = player_type;
        enumerate_type_profiles_for(
            &self.type_spaces,
            player,
            player_type,
            &mut type_profiles,
            0,
            &mut |tp| {
                let type_prob = self.type_profile_probability(&tp);
                if type_prob < 1e-15 { return; }

                // Enumerate all strategy profiles
                let mut strat_profiles = vec![0usize; self.n_players];
                enumerate_strategy_profiles(
                    &self.n_strategies,
                    &mut strat_profiles,
                    0,
                    &mut |sp| {
                        let mut strat_prob = 1.0;
                        for p in 0..self.n_players {
                            strat_prob *= strategy_rules[p][tp[p]][sp[p]];
                        }
                        total += type_prob * strat_prob * self.get_payoff(player, &tp, &sp);
                    },
                );
            },
        );
        total
    }

    /// Apply Harsanyi transformation: convert to a normal form game with one player per type.
    pub fn harsanyi_transform(&self) -> HarsanyiGame {
        // Each "player" in the transformed game is a (player, type) pair
        let transformed_players: Vec<(usize, usize)> = self.type_spaces.iter().enumerate()
            .flat_map(|(p, types)| (0..types.len()).map(move |t| (p, t)))
            .collect();

        HarsanyiGame {
            bayesian_game: self.clone(),
            transformed_players,
        }
    }
}

fn enumerate_type_profiles_for<F: FnMut(&[usize])>(
    type_spaces: &[Vec<PlayerType>],
    fixed_player: usize,
    fixed_type: usize,
    current: &mut [usize],
    player_idx: usize,
    callback: &mut F,
) {
    if player_idx == type_spaces.len() {
        callback(current);
        return;
    }
    if player_idx == fixed_player {
        enumerate_type_profiles_for(type_spaces, fixed_player, fixed_type, current, player_idx + 1, callback);
    } else {
        for t in 0..type_spaces[player_idx].len() {
            current[player_idx] = t;
            enumerate_type_profiles_for(type_spaces, fixed_player, fixed_type, current, player_idx + 1, callback);
        }
    }
}

fn enumerate_strategy_profiles<F: FnMut(&[usize])>(
    n_strategies: &[usize],
    current: &mut [usize],
    player_idx: usize,
    callback: &mut F,
) {
    if player_idx == n_strategies.len() {
        callback(current);
        return;
    }
    for s in 0..n_strategies[player_idx] {
        current[player_idx] = s;
        enumerate_strategy_profiles(n_strategies, current, player_idx + 1, callback);
    }
}

/// Result of Harsanyi transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarsanyiGame {
    pub bayesian_game: BayesianGame,
    pub transformed_players: Vec<(usize, usize)>,
}

impl HarsanyiGame {
    /// Number of players in transformed game.
    pub fn n_transformed(&self) -> usize {
        self.transformed_players.len()
    }
}

/// A Bayesian Nash equilibrium: strategy rules for each player-type pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianNashEquilibrium {
    /// strategy_rules[player][type] = mixed strategy (probabilities over actions)
    pub strategy_rules: Vec<Vec<DVector<f64>>>,
}

/// Solve a simple 2-player, 2-type Bayesian game by brute force over pure strategies.
pub fn solve_bayesian_pure(game: &BayesianGame) -> Vec<BayesianNashEquilibrium> {
    assert_eq!(game.n_players, 2);
    let n1 = game.n_strategies[0];
    let n2 = game.n_strategies[1];
    let nt1 = game.type_spaces[0].len();
    let nt2 = game.type_spaces[1].len();

    let mut equilibria = Vec::new();

    // Enumerate all pure strategy rules: for each (type, strategy) combination
    let total_combos = (0..nt1 * nt2 * n1 * n2).map(|_| true).count(); // just count

    for s1_types in 0..(n1 as u64).pow(nt1 as u32) {
        for s2_types in 0..(n2 as u64).pow(nt2 as u32) {
            // Decode strategy rules
            let mut rule1: Vec<usize> = Vec::new();
            let mut temp = s1_types;
            for _ in 0..nt1 {
                rule1.push((temp % n1 as u64) as usize);
                temp /= n1 as u64;
            }
            let mut rule2: Vec<usize> = Vec::new();
            temp = s2_types;
            for _ in 0..nt2 {
                rule2.push((temp % n2 as u64) as usize);
                temp /= n2 as u64;
            }

            // Check if this is a Bayesian NE
            if is_bayesian_ne(game, &rule1, &rule2) {
                let mut strat_rules = Vec::new();
                let mut sr1 = Vec::new();
                for &s in &rule1 {
                    let mut v = DVector::zeros(n1);
                    v[s] = 1.0;
                    sr1.push(v);
                }
                let mut sr2 = Vec::new();
                for &s in &rule2 {
                    let mut v = DVector::zeros(n2);
                    v[s] = 1.0;
                    sr2.push(v);
                }
                strat_rules.push(sr1);
                strat_rules.push(sr2);
                equilibria.push(BayesianNashEquilibrium { strategy_rules: strat_rules });
            }
        }
    }
    equilibria
}

fn is_bayesian_ne(game: &BayesianGame, rule1: &[usize], rule2: &[usize]) -> bool {
    let n1 = game.n_strategies[0];
    let n2 = game.n_strategies[1];
    let nt1 = game.type_spaces[0].len();
    let nt2 = game.type_spaces[1].len();

    // Check player 1: for each type, no deviation improves expected payoff
    for t1 in 0..nt1 {
        let current_payoff = expected_payoff_pure(game, 0, t1, rule1, rule2);
        for alt_s in 0..n1 {
            if alt_s == rule1[t1] { continue; }
            let mut alt_rule = rule1.to_vec();
            alt_rule[t1] = alt_s;
            let alt_payoff = expected_payoff_pure(game, 0, t1, &alt_rule, rule2);
            if alt_payoff > current_payoff + 1e-10 { return false; }
        }
    }

    // Check player 2
    for t2 in 0..nt2 {
        let current_payoff = expected_payoff_pure(game, 1, t2, rule1, rule2);
        for alt_s in 0..n2 {
            if alt_s == rule2[t2] { continue; }
            let mut alt_rule = rule2.to_vec();
            alt_rule[t2] = alt_s;
            let alt_payoff = expected_payoff_pure(game, 1, t2, rule1, &alt_rule);
            if alt_payoff > current_payoff + 1e-10 { return false; }
        }
    }
    true
}

fn expected_payoff_pure(game: &BayesianGame, player: usize, player_type: usize, rule1: &[usize], rule2: &[usize]) -> f64 {
    let nt_other = if player == 0 { game.type_spaces[1].len() } else { game.type_spaces[0].len() };
    let mut total = 0.0;

    for t_other in 0..nt_other {
        let (types, s1, s2) = if player == 0 {
            let types = vec![player_type, t_other];
            (types, rule1[player_type], rule2[t_other])
        } else {
            let types = vec![t_other, player_type];
            (types, rule1[t_other], rule2[player_type])
        };
        let prob = game.type_profile_probability(&types);
        let strats = vec![s1, s2];
        total += prob * game.get_payoff(player, &types, &strats);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_bayesian_game() -> BayesianGame {
        let mut game = BayesianGame::new(
            2,
            vec![
                vec![
                    PlayerType { id: "high".into(), prior_probability: 0.5 },
                    PlayerType { id: "low".into(), prior_probability: 0.5 },
                ],
                vec![
                    PlayerType { id: "high".into(), prior_probability: 0.5 },
                    PlayerType { id: "low".into(), prior_probability: 0.5 },
                ],
            ],
            vec![2, 2],
        );

        // Set payoffs for all type/strategy combinations
        for t1 in 0..2 {
            for t2 in 0..2 {
                for s1 in 0..2 {
                    for s2 in 0..2 {
                        let p1 = if t1 == 0 { (s1 + 1) as f64 * (s2 + 1) as f64 } else { (2 - s1) as f64 };
                        let p2 = if t2 == 0 { (s2 + 1) as f64 * (s1 + 1) as f64 } else { (2 - s2) as f64 };
                        game.set_payoff(0, &[t1, t2], &[s1, s2], p1);
                        game.set_payoff(1, &[t1, t2], &[s1, s2], p2);
                    }
                }
            }
        }
        game
    }

    #[test]
    fn test_bayesian_game_creation() {
        let game = make_simple_bayesian_game();
        assert_eq!(game.n_players, 2);
        assert_eq!(game.type_spaces[0].len(), 2);
    }

    #[test]
    fn test_type_profile_probability() {
        let game = make_simple_bayesian_game();
        let prob = game.type_profile_probability(&[0, 0]);
        assert!((prob - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_harsanyi_transform() {
        let game = make_simple_bayesian_game();
        let transformed = game.harsanyi_transform();
        assert_eq!(transformed.n_transformed(), 4); // 2 players × 2 types
    }

    #[test]
    fn test_solve_bayesian() {
        let game = make_simple_bayesian_game();
        let equilibria = solve_bayesian_pure(&game);
        assert!(!equilibria.is_empty());
    }

    #[test]
    fn test_payoff_access() {
        let game = make_simple_bayesian_game();
        let v = game.get_payoff(0, &[0, 0], &[0, 0]);
        assert!(v > 0.0);
    }
}
