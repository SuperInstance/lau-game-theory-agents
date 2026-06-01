//! Unified agent game API: strategic reasoning over game theory primitives.

use serde::{Deserialize, Serialize};
use nalgebra::DVector;
use crate::normal_form::NormalFormGame;
use crate::nash::{self, NashEquilibrium};
use crate::extensive::{ExtensiveGame, BackwardInductionResult};
use crate::cooperative::CooperativeGame;
use crate::auction::{self, AuctionConfig, AuctionResult};
use crate::bargaining::{self, BargainingProblem};
use crate::evolutionary::{self, EvolutionaryConfig, EvolutionaryResult};
use crate::mechanism::{self, Mechanism, VCGResult};

/// Type of game being played.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameType {
    NormalForm(NormalFormGame),
    Extensive(ExtensiveGame),
    Cooperative(CooperativeGame),
    Auction(AuctionConfig),
    Bargaining(BargainingProblem),
}

/// Strategic analysis of a game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicAnalysis {
    pub game_type: String,
    pub n_players: usize,
    pub equilibria: Vec<NashEquilibrium>,
    pub dominant_strategies: Vec<Vec<usize>>,
    pub pareto_optimal: Vec<Vec<usize>>,
}

/// AgentGame: unified API for game-theoretic reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGame {
    pub game: GameType,
    pub name: String,
    pub description: String,
}

impl AgentGame {
    /// Create a normal form agent game.
    pub fn normal_form(game: NormalFormGame) -> Self {
        Self {
            game: GameType::NormalForm(game),
            name: "Normal Form Game".into(),
            description: "Strategic form game with payoff matrices".into(),
        }
    }

    /// Create an extensive form agent game.
    pub fn extensive(game: ExtensiveGame) -> Self {
        Self {
            game: GameType::Extensive(game),
            name: "Extensive Form Game".into(),
            description: "Sequential game with game tree".into(),
        }
    }

    /// Create a cooperative agent game.
    pub fn cooperative(game: CooperativeGame) -> Self {
        Self {
            game: GameType::Cooperative(game),
            name: "Cooperative Game".into(),
            description: "Coalitional game with characteristic function".into(),
        }
    }

    /// Create an auction agent game.
    pub fn auction(config: AuctionConfig) -> Self {
        Self {
            game: GameType::Auction(config),
            name: "Auction".into(),
            description: "Auction mechanism".into(),
        }
    }

    /// Create a bargaining agent game.
    pub fn bargaining(problem: BargainingProblem) -> Self {
        Self {
            game: GameType::Bargaining(problem),
            name: "Bargaining Problem".into(),
            description: "Negotiation between two parties".into(),
        }
    }

    /// Perform strategic analysis of the game.
    pub fn analyze(&self) -> StrategicAnalysis {
        match &self.game {
            GameType::NormalForm(g) => {
                let equilibria = nash::support_enumeration(g);
                let dominant: Vec<Vec<usize>> = (0..g.n_players)
                    .map(|p| {
                        (0..g.n_strategies[p])
                            .filter(|&s| {
                                (0..g.n_strategies[p])
                                    .filter(|&other| other != s)
                                    .all(|other| g.strictly_dominates(p, s, other))
                            })
                            .collect()
                    })
                    .collect();

                StrategicAnalysis {
                    game_type: "Normal Form".into(),
                    n_players: g.n_players,
                    equilibria,
                    dominant_strategies: dominant,
                    pareto_optimal: vec![],
                }
            }
            GameType::Extensive(g) => {
                StrategicAnalysis {
                    game_type: "Extensive Form".into(),
                    n_players: g.n_players,
                    equilibria: vec![],
                    dominant_strategies: vec![],
                    pareto_optimal: vec![],
                }
            }
            GameType::Cooperative(g) => {
                StrategicAnalysis {
                    game_type: "Cooperative".into(),
                    n_players: g.n_players,
                    equilibria: vec![],
                    dominant_strategies: vec![],
                    pareto_optimal: vec![],
                }
            }
            GameType::Auction(config) => {
                StrategicAnalysis {
                    game_type: "Auction".into(),
                    n_players: config.n_bidders,
                    equilibria: vec![],
                    dominant_strategies: vec![],
                    pareto_optimal: vec![],
                }
            }
            GameType::Bargaining(_) => {
                StrategicAnalysis {
                    game_type: "Bargaining".into(),
                    n_players: 2,
                    equilibria: vec![],
                    dominant_strategies: vec![],
                    pareto_optimal: vec![],
                }
            }
        }
    }

    /// Solve the game: find equilibria / optimal solutions.
    pub fn solve(&self) -> GameSolution {
        match &self.game {
            GameType::NormalForm(g) => {
                let equilibria = nash::support_enumeration(g);
                GameSolution::NormalForm { equilibria }
            }
            GameType::Extensive(g) => {
                let result = g.backward_induction();
                GameSolution::ExtensiveForm { backward_induction: result }
            }
            GameType::Cooperative(g) => {
                let shapley = g.shapley_value();
                let in_core = g.is_in_core(&shapley);
                let nucleolus = g.nucleolus();
                GameSolution::Cooperative {
                    shapley_value: shapley,
                    in_core,
                    nucleolus,
                }
            }
            GameType::Auction(config) => {
                let fp = auction::first_price_sealed_bid(&config.valuations, config.reserve_price);
                let sp = auction::second_price_sealed_bid(&config.valuations, config.reserve_price);
                GameSolution::Auction {
                    first_price: fp,
                    second_price: sp,
                }
            }
            GameType::Bargaining(problem) => {
                let nash = bargaining::nash_bargaining_solution(problem);
                let ks = bargaining::kalai_smorodinsky_solution(problem);
                let egal = bargaining::egalitarian_solution(problem);
                GameSolution::Bargaining {
                    nash_solution: nash,
                    kalai_smorodinsky: ks,
                    egalitarian: egal,
                }
            }
        }
    }

    /// Simulate evolutionary dynamics (for symmetric normal form games).
    pub fn simulate_evolution(&self, initial_freq: DVector<f64>, config: EvolutionaryConfig) -> Option<EvolutionaryResult> {
        match &self.game {
            GameType::NormalForm(g) if g.n_players == 2 => {
                let payoff = &g.payoffs[0];
                Some(evolutionary::replicator_dynamics(payoff, &initial_freq, &config))
            }
            _ => None,
        }
    }

    /// Get recommendation for an agent.
    pub fn recommend(&self, player: usize) -> StrategyRecommendation {
        match &self.game {
            GameType::NormalForm(g) => {
                let equilibria = nash::support_enumeration(g);
                if equilibria.is_empty() {
                    return StrategyRecommendation {
                        action: "No equilibrium found".into(),
                        confidence: 0.0,
                        reasoning: "Game has no computable equilibrium in pure strategies".into(),
                    };
                }
                let ne = &equilibria[0];
                let strat = &ne.strategies[player];
                let best = (0..strat.len()).max_by(|&a, &b| strat[a].partial_cmp(&strat[b]).unwrap()).unwrap();
                StrategyRecommendation {
                    action: format!("Play strategy {} (probability {:.2})", best, strat[best]),
                    confidence: strat[best],
                    reasoning: format!("Based on Nash equilibrium analysis, {} equilibria found", equilibria.len()),
                }
            }
            GameType::Extensive(g) => {
                let result = g.backward_induction();
                if player < result.strategies.len() && !result.strategies[player].is_empty() {
                    let (info_set, action) = &result.strategies[player][0];
                    StrategyRecommendation {
                        action: format!("At {}: take action {}", info_set, action),
                        confidence: 1.0,
                        reasoning: "Subgame perfect equilibrium via backward induction".into(),
                    }
                } else {
                    StrategyRecommendation {
                        action: "No action needed".into(),
                        confidence: 1.0,
                        reasoning: "Player has no decision nodes".into(),
                    }
                }
            }
            GameType::Cooperative(g) => {
                let sv = g.shapley_value();
                StrategyRecommendation {
                    action: format!("Demand allocation {:.2}", sv[player]),
                    confidence: 0.8,
                    reasoning: "Based on Shapley value (fair allocation)".into(),
                }
            }
            GameType::Auction(config) => {
                let result = auction::second_price_sealed_bid(&config.valuations, config.reserve_price);
                StrategyRecommendation {
                    action: format!("Bid truthfully: {:.2}", config.valuations[player]),
                    confidence: 1.0,
                    reasoning: "In a Vickrey auction, truthful bidding is a dominant strategy".into(),
                }
            }
            GameType::Bargaining(problem) => {
                let nash_sol = bargaining::nash_bargaining_solution(problem);
                StrategyRecommendation {
                    action: format!("Propose ({:.2}, {:.2})", nash_sol[0], nash_sol[1]),
                    confidence: 0.7,
                    reasoning: "Nash bargaining solution maximizes joint gains".into(),
                }
            }
        }
    }
}

/// Game solution variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameSolution {
    NormalForm {
        equilibria: Vec<NashEquilibrium>,
    },
    ExtensiveForm {
        backward_induction: BackwardInductionResult,
    },
    Cooperative {
        shapley_value: Vec<f64>,
        in_core: bool,
        nucleolus: Vec<f64>,
    },
    Auction {
        first_price: AuctionResult,
        second_price: AuctionResult,
    },
    Bargaining {
        nash_solution: [f64; 2],
        kalai_smorodinsky: [f64; 2],
        egalitarian: [f64; 2],
    },
}

/// Strategy recommendation for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRecommendation {
    pub action: String,
    pub confidence: f64,
    pub reasoning: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normal_form::NormalFormGame;
    use crate::cooperative::CooperativeGame;
    use std::collections::HashMap;

    #[test]
    fn test_agent_game_normal_form() {
        let g = NormalFormGame::prisoners_dilemma();
        let ag = AgentGame::normal_form(g);
        let analysis = ag.analyze();
        assert_eq!(analysis.n_players, 2);
        assert!(!analysis.equilibria.is_empty());
    }

    #[test]
    fn test_agent_game_solve() {
        let g = NormalFormGame::prisoners_dilemma();
        let ag = AgentGame::normal_form(g);
        let solution = ag.solve();
        if let GameSolution::NormalForm { equilibria } = solution {
            assert!(!equilibria.is_empty());
        } else {
            panic!("Wrong solution type");
        }
    }

    #[test]
    fn test_agent_game_recommendation() {
        let g = NormalFormGame::prisoners_dilemma();
        let ag = AgentGame::normal_form(g);
        let rec = ag.recommend(0);
        assert!(rec.confidence > 0.0);
        assert!(!rec.action.is_empty());
    }

    #[test]
    fn test_agent_game_cooperative() {
        let mut char_fn = HashMap::new();
        char_fn.insert(0, 0.0);
        char_fn.insert(1, 1.0);
        char_fn.insert(2, 2.0);
        char_fn.insert(3, 4.0);
        let cg = CooperativeGame::new(2, char_fn);
        let ag = AgentGame::cooperative(cg);
        let solution = ag.solve();
        if let GameSolution::Cooperative { shapley_value, .. } = solution {
            assert_eq!(shapley_value.len(), 2);
        } else {
            panic!("Wrong solution type");
        }
    }

    #[test]
    fn test_agent_game_auction() {
        let config = AuctionConfig::new(vec![10.0, 8.0, 6.0]);
        let ag = AgentGame::auction(config);
        let solution = ag.solve();
        if let GameSolution::Auction { first_price, second_price } = solution {
            assert_eq!(first_price.winner, Some(0));
            assert_eq!(second_price.winner, Some(0));
        } else {
            panic!("Wrong solution type");
        }
    }

    #[test]
    fn test_agent_game_bargaining() {
        let problem = BargainingProblem::linear_frontier(10.0, [0.0, 0.0]);
        let ag = AgentGame::bargaining(problem);
        let solution = ag.solve();
        if let GameSolution::Bargaining { nash_solution, .. } = solution {
            assert!(nash_solution[0] > 0.0 && nash_solution[1] > 0.0);
        } else {
            panic!("Wrong solution type");
        }
    }

    #[test]
    fn test_agent_game_evolution() {
        let g = NormalFormGame::prisoners_dilemma();
        let ag = AgentGame::normal_form(g);
        let init = DVector::from_vec(vec![0.5, 0.5]);
        let config = EvolutionaryConfig::default();
        let result = ag.simulate_evolution(init, config);
        assert!(result.is_some());
    }
}
