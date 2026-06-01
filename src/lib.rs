//! # lau-game-theory-agents
//!
//! Game theory for multi-agent strategic interaction — Nash equilibria,
//! mechanism design, cooperative games, and evolutionary dynamics.
//!
//! ## Modules
//! - `normal_form` — Normal form games, payoff matrices, dominance, best response
//! - `nash` — Nash equilibrium computation (support enumeration)
//! - `bayesian` — Bayesian games, incomplete information
//! - `extensive` — Extensive form games, game trees, backward induction
//! - `mechanism` — Mechanism design, VCG, incentive compatibility
//! - `cooperative` — Cooperative games, Shapley value, core, nucleolus
//! - `evolutionary` — Evolutionary game theory, replicator dynamics, ESS
//! - `auction` — Auction theory, first/second price, optimal auctions
//! - `bargaining` — Nash bargaining, Rubinstein, Kalai-Smorodinsky
//! - `agent_game` — Unified API for strategic reasoning

pub mod normal_form;
pub mod nash;
pub mod bayesian;
pub mod extensive;
pub mod mechanism;
pub mod cooperative;
pub mod evolutionary;
pub mod auction;
pub mod bargaining;
pub mod agent_game;

pub use agent_game::{AgentGame, GameType, GameSolution, StrategyRecommendation};
pub use normal_form::NormalFormGame;
pub use nash::NashEquilibrium;
pub use extensive::ExtensiveGame;
pub use cooperative::CooperativeGame;
pub use mechanism::Mechanism;
pub use auction::AuctionConfig;
pub use bargaining::BargainingProblem;
