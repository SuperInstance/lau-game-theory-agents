# lau-game-theory-agents

> Strategic reasoning for agents: Nash equilibria, mechanism design, auctions, bargaining, and evolutionary dynamics

## What This Does

This crate implements game theory for multi-agent strategic interaction. It covers normal form games, extensive form games (game trees with backward induction), Nash equilibrium computation, cooperative games (Shapley value, core, nucleolus), Bayesian games (incomplete information), mechanism design (VCG), auction theory, bargaining solutions, and evolutionary dynamics (replicator dynamics, ESS).

## The Key Idea

When agents share an environment, they need to reason about *each other*. Game theory provides the math: payoff matrices encode incentives, Nash equilibria identify stable outcomes where no agent wants to deviate, mechanism design lets you engineer the rules so selfish agents produce good outcomes, and evolutionary dynamics model how strategies evolve over time. This crate puts all of that in one place with a unified `AgentGame` API.

## Install

```toml
[dependencies]
lau-game-theory-agents = { git = "https://github.com/SuperInstance/lau-game-theory-agents" }
```

## Quick Start

```rust
use lau_game_theory_agents::*;
use nalgebra::DMatrix;

// Normal Form Game
let payoff_a = DMatrix::from_row_slice(2, 2, &[3, 0, 5, 1]);
let payoff_b = DMatrix::from_row_slice(2, 2, &[3, 5, 0, 1]);
let game = NormalFormGame::new(2, 2, vec![payoff_a, payoff_b]);
let equilibria = nash::find_equilibria(&game);

// Cooperative Game (Shapley Value)
let coop = CooperativeGame::from_fn(3, |mask| match mask.count_ones() {
    3 => 1.0, _ => 0.0
});
let shapley = coop.shapley_value();

// Auction (Vickrey / Second Price)
let result = auction::vickrey(&[10.0, 8.0, 12.0], 5.0);

// Bargaining
let problem = BargainingProblem::linear_frontier(100.0, [0.0, 0.0]);
let sol = bargaining::nash_bargaining_solution(&problem);

// Unified API
let agent_game = AgentGame::normal_form(game);
let analysis = agent_game.analyze();
```

## API Reference

### `normal_form` — NormalFormGame, payoff matrices, dominance, best response, Pareto optimality
### `nash` — NashEquilibrium, find_equilibria (support enumeration)
### `extensive` — GameNode (Terminal/Decision/Chance), backward induction, subgame perfection
### `cooperative` — CooperativeGame, shapley_value(), core(), nucleolus()
### `bayesian` — BayesianGame, Harsanyi transformation, BNE
### `mechanism` — Mechanism, VCG, efficient_allocation(), incentive compatibility
### `auction` — first_price_sealed_bid(), vickrey(), revenue equivalence
### `bargaining` — nash_bargaining_solution(), kalai_smorodinsky(), rubinstein()
### `evolutionary` — replicator_dynamics(), ESS check, fitness landscapes
### `agent_game` — AgentGame unified API, StrategicAnalysis, StrategyRecommendation

## How It Works

- **Nash Equilibria**: Support enumeration — try all support sets, solve complementarity conditions.
- **Shapley Value**: Average marginal contribution over all permutations.
- **VCG**: Charge each agent the externality they impose on others.
- **Replicator Dynamics**: dxᵢ/dt = xᵢ(fᵢ − f̄) where fᵢ = (Ax)ᵢ.

## Testing

75 tests covering all game types, equilibrium computation, mechanism design, and evolutionary dynamics.

## License

MIT
