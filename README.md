# lau-game-theory-agents

Game theory for multi-agent strategic interaction — Nash equilibria, mechanism design, cooperative games, and evolutionary dynamics.

## Features

- **Normal Form Games**: Payoff matrices, dominance, best response, IESDS
- **Nash Equilibria**: Support enumeration, best response dynamics, verification
- **Bayesian Games**: Incomplete information, Harsanyi transformation
- **Extensive Form Games**: Game trees, backward induction, subgame perfection
- **Mechanism Design**: VCG mechanism, incentive compatibility, social choice
- **Cooperative Games**: Shapley value, core, nucleolus, Banzhaf index
- **Evolutionary Dynamics**: Replicator dynamics, ESS, Moran process
- **Auction Theory**: First/second price, all-pay, optimal auctions (Myerson)
- **Bargaining**: Nash solution, Kalai-Smorodinsky, Rubinstein alternating offers
- **Agent Game API**: Unified strategic reasoning interface

## Usage

```rust
use lau_game_theory_agents::{AgentGame, NormalFormGame, CooperativeGame};

// Normal form game
let game = NormalFormGame::prisoners_dilemma();
let ag = AgentGame::normal_form(game);
let solution = ag.solve();
let rec = ag.recommend(0);

// Cooperative game
let game = CooperativeGame::from_fn(3, |coalition| {
    coalition.count_ones() as f64 * 2.0
});
let shapley = game.shapley_value();
```

## Dependencies

- `nalgebra` — Linear algebra
- `serde` — Serialization

## License

MIT
