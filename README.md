# lau-game-theory-agents

> Strategic interaction for agents: Nash equilibria, mechanism design, cooperative games, auctions, and evolutionary dynamics.

## What This Does

This crate provides comprehensive game theory for multi-agent systems. It covers normal-form games (payoff matrices, dominance, iterated elimination), Nash equilibrium computation via support enumeration, extensive-form games with backward induction, Bayesian games with incomplete information, mechanism design (VCG auctions, incentive compatibility), cooperative games (Shapley value, core, nucleolus), evolutionary game theory (replicator dynamics, ESS), auction theory, and bargaining solutions (Nash, Rubinstein, Kalai-Smorodinsky).

Use this when agents must reason strategically about each other's behavior — not just optimizing independently, but accounting for the fact that other agents are optimizing too.

## The Key Idea

Game theory studies what happens when multiple decision-makers interact. A **Nash equilibrium** is a profile of strategies where no player can improve by unilaterally deviating — everyone is playing a best response to everyone else. This crate computes these equilibria, verifies them, and extends the framework to incomplete information (Bayesian games), sequential interaction (extensive form), coalition formation (cooperative games), and population-level dynamics (evolutionary game theory).

## Install

```bash
cargo add lau-game-theory-agents
```

## Quick Start

```rust
use lau_game_theory_agents::*;
use nalgebra::{DVector, DMatrix};

fn main() {
    // Prisoner's Dilemma
    let game = NormalFormGame::prisoners_dilemma();
    println!("{}", game);

    // Find all Nash equilibria
    let equilibria = nash::support_enumeration(&game);
    for ne in &equilibria {
        println!("NE: P1 plays {:?}, P2 plays {:?}", ne.strategies[0], ne.strategies[1]);
        println!("Payoffs: {:?}", ne.expected_payoffs);
    }

    // Extensive-form game: Centipede
    let centipede = ExtensiveGame::new(
        extensive::centipede_game(3).root
    );
    let result = centipede.backward_induction();
    println!("SPE value: {:?}", result.value);

    // Cooperative game: Shapley value
    let coop = CooperativeGame::new(3, |coalition| {
        // Example: v(S) = |S|^2
        (coalition.len() as f64).powi(2)
    });
    println!("Shapley values: {:?}", coop.shapley_value());

    // Auction
    let auction = AuctionConfig::second_price(vec![10.0, 8.0, 6.0]);
    let outcome = auction.resolve();
    println!("Winner: player {}, price: {:.2}", outcome.winner, outcome.price);
}
```

## API Reference

### Normal-Form Games

#### `NormalFormGame`
A strategic-form game with payoff matrices.

```rust
// Classic games
let pd = NormalFormGame::prisoners_dilemma();
let mp = NormalFormGame::matching_pennies();
let bos = NormalFormGame::battle_of_the_sexes();
let sh = NormalFormGame::stag_hunt();
let ch = NormalFormGame::chicken();
let rps = NormalFormGame::rock_paper_scissors();

// Custom game
let game = NormalFormGame::from_arrays(2, 2,
    &[3.0, 0.0, 5.0, 1.0],  // player 1 payoffs
    &[3.0, 5.0, 0.0, 1.0],  // player 2 payoffs
);

// From matrices
let game = NormalFormGame::two_player(row_matrix, col_matrix);

// Payoff lookup
game.payoff(player, &[row_strat, col_strat]);

// Dominance
game.is_strictly_dominated(player, strategy);
game.strictly_dominates(player, s1, s2);
game.dominated_strategies(player);

// Best response
let br = game.best_responses(player, &opponent_mixed);

// Iterated Elimination of Strictly Dominated Strategies
let surviving = game.iesds();

// Expected payoff under mixed strategies
let val = game.expected_payoff(player, &[p, q]);
```

### Nash Equilibrium

#### `support_enumeration`
Find all Nash equilibria of a 2-player game by enumerating strategy supports.

```rust
let equilibria = nash::support_enumeration(&game);
for ne in &equilibria {
    // ne.strategies[0] — mixed strategy for player 1
    // ne.strategies[1] — mixed strategy for player 2
    // ne.expected_payoffs — expected payoff per player
}
```

#### `best_response_dynamics`
Iterative best-response search for Nash equilibrium.

```rust
let ne = nash::best_response_dynamics(&game, 100);
```

#### `verify_nash`
Check if a given strategy profile is a Nash equilibrium.

```rust
assert!(nash::verify_nash(&game, &ne));
```

### Extensive-Form Games

#### `ExtensiveGame`
A game tree with decision nodes, chance nodes, and terminal payoffs.

```rust
let tree = GameNode::decision(0, "root", vec![
    ("Left".into(), GameNode::decision(1, "after_L", vec![
        ("Up".into(), GameNode::terminal(vec![3.0, 1.0])),
        ("Down".into(), GameNode::terminal(vec![1.0, 3.0])),
    ])),
    ("Right".into(), GameNode::terminal(vec![2.0, 2.0])),
]);
let game = ExtensiveGame::new(tree);
```

#### `GameNode`

```rust
GameNode::terminal(payoffs)
GameNode::decision(player, info_set, children)
GameNode::chance(probabilities, outcomes)
```

#### Backward Induction

```rust
let result = game.backward_induction();
// result.strategies[player] = Vec<(info_set, action_index)>
// result.value = expected payoffs at root
```

#### Pre-built Games

```rust
let game = centipede_game(3);              // n-stage centipede
let game = ultimatum_game(10.0, 5.0);     // proposer/responder
```

### Bayesian Games

#### `BayesianGame`
Games with incomplete information — players have private types.

```rust
let mut bg = BayesianGame::new(2, type_spaces, n_strategies);
bg.set_payoff(player, &[type1, type2], &[strat1, strat2], payoff);
bg.get_payoff(player, &[type1, type2], &[strat1, strat2]);
bg.type_profile_probability(&[t1, t2]);
```

#### Harsanyi Transformation

```rust
let harsanyi = bg.harsanyi_transform();
harsanyi.n_transformed(); // one "player" per (player, type) pair
```

#### `solve_bayesian_pure`
Brute-force search for pure-strategy Bayesian Nash equilibria.

### Mechanism Design

#### `Mechanism`
Design rules where truthful reporting is optimal.

```rust
// VCG mechanism (truthful bidding is a dominant strategy)
let outcome = mechanism::vcg(&valuations);

// Incentive compatibility verification
mechanism::is_incentive_compatible(&mechanism, &types);

// Individual rationality
mechanism::is_individually_rational(&mechanism, &types);
```

### Cooperative Games

#### `CooperativeGame`
A game where players form coalitions.

```rust
let game = CooperativeGame::new(n_players, |coalition| {
    // characteristic function v(S)
    coalition.len() as f64
});

// Shapley value: fair allocation
let sv = game.shapley_value();

// Core: set of allocations no coalition can improve upon
let core = game.core();

// Nucleolus: lexicographically minimal excess
let nuc = game.nucleolus();

// Check stability
game.is_in_core(&allocation);
```

### Evolutionary Game Theory

#### Replicator Dynamics

```rust
// dx_i/dt = x_i * (f_i(x) - f̄(x))
let next = evolutionary::replicator_step(&population, &game, dt);
```

#### Evolutionarily Stable Strategy (ESS)

```rust
let is_ess = evolutionary::is_ess(&strategy, &game);
```

### Auction Theory

#### `AuctionConfig`

```rust
let first_price = AuctionConfig::first_price(vec![10.0, 8.0, 6.0]);
let second_price = AuctionConfig::second_price(vec![10.0, 8.0, 6.0]);
let outcome = auction.resolve();
// outcome.winner, outcome.price, outcome.revenue
```

### Bargaining

#### `BargainingProblem`

```rust
let problem = BargainingProblem::new(disagreement_point, feasible_set);

// Nash bargaining solution: maximize (u1-d1)(u2-d2)
let nash_sol = problem.nash_solution();

// Kalai-Smorodinsky solution
let ks_sol = problem.kalai_smorodinsky();

// Rubinstein alternating offers
let rubinstein = problem.rubinstein(discount_factor);
```

### Agent Integration

#### `AgentGame`
Unified API combining all game types.

```rust
let ag = AgentGame::new(GameType::NormalForm(game));
let solution = ag.solve();
// solution.equilibria, solution.payoffs, solution.recommendations
```

#### `StrategyRecommendation`

```rust
pub struct StrategyRecommendation {
    pub player: usize,
    pub strategy: DVector<f64>,  // mixed strategy
    pub expected_payoff: f64,
    pub rationale: String,
}
```

## How It Works

**Nash equilibrium** is found via support enumeration: for each possible support (set of strategies played with positive probability), solve the linear system of indifference conditions. If the solution is a valid probability distribution and no player can deviate profitably, it's a Nash equilibrium.

**Backward induction** traverses the game tree from leaves to root. At each decision node, the acting player picks the action leading to the highest payoff subtree. The resulting strategy profile is a subgame-perfect equilibrium (SPE).

**Bayesian games** use the Harsanyi transformation to convert incomplete information (type uncertainty) into a larger complete-information game. Pure-strategy BNEs are found by brute-force enumeration of type-contingent strategy rules.

**Cooperative games** compute the Shapley value by averaging marginal contributions over all player orderings: φᵢ = (1/n!) Σ |π| [v(S_π(i) ∪ {i}) − v(S_π(i))]. The core is the set of allocations x where Σᵢ xᵢ = v(N) and Σᵢ∈S xᵢ ≥ v(S) for all S.

**Replicator dynamics** model population evolution: dx_i/dt = xᵢ(fᵢ(x) − f̄(x)), where strategies with above-average fitness grow. An ESS is a strategy that, if adopted by the population, cannot be invaded by any mutant.

## The Math

### Nash Equilibrium

A strategy profile σ* is a Nash equilibrium if for every player i:

$$u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i, \sigma_{-i}^*) \quad \forall \sigma_i$$

### Support Enumeration

For support (S₁, S₂), solve the linear system:

$$\sum_j A[s_1, j] \cdot q_j = v \quad \forall s_1 \in S_1$$
$$\sum_{j \in S_2} q_j = 1, \quad q_j \geq 0$$

### Backward Induction (Subgame Perfection)

At each node, player i selects:

$$a^* = \arg\max_a \, V_i(\text{subtree}(a))$$

### Shapley Value

$$\varphi_i = \frac{1}{n!} \sum_{\pi} \left[ v(S_{\pi(i)} \cup \{i\}) - v(S_{\pi(i)}) \right]$$

### Replicator Dynamics

$$\dot{x}_i = x_i \left( f_i(\mathbf{x}) - \bar{f}(\mathbf{x}) \right)$$

### VCG Mechanism

Each agent pays the externality they impose:

$$p_i = \max_{a} \sum_{j \neq i} v_j(a) - \sum_{j \neq i} v_j(a^*)$$

### Nash Bargaining Solution

$$\max_{(u_1, u_2) \in S} (u_1 - d_1)(u_2 - d_2)$$

## License

MIT
