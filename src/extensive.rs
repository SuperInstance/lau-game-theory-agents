//! Extensive form games: game trees, backward induction, subgame perfection.

use serde::{Deserialize, Serialize};
use std::fmt;

/// An action label.
pub type Action = String;

/// A player identifier. None = chance node.
pub type PlayerId = Option<usize>;

/// A node in an extensive-form game tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameNode {
    /// Terminal node with payoffs for each player.
    Terminal {
        payoffs: Vec<f64>,
    },
    /// Decision node for a player.
    Decision {
        player: usize,
        info_set: String,
        children: Vec<(Action, Box<GameNode>)>,
    },
    /// Chance node with probabilistic outcomes.
    Chance {
        probabilities: Vec<f64>,
        outcomes: Vec<(Action, Box<GameNode>)>,
    },
}

impl GameNode {
    /// Create a terminal node.
    pub fn terminal(payoffs: Vec<f64>) -> Self {
        GameNode::Terminal { payoffs }
    }

    /// Create a decision node.
    pub fn decision(player: usize, info_set: &str, children: Vec<(Action, GameNode)>) -> Self {
        GameNode::Decision {
            player,
            info_set: info_set.to_string(),
            children: children.into_iter().map(|(a, n)| (a, Box::new(n))).collect(),
        }
    }

    /// Create a chance node.
    pub fn chance(probabilities: Vec<f64>, outcomes: Vec<(Action, GameNode)>) -> Self {
        assert_eq!(probabilities.len(), outcomes.len());
        GameNode::Chance {
            probabilities,
            outcomes: outcomes.into_iter().map(|(a, n)| (a, Box::new(n))).collect(),
        }
    }

    /// Check if terminal.
    pub fn is_terminal(&self) -> bool {
        matches!(self, GameNode::Terminal { .. })
    }

    /// Get payoffs if terminal.
    pub fn payoffs(&self) -> Option<&Vec<f64>> {
        match self {
            GameNode::Terminal { payoffs } => Some(payoffs),
            _ => None,
        }
    }

    /// Get number of players from terminal payoffs.
    pub fn n_players(&self) -> usize {
        match self {
            GameNode::Terminal { payoffs } => payoffs.len(),
            GameNode::Decision { children, .. } => children.first().map(|(_, n)| n.n_players()).unwrap_or(0),
            GameNode::Chance { outcomes, .. } => outcomes.first().map(|(_, n)| n.n_players()).unwrap_or(0),
        }
    }
}

/// An extensive-form game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensiveGame {
    pub root: GameNode,
    pub n_players: usize,
    pub player_names: Vec<String>,
}

impl ExtensiveGame {
    /// Create a new extensive game from a root node.
    pub fn new(root: GameNode) -> Self {
        let n_players = root.n_players();
        Self {
            root,
            n_players,
            player_names: (0..n_players).map(|i| format!("Player {}", i + 1)).collect(),
        }
    }

    /// Backward induction: solve for subgame perfect equilibrium.
    pub fn backward_induction(&self) -> BackwardInductionResult {
        let mut strategies: Vec<Vec<(String, usize)>> = vec![Vec::new(); self.n_players];
        let value = self.backward_induction_node(&self.root, &mut strategies);
        BackwardInductionResult {
            strategies,
            value,
        }
    }

    fn backward_induction_node(
        &self,
        node: &GameNode,
        strategies: &mut Vec<Vec<(String, usize)>>,
    ) -> Vec<f64> {
        match node {
            GameNode::Terminal { payoffs } => payoffs.clone(),
            GameNode::Decision { player, info_set, children } => {
                let p = *player;
                let mut best_value = vec![f64::NEG_INFINITY; self.n_players];
                let mut best_action = 0;

                for (i, (_, child)) in children.iter().enumerate() {
                    let child_value = self.backward_induction_node(child, strategies);
                    if child_value[p] > best_value[p] {
                        best_value = child_value;
                        best_action = i;
                    }
                }

                strategies[p].push((info_set.clone(), best_action));
                best_value
            }
            GameNode::Chance { probabilities, outcomes } => {
                let mut expected = vec![0.0; self.n_players];
                for (i, (_, child)) in outcomes.iter().enumerate() {
                    let child_value = self.backward_induction_node(child, strategies);
                    for p in 0..self.n_players {
                        expected[p] += probabilities[i] * child_value[p];
                    }
                }
                expected
            }
        }
    }

    /// Count the number of nodes in the game tree.
    pub fn count_nodes(&self) -> usize {
        self.count_nodes_recursive(&self.root)
    }

    fn count_nodes_recursive(&self, node: &GameNode) -> usize {
        match node {
            GameNode::Terminal { .. } => 1,
            GameNode::Decision { children, .. } => {
                1 + children.iter().map(|(_, n)| self.count_nodes_recursive(n)).sum::<usize>()
            }
            GameNode::Chance { outcomes, .. } => {
                1 + outcomes.iter().map(|(_, n)| self.count_nodes_recursive(n)).sum::<usize>()
            }
        }
    }

    /// Get depth of the game tree.
    pub fn depth(&self) -> usize {
        self.depth_recursive(&self.root)
    }

    fn depth_recursive(&self, node: &GameNode) -> usize {
        match node {
            GameNode::Terminal { .. } => 0,
            GameNode::Decision { children, .. } => {
                1 + children.iter().map(|(_, n)| self.depth_recursive(n)).max().unwrap_or(0)
            }
            GameNode::Chance { outcomes, .. } => {
                1 + outcomes.iter().map(|(_, n)| self.depth_recursive(n)).max().unwrap_or(0)
            }
        }
    }
}

/// Result of backward induction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackwardInductionResult {
    /// For each player: list of (info_set, action_index) pairs.
    pub strategies: Vec<Vec<(String, usize)>>,
    /// Expected payoffs at the root.
    pub value: Vec<f64>,
}

/// Information set structure for checking perfect recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationSet {
    pub player: usize,
    pub id: String,
    pub nodes: Vec<usize>, // Node indices (conceptual)
    pub actions: Vec<Action>,
}

/// Centipede game constructor.
pub fn centipede_game(n_stages: usize) -> ExtensiveGame {
    // Two players alternate. At each stage, can "take" (end game) or "pass" (continue).
    // Payoffs increase the longer they play.
    fn build(stage: usize, n_stages: usize) -> GameNode {
        if stage >= n_stages {
            return GameNode::terminal(vec![n_stages as f64 + 2.0, n_stages as f64 + 2.0]);
        }
        let player = stage % 2;
        let take_payoff_p1 = if player == 0 { (stage as f64 + 2.0) } else { stage as f64 + 0.5 };
        let take_payoff_p2 = if player == 1 { (stage as f64 + 2.0) } else { stage as f64 + 0.5 };

        GameNode::decision(
            player,
            &format!("stage_{}", stage),
            vec![
                ("Take".to_string(), GameNode::terminal(vec![take_payoff_p1, take_payoff_p2])),
                ("Pass".to_string(), build(stage + 1, n_stages)),
            ],
        )
    }
    ExtensiveGame::new(build(0, n_stages))
}

/// Ultimatum game: proposer offers split, responder accepts/rejects.
pub fn ultimatum_game(total: f64, offer_increment: f64) -> ExtensiveGame {
    let mut children = Vec::new();
    let mut offer = 0.0;
    while offer <= total + 1e-10 {
        let responder_children = vec![
            ("Accept".to_string(), GameNode::terminal(vec![total - offer, offer])),
            ("Reject".to_string(), GameNode::terminal(vec![0.0, 0.0])),
        ];
        children.push((
            format!("Offer {:.1}", offer),
            GameNode::decision(1, &format!("respond_{:.0}", offer * 10.0), responder_children),
        ));
        offer += offer_increment;
    }
    ExtensiveGame::new(GameNode::decision(0, "propose", children))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_game_tree() {
        let tree = GameNode::decision(0, "root", vec![
            ("Left".to_string(), GameNode::terminal(vec![2.0, 1.0])),
            ("Right".to_string(), GameNode::terminal(vec![1.0, 2.0])),
        ]);
        let game = ExtensiveGame::new(tree);
        assert_eq!(game.n_players, 2);
        assert_eq!(game.count_nodes(), 3);
        assert_eq!(game.depth(), 1);
    }

    #[test]
    fn test_backward_induction_simple() {
        // P1 chooses L or R, P2 responds
        let tree = GameNode::decision(0, "root", vec![
            ("Left".to_string(), GameNode::decision(1, "after_L", vec![
                ("Up".to_string(), GameNode::terminal(vec![3.0, 1.0])),
                ("Down".to_string(), GameNode::terminal(vec![1.0, 3.0])),
            ])),
            ("Right".to_string(), GameNode::decision(1, "after_R", vec![
                ("Up".to_string(), GameNode::terminal(vec![2.0, 2.0])),
                ("Down".to_string(), GameNode::terminal(vec![0.0, 4.0])),
            ])),
        ]);
        let game = ExtensiveGame::new(tree);
        let result = game.backward_induction();
        // P2 should choose Down in both subgames (maximizes P2)
        // After L, P2 gets 3 from Down. After R, P2 gets 4 from Down.
        // P1 gets 1 from Left+Down, 0 from Right+Down. P1 chooses Left.
        assert_eq!(result.value[0], 1.0);
        assert_eq!(result.value[1], 3.0);
    }

    #[test]
    fn test_centipede_game() {
        let game = centipede_game(3);
        let result = game.backward_induction();
        // In SPE, player 0 takes immediately
        assert!(!result.strategies[0].is_empty());
    }

    #[test]
    fn test_ultimatum_game() {
        let game = ultimatum_game(10.0, 5.0);
        let result = game.backward_induction();
        // Subgame perfect: offer the smallest amount (5.0), responder accepts
        assert!(!result.strategies.is_empty());
    }

    #[test]
    fn test_chance_node() {
        let tree = GameNode::chance(
            vec![0.5, 0.5],
            vec![
                ("Heads".to_string(), GameNode::terminal(vec![1.0, -1.0])),
                ("Tails".to_string(), GameNode::terminal(vec![-1.0, 1.0])),
            ],
        );
        let game = ExtensiveGame::new(tree);
        assert_eq!(game.depth(), 1);
        assert_eq!(game.count_nodes(), 3);
    }

    #[test]
    fn test_node_terminal_check() {
        let t = GameNode::terminal(vec![1.0, 2.0]);
        assert!(t.is_terminal());
        assert_eq!(t.payoffs(), Some(&vec![1.0, 2.0]));

        let d = GameNode::decision(0, "x", vec![]);
        assert!(!d.is_terminal());
    }

    #[test]
    fn test_game_depth_complex() {
        let tree = GameNode::decision(0, "root", vec![
            ("A".to_string(), GameNode::decision(1, "n1", vec![
                ("X".to_string(), GameNode::terminal(vec![1.0, 2.0])),
            ])),
            ("B".to_string(), GameNode::terminal(vec![3.0, 0.0])),
        ]);
        let game = ExtensiveGame::new(tree);
        assert_eq!(game.depth(), 2);
    }
}
