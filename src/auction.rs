//! Auction theory: first-price, second-price (Vickrey), revenue equivalence, optimal auctions.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// An auction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionConfig {
    pub n_bidders: usize,
    /// Valuations per bidder.
    pub valuations: Vec<f64>,
    /// Reserve price.
    pub reserve_price: f64,
}

impl AuctionConfig {
    pub fn new(valuations: Vec<f64>) -> Self {
        Self {
            n_bidders: valuations.len(),
            valuations,
            reserve_price: 0.0,
        }
    }

    pub fn with_reserve(mut self, reserve: f64) -> Self {
        self.reserve_price = reserve;
        self
    }
}

/// Result of an auction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionResult {
    pub winner: Option<usize>,
    pub payment: f64,
    pub revenue: f64,
    pub allocations: Vec<f64>, // 1.0 for winner, 0.0 otherwise
    pub surplus: Vec<f64>,     // Consumer surplus per bidder
}

/// First-price sealed-bid auction: highest bidder wins, pays their bid.
pub fn first_price_sealed_bid(bids: &[f64], reserve: f64) -> AuctionResult {
    let n = bids.len();
    let mut winner: Option<usize> = None;
    let mut highest_bid = reserve;

    for (i, &bid) in bids.iter().enumerate() {
        if bid > highest_bid {
            highest_bid = bid;
            winner = Some(i);
        }
    }

    let mut allocations = vec![0.0; n];
    let mut surplus = vec![0.0; n];
    if let Some(w) = winner {
        allocations[w] = 1.0;
    }

    AuctionResult {
        winner,
        payment: highest_bid,
        revenue: highest_bid,
        allocations,
        surplus,
    }
}

/// Second-price sealed-bid (Vickrey) auction: highest bidder wins, pays second-highest bid.
pub fn second_price_sealed_bid(bids: &[f64], reserve: f64) -> AuctionResult {
    let n = bids.len();
    let mut indexed_bids: Vec<(usize, f64)> = bids.iter().copied().enumerate()
        .filter(|(_, b)| *b >= reserve)
        .collect();

    indexed_bids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let (winner, payment) = if indexed_bids.len() >= 2 {
        (Some(indexed_bids[0].0), indexed_bids[1].1)
    } else if indexed_bids.len() == 1 {
        (Some(indexed_bids[0].0), reserve)
    } else {
        (None, 0.0)
    };

    let mut allocations = vec![0.0; n];
    let mut surplus = vec![0.0; n];
    if let Some(w) = winner {
        allocations[w] = 1.0;
        surplus[w] = bids[w] - payment;
    }

    AuctionResult {
        winner,
        payment,
        revenue: payment,
        allocations,
        surplus,
    }
}

/// English auction (ascending): equivalent to second-price in private value model.
pub fn english_auction(valuations: &[f64], reserve: f64) -> AuctionResult {
    // Bidders bid truthfully; outcome is same as second-price
    second_price_sealed_bid(valuations, reserve)
}

/// Dutch auction (descending): equivalent to first-price in private value model.
/// Optimal bid for bidder i with uniform [0,1] values and n bidders: b_i = (n-1)/n * v_i
pub fn dutch_auction(valuations: &[f64], reserve: f64) -> AuctionResult {
    let n = valuations.len();
    let bids: Vec<f64> = valuations.iter().map(|&v| ((n - 1) as f64 / n as f64) * v).collect();
    first_price_sealed_bid(&bids, reserve)
}

/// Compute optimal reserve price for a uniform [0,1] auction with n bidders.
/// For regular distributions: r* = φ^{-1}(0) where φ is virtual valuation.
/// For uniform[0,1]: r* = 1/2.
pub fn optimal_reserve_uniform(n_bidders: usize) -> f64 {
    0.5 // For uniform [0,1]
}

/// Revenue equivalence theorem verification:
/// Under certain conditions, all standard auction formats yield the same expected revenue.
pub fn revenue_equivalence_check(
    valuations: &[f64],
) -> RevenueEquivalenceResult {
    let n = valuations.len();
    let reserve = 0.0;

    // Expected revenue under different formats
    let fp = first_price_sealed_bid(valuations, reserve);
    let sp = second_price_sealed_bid(valuations, reserve);

    RevenueEquivalenceResult {
        first_price_revenue: fp.revenue,
        second_price_revenue: sp.revenue,
        equivalent: (fp.revenue - sp.revenue).abs() < 1e-8,
    }
}

/// Result of revenue equivalence check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevenueEquivalenceResult {
    pub first_price_revenue: f64,
    pub second_price_revenue: f64,
    pub equivalent: bool,
}

/// Myerson's optimal auction for regular distributions.
/// Computes virtual valuations and runs VCG-like mechanism.
pub fn optimal_auction(valuations: &[f64], distribution: ValueDistribution) -> AuctionResult {
    let virtual_vals: Vec<f64> = valuations.iter()
        .map(|&v| virtual_valuation(v, &distribution))
        .collect();

    // Winner is highest virtual valuation (if positive)
    let mut best: Option<(usize, f64)> = None;
    for (i, &vv) in virtual_vals.iter().enumerate() {
        if vv > 0.0 {
            if best.is_none() || vv > best.unwrap().1 {
                best = Some((i, vv));
            }
        }
    }

    match best {
        Some((winner, _)) => {
            let payment = critical_payment(winner, &virtual_vals, valuations, &distribution);
            let n = valuations.len();
            let mut allocations = vec![0.0; n];
            let mut surplus = vec![0.0; n];
            allocations[winner] = 1.0;
            surplus[winner] = valuations[winner] - payment;

            AuctionResult {
                winner: Some(winner),
                payment,
                revenue: payment,
                allocations,
                surplus,
            }
        }
        None => {
            let n = valuations.len();
            AuctionResult {
                winner: None,
                payment: 0.0,
                revenue: 0.0,
                allocations: vec![0.0; n],
                surplus: vec![0.0; n],
            }
        }
    }
}

/// Value distribution for auction analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueDistribution {
    Uniform { low: f64, high: f64 },
    Exponential { lambda: f64 },
    Normal { mean: f64, std_dev: f64 },
}

/// Virtual valuation: φ(v) = v - (1-F(v))/f(v)
fn virtual_valuation(v: f64, dist: &ValueDistribution) -> f64 {
    match dist {
        ValueDistribution::Uniform { low, high } => {
            // φ(v) = 2v - high for uniform[low, high]
            2.0 * v - high
        }
        ValueDistribution::Exponential { lambda } => {
            // φ(v) = v - 1/lambda
            v - 1.0 / lambda
        }
        ValueDistribution::Normal { mean, std_dev } => {
            // Approximation
            v - std_dev * std_dev / (v - mean).max(0.01)
        }
    }
}

fn critical_payment(winner: usize, virtual_vals: &[f64], valuations: &[f64], dist: &ValueDistribution) -> f64 {
    // Find the critical virtual valuation (second highest or 0)
    let mut second_virtual = 0.0f64;
    for (i, &vv) in virtual_vals.iter().enumerate() {
        if i != winner && vv > second_virtual {
            second_virtual = vv;
        }
    }

    // Invert virtual valuation to get payment
    match dist {
        ValueDistribution::Uniform { low, high } => {
            // φ(v) = 2v - high => v = (φ + high) / 2
            ((second_virtual + high) / 2.0).max(*low)
        }
        ValueDistribution::Exponential { lambda } => {
            second_virtual + 1.0 / lambda
        }
        ValueDistribution::Normal { .. } => {
            // Approximate inversion
            second_virtual
        }
    }
}

/// All-pay auction: everyone pays their bid, highest bidder wins.
pub fn all_pay_auction(bids: &[f64], reserve: f64) -> AuctionResult {
    let n = bids.len();
    let mut winner: Option<usize> = None;
    let mut highest_bid = reserve;

    for (i, &bid) in bids.iter().enumerate() {
        if bid > highest_bid {
            highest_bid = bid;
            winner = Some(i);
        }
    }

    let total_revenue: f64 = bids.iter().sum();
    let mut allocations = vec![0.0; n];
    let mut surplus = vec![0.0; n];
    if let Some(w) = winner {
        allocations[w] = 1.0;
        surplus[w] = highest_bid - bids[w]; // Only winner gets value
        // In all-pay, everyone pays their bid
    }

    AuctionResult {
        winner,
        payment: highest_bid,
        revenue: total_revenue,
        allocations,
        surplus,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_second_price_auction() {
        let result = second_price_sealed_bid(&[10.0, 8.0, 6.0], 0.0);
        assert_eq!(result.winner, Some(0));
        assert!((result.payment - 8.0).abs() < 1e-10);
        assert!((result.surplus[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_first_price_auction() {
        let result = first_price_sealed_bid(&[10.0, 8.0, 6.0], 0.0);
        assert_eq!(result.winner, Some(0));
        assert!((result.payment - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_reserve_price() {
        let result = second_price_sealed_bid(&[3.0, 8.0], 5.0);
        assert_eq!(result.winner, Some(1));
        assert!((result.payment - 5.0).abs() < 1e-10); // Pays reserve (only valid bidder)
    }

    #[test]
    fn test_no_valid_bids() {
        let result = second_price_sealed_bid(&[2.0, 3.0], 10.0);
        assert_eq!(result.winner, None);
    }

    #[test]
    fn test_english_auction() {
        let result = english_auction(&[10.0, 8.0], 0.0);
        assert_eq!(result.winner, Some(0));
        assert!((result.payment - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_auction_uniform() {
        let dist = ValueDistribution::Uniform { low: 0.0, high: 1.0 };
        let result = optimal_auction(&[0.8, 0.6, 0.4], dist);
        assert_eq!(result.winner, Some(0));
        assert!(result.payment > 0.0);
    }

    #[test]
    fn test_all_pay_auction() {
        let result = all_pay_auction(&[5.0, 3.0, 2.0], 0.0);
        assert_eq!(result.winner, Some(0));
        assert!((result.revenue - 10.0).abs() < 1e-10); // Everyone pays
    }

    #[test]
    fn test_optimal_reserve() {
        let r = optimal_reserve_uniform(5);
        assert!((r - 0.5).abs() < 1e-10);
    }
}
