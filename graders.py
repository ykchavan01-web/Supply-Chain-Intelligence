# Copyright (c) 2024. Supply Chain OpenEnv Environment.
# All rights reserved.

"""
Programmatic graders that score agent trajectories.

Each grader computes a deterministic score ∈ [0.0, 1.0] from a
complete episode trajectory.  Graders provide partial-credit
signals — agents are never scored purely pass/fail.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TrajectoryStep:
    """One step from a recorded episode."""
    day: int
    reward: float
    info: Dict[str, Any]


class BaseGrader:
    """Base class for task graders."""

    def score(self, trajectory: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        raise NotImplementedError


class EasyGrader(BaseGrader):
    """
    Easy task grader.

    Scoring (0.0–1.0):
      60 % — Service level (fraction of demand fulfilled)
      25 % — Inventory efficiency (lower avg inventory is better)
      15 % — Profitability (cumulative reward vs. theoretical max)
    """

    def score(self, trajectory: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        total_demand = 0
        total_fulfilled = 0
        total_inventory_days = 0
        cumulative_reward = 0.0
        days = len(trajectory)

        for step in trajectory:
            info = step.get("info", {})
            total_demand += info.get("total_demand", 0)
            total_fulfilled += info.get("total_fulfilled", 0)
            total_inventory_days += info.get("total_inventory", 0)
            cumulative_reward += step.get("reward", 0.0)

        # Service level score (target: 95%)
        if total_demand > 0:
            service_level = total_fulfilled / total_demand
            sl_score = min(1.0, service_level / 0.95)
        else:
            sl_score = 1.0

        # Inventory efficiency (lower is better; normalize against max)
        if days > 0 and total_demand > 0:
            avg_inv = total_inventory_days / days
            # Optimal avg inventory ≈ lead_time × daily_demand
            optimal_inv = 3 * (total_demand / days)
            inv_ratio = optimal_inv / max(avg_inv, 1)
            inv_score = min(1.0, inv_ratio)
        else:
            inv_score = 0.5

        # Profitability score
        max_possible = total_demand * 22.0  # sell_price from easy config
        if max_possible > 0:
            profit_score = min(1.0, max(0.0, cumulative_reward / (max_possible * 0.3)))
        else:
            profit_score = 0.0

        final = 0.60 * sl_score + 0.25 * inv_score + 0.15 * profit_score
        return round(max(0.0, min(1.0, final)), 4)


class MediumGrader(BaseGrader):
    """
    Medium task grader.

    Scoring (0.0–1.0):
      40 % — Aggregate service level across all 10 SKUs
      30 % — Per-SKU balance (penalty for high variance in SL across SKUs)
      20 % — Cost efficiency (holding costs kept reasonable)
      10 % — Cumulative reward positivity
    """

    def score(self, trajectory: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        total_demand = 0
        total_fulfilled = 0
        total_holding = 0.0
        cumulative_reward = 0.0
        per_sku_demand: Dict[str, int] = {}
        per_sku_fulfilled: Dict[str, int] = {}

        for step in trajectory:
            info = step.get("info", {})
            total_demand += info.get("total_demand", 0)
            total_fulfilled += info.get("total_fulfilled", 0)
            total_holding += info.get("total_holding_cost", 0.0)
            cumulative_reward += step.get("reward", 0.0)

            sku_details = info.get("sku_details", {})
            for sku, det in sku_details.items():
                per_sku_demand[sku] = per_sku_demand.get(sku, 0) + det.get("demand", 0)
                per_sku_fulfilled[sku] = per_sku_fulfilled.get(sku, 0) + det.get("fulfilled", 0)

        # Aggregate service level
        if total_demand > 0:
            agg_sl = total_fulfilled / total_demand
            sl_score = min(1.0, agg_sl / 0.92)
        else:
            sl_score = 1.0

        # Per-SKU balance
        sku_sls = []
        for sku in per_sku_demand:
            d = per_sku_demand[sku]
            f = per_sku_fulfilled.get(sku, 0)
            sku_sls.append(f / d if d > 0 else 1.0)

        if sku_sls:
            import statistics
            mean_sl = statistics.mean(sku_sls)
            stdev_sl = statistics.stdev(sku_sls) if len(sku_sls) > 1 else 0
            balance_score = max(0.0, 1.0 - stdev_sl * 3)
        else:
            balance_score = 0.5

        # Holding cost efficiency
        if total_demand > 0:
            holding_per_unit = total_holding / total_demand
            cost_score = max(0.0, 1.0 - holding_per_unit / 5.0)
        else:
            cost_score = 0.5

        # Profitability
        profit_score = 1.0 if cumulative_reward > 0 else max(0.0, 0.5 + cumulative_reward / 100_000)

        final = 0.40 * sl_score + 0.30 * balance_score + 0.20 * cost_score + 0.10 * profit_score
        return round(max(0.0, min(1.0, final)), 4)


class HardGrader(BaseGrader):
    """
    Hard task grader.

    Scoring (0.0–1.0):
      35 % — Aggregate service level (target 90%)
      25 % — Net profit vs naive heuristic baseline
      20 % — Negotiation effectiveness (savings achieved)
      10 % — Stockout rate below 3%
      10 % — Budget management (didn't go bankrupt)
    """

    def score(self, trajectory: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
        total_demand = 0
        total_fulfilled = 0
        total_stockouts = 0
        cumulative_reward = 0.0
        total_neg_savings = 0.0
        total_negotiations = 0
        successful_negotiations = 0
        days = len(trajectory)
        final_budget = config.get("initial_budget", 500_000)

        for step in trajectory:
            info = step.get("info", {})
            total_demand += info.get("total_demand", 0)
            total_fulfilled += info.get("total_fulfilled", 0)
            total_stockouts += info.get("total_stockouts", 0)
            cumulative_reward += step.get("reward", 0.0)
            total_neg_savings += info.get("negotiation_savings", 0.0)
            total_negotiations += info.get("negotiations_attempted", 0)
            successful_negotiations += info.get("negotiations_succeeded", 0)
            final_budget = info.get("budget", final_budget)

        # Service level
        if total_demand > 0:
            sl = total_fulfilled / total_demand
            sl_score = min(1.0, sl / 0.90)
        else:
            sl_score = 1.0

        # Net profit vs baseline (simple heuristic would yield ~40% of max)
        max_revenue = total_demand * 80  # approximate average sell_price
        if max_revenue > 0:
            profit_ratio = cumulative_reward / (max_revenue * 0.25)
            profit_score = min(1.0, max(0.0, profit_ratio))
        else:
            profit_score = 0.0

        # Negotiation effectiveness
        if total_negotiations > 0:
            neg_success_rate = successful_negotiations / total_negotiations
            neg_savings_score = min(1.0, total_neg_savings / 10_000)
            neg_score = 0.5 * neg_success_rate + 0.5 * neg_savings_score
        else:
            neg_score = 0.3  # Partial credit for not negotiating at all

        # Stockout rate
        if total_demand > 0:
            stockout_rate = total_stockouts / total_demand
            stockout_score = 1.0 if stockout_rate < 0.03 else max(0.0, 1.0 - (stockout_rate - 0.03) * 10)
        else:
            stockout_score = 1.0

        # Budget management
        budget_score = 1.0 if final_budget > 0 else 0.0

        final = (
            0.35 * sl_score
            + 0.25 * profit_score
            + 0.20 * neg_score
            + 0.10 * stockout_score
            + 0.10 * budget_score
        )
        return round(max(0.0, min(1.0, final)), 4)


GRADER_REGISTRY = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}
