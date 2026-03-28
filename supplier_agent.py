# Copyright (c) 2024. Supply Chain OpenEnv Environment.
# All rights reserved.

"""
Simulated supplier agent for negotiation sub-MDP.

Each supplier has hidden parameters (cost floor, patience, discount
willingness) that the agent must discover through proposals.
Negotiations are multi-round: the agent proposes terms and the
supplier responds with accept / counter-offer / reject.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import random as _random


@dataclass
class SupplierProfile:
    """Hidden supplier parameters (not visible to agent)."""
    sku_id: str
    base_price: float
    min_price_floor: float           # Lowest price they'll accept
    base_lead_time: int              # Default lead time
    min_lead_time: int               # Fastest they can deliver
    bulk_discount_threshold: int     # Qty above which bulk discount kicks in
    bulk_discount_pct: float         # % discount for bulk
    patience: int                    # Max negotiation rounds before walking away
    flexibility: float               # 0-1, how willing to compromise


@dataclass
class NegotiationResult:
    """Outcome of one negotiation round."""
    accepted: bool
    counter_price: float
    counter_lead_time: int
    message: str
    rounds_remaining: int


class SupplierAgent:
    """
    Autonomous supplier negotiation counterpart.

    The supplier evaluates proposals against hidden cost floors and
    responds with accept, counter-offer, or reject.
    """

    def __init__(self, profile: SupplierProfile, seed: int = 42):
        self.profile = profile
        self.rng = _random.Random(seed)
        self.rounds_used = 0
        self.best_offer_price: Optional[float] = None
        self.locked = False  # If True, negotiation is over

    def evaluate_proposal(
        self,
        proposed_price: float,
        proposed_lead_time: int,
        proposed_batch_size: int,
    ) -> NegotiationResult:
        """
        Evaluate the agent's proposal and return a response.

        Returns:
            NegotiationResult with acceptance status and counter terms
        """
        p = self.profile

        if self.locked:
            return NegotiationResult(
                accepted=False,
                counter_price=p.base_price,
                counter_lead_time=p.base_lead_time,
                message="Negotiation closed. No further proposals accepted.",
                rounds_remaining=0,
            )

        self.rounds_used += 1
        remaining = max(0, p.patience - self.rounds_used)

        # ── Price evaluation ─────────────────────────────────────

        # Apply bulk discount to the floor if batch is large enough
        effective_floor = p.min_price_floor
        if proposed_batch_size >= p.bulk_discount_threshold:
            effective_floor *= (1 - p.bulk_discount_pct)

        price_acceptable = proposed_price >= effective_floor
        lead_acceptable = proposed_lead_time >= p.min_lead_time

        # ── Decision logic ───────────────────────────────────────

        if price_acceptable and lead_acceptable:
            # Good deal — accept
            self.locked = True
            self.best_offer_price = proposed_price
            return NegotiationResult(
                accepted=True,
                counter_price=proposed_price,
                counter_lead_time=proposed_lead_time,
                message=f"Deal accepted: ${proposed_price:.2f}/unit, {proposed_lead_time}d lead.",
                rounds_remaining=remaining,
            )

        if remaining <= 0:
            # Out of patience — reject and lock
            self.locked = True
            return NegotiationResult(
                accepted=False,
                counter_price=p.base_price,
                counter_lead_time=p.base_lead_time,
                message="Supplier has walked away. Reverting to base terms.",
                rounds_remaining=0,
            )

        # ── Counter-offer ────────────────────────────────────────

        # Supplier moves toward agent's position based on flexibility
        if not price_acceptable:
            gap = effective_floor - proposed_price
            counter_price = proposed_price + gap * (1 - p.flexibility * 0.5)
            counter_price = max(counter_price, effective_floor)
        else:
            counter_price = proposed_price

        if not lead_acceptable:
            counter_lead = max(p.min_lead_time, proposed_lead_time + 1)
        else:
            counter_lead = proposed_lead_time

        # Add slight randomness to counter
        jitter = self.rng.uniform(-0.02, 0.02) * counter_price
        counter_price = round(counter_price + jitter, 2)

        return NegotiationResult(
            accepted=False,
            counter_price=counter_price,
            counter_lead_time=counter_lead,
            message=f"Counter-offer: ${counter_price:.2f}/unit, {counter_lead}d lead. {remaining} rounds left.",
            rounds_remaining=remaining,
        )

    def reset(self) -> None:
        """Reset negotiation state for a new episode."""
        self.rounds_used = 0
        self.best_offer_price = None
        self.locked = False


def create_supplier_for_sku(
    sku_id: str,
    base_price: float,
    lead_time: int,
    seed: int = 42,
) -> SupplierAgent:
    """Factory: build a supplier with realistic hidden parameters."""
    rng = _random.Random(seed)
    profile = SupplierProfile(
        sku_id=sku_id,
        base_price=base_price,
        min_price_floor=base_price * rng.uniform(0.70, 0.88),
        base_lead_time=lead_time,
        min_lead_time=max(1, lead_time - rng.randint(1, 3)),
        bulk_discount_threshold=rng.randint(50, 200),
        bulk_discount_pct=rng.uniform(0.03, 0.12),
        patience=rng.randint(2, 5),
        flexibility=rng.uniform(0.2, 0.8),
    )
    return SupplierAgent(profile, seed=seed)
