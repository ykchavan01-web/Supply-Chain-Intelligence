# Copyright (c) 2024. Supply Chain OpenEnv Environment.
# All rights reserved.

"""
Demand simulation engine producing realistic SKU demand patterns.

Supports three complexity tiers:
  - Flat:       Stable demand with Gaussian noise (Easy)
  - Seasonal:   Sinusoidal seasonality + trend + noise (Medium)
  - Correlated: Multi-product correlated demands with shocks (Hard)

All generators are seeded for deterministic, reproducible episodes.
"""

import math
from typing import Dict, List

import numpy as np


class DemandSimulator:
    """Deterministic demand sequence generator."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Easy: flat demand ────────────────────────────────────────────

    def flat(self, base: float, horizon: int) -> List[int]:
        """Flat demand with ±10 % Gaussian noise, clipped to ≥ 0."""
        noise = self.rng.normal(0, base * 0.10, horizon)
        return np.maximum(0, np.round(base + noise)).astype(int).tolist()

    # ── Medium: seasonal demand ──────────────────────────────────────

    def seasonal(
        self,
        base: float,
        horizon: int,
        amplitude: float = 0.40,
        period: int = 90,
        trend: float = 0.0,
        phase: float = 0.0,
    ) -> List[int]:
        """
        Sinusoidal seasonal demand.

        Args:
            base:      Mean daily demand
            horizon:   Number of days
            amplitude: Fraction of base for seasonal swing
            period:    Days per season cycle
            trend:     Linear daily trend added
            phase:     Phase offset (radians)
        """
        days = np.arange(horizon)
        seasonal = np.sin(2 * math.pi * days / period + phase) * (base * amplitude)
        trend_component = trend * days
        noise = self.rng.normal(0, base * 0.12, horizon)
        demand = base + seasonal + trend_component + noise
        return np.maximum(0, np.round(demand)).astype(int).tolist()

    # ── Hard: correlated multi-product demands ───────────────────────

    def correlated(
        self,
        base_levels: Dict[str, float],
        horizon: int,
        correlation_strength: float = 0.3,
        shock_probability: float = 0.02,
    ) -> Dict[str, List[int]]:
        """
        Correlated multi-SKU demands with random demand shocks.

        Products within the same index-bucket share a latent factor
        simulating category-level demand correlation.
        """
        n = len(base_levels)
        skus = list(base_levels.keys())

        # Generate a shared market-level latent noise signal
        market = self.rng.normal(0, 1, horizon)

        demands: Dict[str, List[int]] = {}
        for i, sku in enumerate(skus):
            base = base_levels[sku]
            days = np.arange(horizon)

            # Each SKU has its own random seasonal parameters
            period = self.rng.integers(60, 365)
            phase = self.rng.uniform(0, 2 * math.pi)
            amp = self.rng.uniform(0.15, 0.45)

            seasonal = np.sin(2 * math.pi * days / period + phase) * (base * amp)

            # Slight linear trend (some products grow, some shrink)
            trend = self.rng.uniform(-0.02, 0.02) * days

            # Idiosyncratic noise
            noise = self.rng.normal(0, base * 0.18, horizon)

            # Correlated component from market factor
            correlated_component = market * base * correlation_strength

            # Random demand shocks (sudden drops or spikes)
            shocks = np.zeros(horizon)
            shock_mask = self.rng.random(horizon) < shock_probability
            shock_magnitude = self.rng.choice([-0.6, 0.8], size=horizon)
            shocks[shock_mask] = base * shock_magnitude[shock_mask]

            raw = base + seasonal + trend + noise + correlated_component + shocks
            demands[sku] = np.maximum(0, np.round(raw)).astype(int).tolist()

        return demands
