# Copyright (c) 2024. Supply Chain OpenEnv Environment.
# All rights reserved.

"""
Supply Chain & Inventory Management Environment - An OpenEnv environment
for training AI agents on real-world warehouse management with supplier negotiation.

Tasks:
  - Easy:   Single SKU reorder optimization (90 days)
  - Medium: 10 SKUs with seasonal demand (180 days)
  - Hard:   50 SKUs with supplier constraints & negotiation (365 days)

Example:
    >>> from supply_chain_env import SupplyChainEnv, SupplyChainAction
    >>>
    >>> with SupplyChainEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     result = env.step(SupplyChainAction(
    ...         reorders=[{"sku_id": "SKU_001", "quantity": 50}],
    ...         negotiations=[]
    ...     ))
    ...     print(result.observation)
"""

from .compat import Action, Observation, State
from .client import SupplyChainEnv
from .models import SupplyChainAction, SupplyChainObservation, SupplyChainState

__all__ = [
    "SupplyChainEnv",
    "SupplyChainAction",
    "SupplyChainObservation",
    "SupplyChainState",
]
