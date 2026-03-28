# Copyright (c) 2024. Supply Chain OpenEnv Environment.
# All rights reserved.

"""
OpenEnv base type stubs.

When openenv-core is installed these re-export the real types.
When running standalone these provide compatible fallback dataclasses
so the environment works out-of-the-box without the full framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # ── Standalone fallback types ────────────────────────────────

    @dataclass
    class Action:
        """Base action type."""
        pass

    @dataclass
    class Observation:
        """Base observation type."""
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class State:
        """Base episode state type."""
        episode_id: str = ""
        step_count: int = 0
