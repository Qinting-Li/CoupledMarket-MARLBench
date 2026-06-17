"""Base simulator contract for coupled-market MARL scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from market_env.core.agent import AgentSpec
from market_env.core.market import Market
from market_env.core.resource import Resource
from market_env.core.settlement import SettlementMechanism


@dataclass
class CoupledMarketSimulator:
    """Scenario metadata and common Gym-like interface shape."""

    agents: list[AgentSpec]
    markets: list[Market]
    resources: list[Resource]
    settlement: SettlementMechanism = field(default_factory=SettlementMechanism)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def scenario_description(self) -> dict[str, Any]:
        return {
            "agents": [agent.describe() for agent in self.agents],
            "markets": [market.market_id for market in self.markets],
            "resources": [resource.name for resource in self.resources],
            "metadata": dict(self.metadata),
        }
