"""Agent abstractions for heterogeneous coupled-market participants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable


class AgentRole(str, Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    STORAGE = "storage"
    AGGREGATOR = "aggregator"
    TRADER = "trader"
    REGULATOR = "regulator"
    MARKET_OPERATOR = "market_operator"


@dataclass
class AgentSpec:
    """Scenario-independent description of a decision-making entity."""

    agent_id: str
    role: AgentRole
    resources: list[str] = field(default_factory=list)
    markets: list[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def describe(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "resources": list(self.resources),
            "markets": list(self.markets),
            "attributes": dict(self.attributes),
        }


class ProducerAgent(AgentSpec):
    def __init__(self, agent_id: str, resources: Iterable[str], markets: Iterable[str], **attributes: Any):
        super().__init__(agent_id, AgentRole.PRODUCER, list(resources), list(markets), dict(attributes))


class ConsumerAgent(AgentSpec):
    def __init__(self, agent_id: str, resources: Iterable[str], markets: Iterable[str], **attributes: Any):
        super().__init__(agent_id, AgentRole.CONSUMER, list(resources), list(markets), dict(attributes))


class StorageAgent(AgentSpec):
    def __init__(self, agent_id: str, resources: Iterable[str], markets: Iterable[str], **attributes: Any):
        super().__init__(agent_id, AgentRole.STORAGE, list(resources), list(markets), dict(attributes))


class AggregatorAgent(AgentSpec):
    def __init__(self, agent_id: str, resources: Iterable[str], markets: Iterable[str], **attributes: Any):
        super().__init__(agent_id, AgentRole.AGGREGATOR, list(resources), list(markets), dict(attributes))


class TraderAgent(AgentSpec):
    def __init__(self, agent_id: str, resources: Iterable[str], markets: Iterable[str], **attributes: Any):
        super().__init__(agent_id, AgentRole.TRADER, list(resources), list(markets), dict(attributes))


class RegulatorAgent(AgentSpec):
    def __init__(self, agent_id: str = "regulator", **attributes: Any):
        super().__init__(agent_id, AgentRole.REGULATOR, [], [], dict(attributes))


class MarketOperator(AgentSpec):
    def __init__(self, agent_id: str = "market_operator", **attributes: Any):
        super().__init__(agent_id, AgentRole.MARKET_OPERATOR, [], [], dict(attributes))
