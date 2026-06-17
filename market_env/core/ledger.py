"""Reward ledger for multi-layer settlement and externality pricing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LedgerEntry:
    market_revenue: float = 0.0
    operating_cost: float = 0.0
    resource_cost: float = 0.0
    externality_cost: float = 0.0
    violation_penalty: float = 0.0
    certificate_or_credit_revenue: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        return (
            self.market_revenue
            - self.operating_cost
            - self.resource_cost
            - self.externality_cost
            - self.violation_penalty
            + self.certificate_or_credit_revenue
        )


class RewardLedger:
    """Accumulates scenario-independent reward components by agent."""

    def __init__(self) -> None:
        self.entries: Dict[str, LedgerEntry] = {}

    def add(self, agent_id: str, **components: float) -> LedgerEntry:
        entry = self.entries.setdefault(agent_id, LedgerEntry())
        for key, value in components.items():
            if hasattr(entry, key):
                setattr(entry, key, getattr(entry, key) + float(value))
            else:
                entry.metadata[key] = entry.metadata.get(key, 0.0) + float(value)
        return entry

    def rewards(self) -> Dict[str, float]:
        return {agent_id: entry.reward for agent_id, entry in self.entries.items()}
