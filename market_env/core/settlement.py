"""Settlement mechanism abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from market_env.core.ledger import RewardLedger
from market_env.core.market import MarketResult


@dataclass
class SettlementMechanism:
    """Combine market allocations, costs, externalities, and penalties."""

    market_weights: Dict[str, float] = field(default_factory=dict)

    def settle(
        self,
        market_results: list[MarketResult],
        operating_costs: Dict[str, float] | None = None,
        externality_costs: Dict[str, float] | None = None,
        violation_penalties: Dict[str, float] | None = None,
        credits: Dict[str, float] | None = None,
    ) -> RewardLedger:
        operating_costs = operating_costs or {}
        externality_costs = externality_costs or {}
        violation_penalties = violation_penalties or {}
        credits = credits or {}

        ledger = RewardLedger()
        for result in market_results:
            price = next(iter(result.prices.values()), 0.0)
            weight = self.market_weights.get(result.market_id, 1.0)
            for agent_id, quantity in result.allocations.items():
                ledger.add(agent_id, market_revenue=weight * price * quantity)

        for agent_id, value in operating_costs.items():
            ledger.add(agent_id, operating_cost=value)
        for agent_id, value in externality_costs.items():
            ledger.add(agent_id, externality_cost=value)
        for agent_id, value in violation_penalties.items():
            ledger.add(agent_id, violation_penalty=value)
        for agent_id, value in credits.items():
            ledger.add(agent_id, certificate_or_credit_revenue=value)
        return ledger
