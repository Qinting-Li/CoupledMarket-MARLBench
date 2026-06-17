"""Market abstractions and a simple merit-order clearing implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import numpy as np


@dataclass
class MarketResult:
    market_id: str
    prices: Dict[str, float]
    allocations: Dict[str, float]
    unmet_demand: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class Market:
    market_id: str
    resource: str
    demand: float = 0.0
    price_floor: float = 0.0
    scarcity_price: float = 0.0

    def clear_merit_order(
        self,
        bids: Mapping[str, float],
        quantities: Mapping[str, float],
        eligible_agents: Iterable[str] | None = None,
    ) -> MarketResult:
        """Clear a single-sided supply market by ascending bid price."""
        agents = list(eligible_agents or bids.keys())
        order = sorted(agents, key=lambda aid: bids.get(aid, np.inf))
        remaining = float(self.demand)
        allocations = {aid: 0.0 for aid in agents}
        clearing_price = self.price_floor

        for aid in order:
            if remaining <= 0:
                break
            alloc = min(float(quantities.get(aid, 0.0)), remaining)
            allocations[aid] = alloc
            remaining -= alloc
            if alloc > 0:
                clearing_price = max(self.price_floor, float(bids.get(aid, self.price_floor)))

        unmet = max(0.0, remaining)
        if self.demand > 0:
            clearing_price += self.scarcity_price * unmet / self.demand

        return MarketResult(
            market_id=self.market_id,
            prices={self.resource: float(clearing_price)},
            allocations=allocations,
            unmet_demand=float(unmet),
        )
