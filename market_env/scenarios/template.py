"""Lightweight template environment for new coupled-market scenarios."""

from __future__ import annotations

import numpy as np

from market_env.core import (
    AgentSpec,
    AgentRole,
    CoupledMarketSimulator,
    Market,
    Resource,
    SettlementMechanism,
)


class TemplateCoupledMarketEnv(CoupledMarketSimulator):
    """A small runnable skeleton for adding new benchmark scenarios.

    The template exposes the same Gym-like API as the concrete electricity
    scenario. It is intentionally simple: it validates the benchmark interface
    and gives new scenarios a working starting point before domain-specific
    clearing, externality pricing, and constraints are implemented.
    """

    def __init__(
        self,
        scenario_name: str,
        markets: list[str],
        resources: list[str],
        agent_roles: list[str],
        num_periods: int = 24,
        seed: int = 42,
    ) -> None:
        self.scenario_name = scenario_name
        self.num_periods = int(num_periods)
        self.rng = np.random.default_rng(seed)
        agents = [
            AgentSpec(
                agent_id=f"{role}_{idx + 1}",
                role=AgentRole.PRODUCER if idx == 0 else AgentRole.CONSUMER,
                resources=list(resources),
                markets=list(markets),
                attributes={"template_role": role},
            )
            for idx, role in enumerate(agent_roles)
        ]
        market_objs = [
            Market(market_id=name, resource=resources[min(i, len(resources) - 1)], demand=100.0, price_floor=1.0)
            for i, name in enumerate(markets)
        ]
        resource_objs = [Resource(name=name, unit="normalized") for name in resources]
        super().__init__(
            agents=agents,
            markets=market_objs,
            resources=resource_objs,
            settlement=SettlementMechanism(),
            metadata={"scenario_name": scenario_name, "status": "template"},
        )
        self.state_dim = 4 + self.num_agents
        self.action_dim = 2
        self.reset()

    def reset(self):
        self.t = 0
        self.prev_allocations = np.zeros(self.num_agents, dtype=float)
        return self._state()

    def step(self, actions):
        quantities = np.zeros(self.num_agents, dtype=float)
        bids = np.zeros(self.num_agents, dtype=float)
        for i in range(self.num_agents):
            action = np.asarray(actions[i], dtype=float)
            quantities[i] = 50.0 * np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0)
            bids[i] = 20.0 + 80.0 * np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)

        demand = 80.0 + 20.0 * np.sin(2.0 * np.pi * self.t / max(self.num_periods, 1))
        market = self.markets[0]
        market.demand = float(demand)
        result = market.clear_merit_order(
            {agent.agent_id: bids[i] for i, agent in enumerate(self.agents)},
            {agent.agent_id: quantities[i] for i, agent in enumerate(self.agents)},
        )
        rewards_by_id = self.settlement.settle([result]).rewards()
        rewards = np.array([rewards_by_id.get(agent.agent_id, 0.0) for agent in self.agents], dtype=float)
        self.prev_allocations = np.array(
            [result.allocations.get(agent.agent_id, 0.0) for agent in self.agents], dtype=float
        )
        self.t += 1
        done = self.t >= self.num_periods
        info = {
            "scenario_name": self.scenario_name,
            "clearing_price": next(iter(result.prices.values())),
            "outputs": self.prev_allocations.copy(),
            "bids": bids,
            "quantities": quantities,
        }
        return (np.zeros(self.state_dim) if done else self._state()), rewards, done, info

    def _state(self):
        state = np.zeros(self.state_dim, dtype=float)
        state[0] = self.t / max(self.num_periods - 1, 1)
        state[1:1 + self.num_agents] = self.prev_allocations / 100.0
        state[-3:] = [1.0, 0.0, 0.0]
        return state
