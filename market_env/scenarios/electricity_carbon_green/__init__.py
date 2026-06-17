"""Electricity-carbon-green certificate scenario."""

from market_env.scenarios.electricity_carbon_green.environment import CoupledMarketEnv


def make_env(**kwargs) -> CoupledMarketEnv:
    return CoupledMarketEnv(**kwargs)


__all__ = ["CoupledMarketEnv", "make_env"]
