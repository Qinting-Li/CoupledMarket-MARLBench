"""Backward-compatible default environment entry point.

The original project imported ``CoupledMarketEnv`` from this module. In the
benchmark layout, the concrete electricity-carbon-green scenario lives under
``market_env.scenarios.electricity_carbon_green`` and remains the default.
"""

from market_env.scenarios.electricity_carbon_green.environment import *  # noqa: F401,F403
from market_env.scenarios.electricity_carbon_green.environment import CoupledMarketEnv

__all__ = ["CoupledMarketEnv"]
