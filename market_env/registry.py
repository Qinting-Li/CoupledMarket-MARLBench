"""Scenario registry for CoupledMarket-MARLBench."""

from __future__ import annotations

from typing import Callable, Dict


_SCENARIOS: Dict[str, Callable[..., object]] = {}


def register_scenario(name: str, factory: Callable[..., object]) -> None:
    """Register an environment factory under a stable scenario name."""
    if not name:
        raise ValueError("scenario name must be non-empty")
    _SCENARIOS[name] = factory


def list_scenarios() -> list[str]:
    """Return available scenario identifiers."""
    _ensure_builtin_scenarios()
    return sorted(_SCENARIOS)


def make_env(name: str = "electricity_carbon_green", **kwargs):
    """Build a scenario environment by name."""
    _ensure_builtin_scenarios()
    if name not in _SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. Available: {list_scenarios()}")
    return _SCENARIOS[name](**kwargs)


def _ensure_builtin_scenarios() -> None:
    if _SCENARIOS:
        return
    from market_env.scenarios.electricity_carbon_green import make_env as ecg_factory
    from market_env.scenarios.cloud_energy_carbon import make_env as cloud_factory
    from market_env.scenarios.edge_computing_energy import make_env as edge_factory
    from market_env.scenarios.logistics_carbon_credit import make_env as logistics_factory
    from market_env.scenarios.traffic_energy_carbon import make_env as traffic_factory

    register_scenario("electricity_carbon_green", ecg_factory)
    register_scenario("cloud_energy_carbon", cloud_factory)
    register_scenario("edge_computing_energy", edge_factory)
    register_scenario("logistics_carbon_credit", logistics_factory)
    register_scenario("traffic_energy_carbon", traffic_factory)
