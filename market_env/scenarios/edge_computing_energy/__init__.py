"""Edge-computing resource, bandwidth, energy, and carbon scenario template."""

from market_env.scenarios.template import TemplateCoupledMarketEnv


def make_env(**kwargs):
    return TemplateCoupledMarketEnv(
        scenario_name="edge_computing_energy",
        markets=["compute_market", "bandwidth_market", "electricity_market"],
        resources=["edge_compute", "bandwidth", "electricity", "carbon_cost"],
        agent_roles=["edge_provider", "mobile_consumer", "aggregator"],
        **kwargs,
    )
