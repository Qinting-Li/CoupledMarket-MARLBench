"""Cloud-computing, energy, and carbon-cost scenario template."""

from market_env.scenarios.template import TemplateCoupledMarketEnv


def make_env(**kwargs):
    return TemplateCoupledMarketEnv(
        scenario_name="cloud_energy_carbon",
        markets=["compute_market", "electricity_market", "carbon_market"],
        resources=["compute", "electricity", "carbon_allowance"],
        agent_roles=["cloud_provider", "workload_consumer", "grid_operator"],
        **kwargs,
    )
