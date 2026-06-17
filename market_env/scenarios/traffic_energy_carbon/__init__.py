"""Traffic demand, charging energy, and emissions scenario template."""

from market_env.scenarios.template import TemplateCoupledMarketEnv


def make_env(**kwargs):
    return TemplateCoupledMarketEnv(
        scenario_name="traffic_energy_carbon",
        markets=["mobility_market", "charging_market", "carbon_market"],
        resources=["mobility_service", "electricity", "emissions"],
        agent_roles=["ev_fleet", "traveler", "charging_operator"],
        **kwargs,
    )
